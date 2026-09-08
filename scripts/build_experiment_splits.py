from pathlib import Path
import json

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]

MASTER_PATH = (
    ROOT
    / "data"
    / "processed"
    / "metropt3_model_master.parquet"
)

EVENT_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "metropt3_failure_events.csv"
)

CONFIG_PATH = (
    ROOT
    / "configs"
    / "experiment_protocol.yaml"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "splits"
)

TABLE_DIR = (
    ROOT
    / "artifacts"
    / "tables"
)

METRIC_DIR = (
    ROOT
    / "artifacts"
    / "metrics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRIC_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 82)
print("AIRGUARD-LK — CHRONOLOGICAL EVENT SPLIT GENERATOR")
print("=" * 82)


# ============================================================
# LOAD
# ============================================================

master = pd.read_parquet(
    MASTER_PATH
)

master["timestamp"] = pd.to_datetime(
    master["timestamp"]
)

master = (
    master.sort_values("timestamp")
          .reset_index(drop=True)
)

events = pd.read_csv(
    EVENT_PATH,
    parse_dates=[
        "start_time",
        "end_time",
    ],
)

events = (
    events.sort_values("start_time")
          .reset_index(drop=True)
)

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8",
) as f:

    config = yaml.safe_load(f)


HORIZONS = (
    config["target"]
          ["candidate_warning_horizons_minutes"]
)

EVENT_NEIGHBORHOOD = int(
    config["target"]
          ["event_neighborhood_minutes"]
)

ELIGIBLE_COLUMN = (
    "quality__model_feature_eligible"
)


event_lookup = (
    events.set_index("event_id")
          .to_dict("index")
)


# ============================================================
# HELPERS
# ============================================================

def event_info(event_id):

    if event_id not in event_lookup:
        raise KeyError(
            f"Unknown event: {event_id}"
        )

    item = event_lookup[event_id]

    return {
        "start":
            pd.Timestamp(
                item["start_time"]
            ),

        "end":
            pd.Timestamp(
                item["end_time"]
            ),
    }


def mark_training_roles(
    data,
    training_events,
    horizon,
    train_end,
):

    """
    Returns roles for timestamps before train_end.

    Eligible rows begin as train_negative.

    Around every known training event:
      [T-360, T-H)  -> ambiguous
      [T-H, T)      -> positive
      [T, end]      -> failure
      (end, end+360] -> recovery

    Ineligible rows remain explicitly marked.
    """

    result = pd.DataFrame({
        "timestamp":
            data["timestamp"],
    })

    in_train_period = (
        data["timestamp"] < train_end
    )

    eligible = (
        data[ELIGIBLE_COLUMN] == 1
    )

    result["role"] = "outside"

    result.loc[
        in_train_period
        &
        ~eligible,
        "role"
    ] = "train_ineligible"

    result.loc[
        in_train_period
        &
        eligible,
        "role"
    ] = "train_negative"


    for event_id in training_events:

        info = event_info(
            event_id
        )

        start = info["start"]
        end = info["end"]

        positive_start = (
            start
            -
            pd.Timedelta(
                minutes=horizon
            )
        )

        ambiguous_start = (
            start
            -
            pd.Timedelta(
                minutes=EVENT_NEIGHBORHOOD
            )
        )

        recovery_end = (
            end
            +
            pd.Timedelta(
                minutes=EVENT_NEIGHBORHOOD
            )
        )


        # ----------------------------------------------
        # Ambiguous pre-event zone
        # ----------------------------------------------

        mask = (
            in_train_period
            &
            eligible
            &
            (
                data["timestamp"]
                >= ambiguous_start
            )
            &
            (
                data["timestamp"]
                < positive_start
            )
        )

        result.loc[
            mask,
            "role"
        ] = "train_ambiguous"


        # ----------------------------------------------
        # Positive warning window
        # ----------------------------------------------

        mask = (
            in_train_period
            &
            eligible
            &
            (
                data["timestamp"]
                >= positive_start
            )
            &
            (
                data["timestamp"]
                < start
            )
        )

        result.loc[
            mask,
            "role"
        ] = "train_positive"


        # ----------------------------------------------
        # Documented failure
        # ----------------------------------------------

        mask = (
            in_train_period
            &
            (
                data["timestamp"]
                >= start
            )
            &
            (
                data["timestamp"]
                <= end
            )
        )

        result.loc[
            mask,
            "role"
        ] = "train_failure"


        # ----------------------------------------------
        # Post-failure recovery
        # ----------------------------------------------

        mask = (
            in_train_period
            &
            (
                data["timestamp"]
                > end
            )
            &
            (
                data["timestamp"]
                <= recovery_end
            )
        )

        result.loc[
            mask,
            "role"
        ] = "train_recovery"


    return result


def mark_evaluation_roles(
    data,
    roles,
    event_id,
    horizon,
    evaluation_start,
    role_prefix,
):

    """
    Evaluation contains:
        normal interval
        ambiguous 360m -> H zone
        positive H-minute warning interval

    The documented failure itself is not part of the
    binary early-warning evaluation.
    """

    info = event_info(
        event_id
    )

    start = info["start"]

    positive_start = (
        start
        -
        pd.Timedelta(
            minutes=horizon
        )
    )

    ambiguous_start = (
        start
        -
        pd.Timedelta(
            minutes=EVENT_NEIGHBORHOOD
        )
    )

    evaluation_period = (
        (
            data["timestamp"]
            >= evaluation_start
        )
        &
        (
            data["timestamp"]
            < start
        )
    )

    eligible = (
        data[ELIGIBLE_COLUMN] == 1
    )


    # ----------------------------------------------
    # Ineligible evaluation rows
    # ----------------------------------------------

    roles.loc[
        evaluation_period
        &
        ~eligible,
        "role"
    ] = (
        f"{role_prefix}_ineligible"
    )


    # ----------------------------------------------
    # Normal evaluation period
    # ----------------------------------------------

    normal_mask = (
        evaluation_period
        &
        eligible
        &
        (
            data["timestamp"]
            < ambiguous_start
        )
    )

    roles.loc[
        normal_mask,
        "role"
    ] = (
        f"{role_prefix}_negative"
    )


    # ----------------------------------------------
    # Ambiguous early pre-event period
    # ----------------------------------------------

    ambiguous_mask = (
        evaluation_period
        &
        eligible
        &
        (
            data["timestamp"]
            >= ambiguous_start
        )
        &
        (
            data["timestamp"]
            < positive_start
        )
    )

    roles.loc[
        ambiguous_mask,
        "role"
    ] = (
        f"{role_prefix}_ambiguous"
    )


    # ----------------------------------------------
    # Positive warning period
    # ----------------------------------------------

    positive_mask = (
        evaluation_period
        &
        eligible
        &
        (
            data["timestamp"]
            >= positive_start
        )
        &
        (
            data["timestamp"]
            < start
        )
    )

    roles.loc[
        positive_mask,
        "role"
    ] = (
        f"{role_prefix}_positive"
    )


    return roles


# ============================================================
# BUILD SPLITS
# ============================================================

summary_rows = []
boundary_rows = []
membership_parts = []


# ------------------------------------------------------------
# DEVELOPMENT FOLDS
# ------------------------------------------------------------

for fold in config["development_folds"]:

    fold_name = fold["name"]

    training_events = (
        fold["training_events"]
    )

    validation_event = (
        fold["validation_event"]
    )

    last_training_event = (
        training_events[-1]
    )

    last_training_info = (
        event_info(
            last_training_event
        )
    )

    validation_start = (
        last_training_info["end"]
        +
        pd.Timedelta(
            minutes=EVENT_NEIGHBORHOOD
        )
    )

    validation_info = (
        event_info(
            validation_event
        )
    )

    boundary_rows.append({
        "split": fold_name,
        "type": "development",
        "training_events":
            ",".join(training_events),
        "evaluation_event":
            validation_event,
        "training_end":
            validation_start,
        "evaluation_start":
            validation_start,
        "evaluation_end":
            validation_info["start"],
    })


    for horizon in HORIZONS:

        roles = mark_training_roles(
            master,
            training_events,
            horizon,
            validation_start,
        )

        roles = mark_evaluation_roles(
            master,
            roles,
            validation_event,
            horizon,
            validation_start,
            "validation",
        )

        roles["split"] = fold_name
        roles["horizon_minutes"] = horizon

        active = roles[
            roles["role"] != "outside"
        ].copy()

        membership_parts.append(
            active
        )


        counts = (
            active["role"]
            .value_counts()
        )

        for role, count in counts.items():

            summary_rows.append({
                "split":
                    fold_name,

                "split_type":
                    "development",

                "horizon_minutes":
                    horizon,

                "role":
                    role,

                "rows":
                    int(count),
            })


# ------------------------------------------------------------
# FINAL HOLDOUT SPLIT
# ------------------------------------------------------------

final_config = (
    config["final_fit"]
)

training_events = (
    final_config[
        "training_events"
    ]
)

holdout_event = (
    final_config[
        "holdout_event"
    ]
)

last_training_event = (
    training_events[-1]
)

last_training_info = (
    event_info(
        last_training_event
    )
)

holdout_start = (
    last_training_info["end"]
    +
    pd.Timedelta(
        minutes=EVENT_NEIGHBORHOOD
    )
)

holdout_info = (
    event_info(
        holdout_event
    )
)

boundary_rows.append({
    "split":
        "final_F04",

    "type":
        "final_holdout",

    "training_events":
        ",".join(training_events),

    "evaluation_event":
        holdout_event,

    "training_end":
        holdout_start,

    "evaluation_start":
        holdout_start,

    "evaluation_end":
        holdout_info["start"],
})


for horizon in HORIZONS:

    roles = mark_training_roles(
        master,
        training_events,
        horizon,
        holdout_start,
    )

    roles = mark_evaluation_roles(
        master,
        roles,
        holdout_event,
        horizon,
        holdout_start,
        "holdout",
    )

    roles["split"] = "final_F04"
    roles["horizon_minutes"] = horizon

    active = roles[
        roles["role"] != "outside"
    ].copy()

    membership_parts.append(
        active
    )


    counts = (
        active["role"]
        .value_counts()
    )

    for role, count in counts.items():

        summary_rows.append({
            "split":
                "final_F04",

            "split_type":
                "final_holdout",

            "horizon_minutes":
                horizon,

            "role":
                role,

            "rows":
                int(count),
        })


# ============================================================
# SAVE MEMBERSHIP
# ============================================================

membership = pd.concat(
    membership_parts,
    ignore_index=True,
)

membership.to_parquet(
    OUTPUT_DIR
    / "split_membership.parquet",
    index=False,
)


# ============================================================
# SAVE SUMMARIES
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    TABLE_DIR
    / "metropt3_split_summary.csv",
    index=False,
)

boundaries = pd.DataFrame(
    boundary_rows
)

boundaries.to_csv(
    TABLE_DIR
    / "metropt3_split_boundaries.csv",
    index=False,
)


# ============================================================
# SANITY CHECKS
# ============================================================

for split_name in membership["split"].unique():

    for horizon in HORIZONS:

        subset = membership[
            (
                membership["split"]
                == split_name
            )
            &
            (
                membership[
                    "horizon_minutes"
                ]
                == horizon
            )
        ]

        train_positive = int(
            subset[
                "role"
            ]
            .eq(
                "train_positive"
            )
            .sum()
        )

        if train_positive == 0:
            raise RuntimeError(
                f"No positive training rows "
                f"for {split_name}, "
                f"{horizon}m"
            )


# ============================================================
# JSON SUMMARY
# ============================================================

metric_summary = {
    "candidate_horizons":
        HORIZONS,

    "event_neighborhood_minutes":
        EVENT_NEIGHBORHOOD,

    "development_folds":
        [
            item["name"]
            for item
            in config[
                "development_folds"
            ]
        ],

    "final_holdout_split":
        "final_F04",

    "final_holdout_event":
        holdout_event,

    "membership_rows":
        int(len(membership)),
}


with open(
    METRIC_DIR
    / "metropt3_split_build.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metric_summary,
        f,
        indent=2
    )


# ============================================================
# CONSOLE
# ============================================================

print("\n" + "=" * 82)
print("SPLIT BOUNDARIES")
print("=" * 82)

print(
    boundaries.to_string(
        index=False
    )
)

print("\n" + "=" * 82)
print("KEY TRAIN / EVALUATION COUNTS")
print("=" * 82)

important_roles = [
    "train_negative",
    "train_positive",
    "validation_negative",
    "validation_positive",
    "holdout_negative",
    "holdout_positive",
]

important = summary[
    summary["role"].isin(
        important_roles
    )
]

pivot = (
    important
    .pivot_table(
        index=[
            "split",
            "horizon_minutes",
        ],
        columns="role",
        values="rows",
        fill_value=0,
    )
)

print(
    pivot.to_string()
)

print("\nSaved:")
print(
    OUTPUT_DIR
    / "split_membership.parquet"
)

print("\nSPLIT GENERATION COMPLETE")
