from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import yaml

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    auc,
    precision_recall_curve,
    roc_auc_score,
)

from lightgbm import LGBMClassifier


# ============================================================
# PATHS
# ============================================================

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
    / "fault_state_protocol.yaml"
)

METRIC_DIR = (
    ROOT
    / "artifacts"
    / "metrics"
)

TABLE_DIR = (
    ROOT
    / "artifacts"
    / "tables"
)

METRIC_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FEATURES
# ============================================================

ANALOG = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
]

DIGITAL = [
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
]


CURRENT_FEATURES = []

for sensor in ANALOG:

    for stat in [
        "mean",
        "std",
        "min",
        "max",
        "last",
    ]:

        CURRENT_FEATURES.append(
            f"{sensor}_{stat}"
        )


for sensor in DIGITAL:

    for stat in [
        "mean",
        "last",
    ]:

        CURRENT_FEATURES.append(
            f"{sensor}_{stat}"
        )

    CURRENT_FEATURES.append(
        f"{sensor}_transition_sum"
    )


CURRENT_FEATURES += [
    "COMP_MPG_mismatch_mean",
    "COMP_DV_equal_mean",

    "TP3_Reservoir_abs_diff_mean",
    "TP3_Reservoir_abs_diff_max",

    "TP3_minus_Reservoirs_mean",
    "TP2_minus_TP3_mean",
    "H1_minus_TP3_mean",
    "TP2_minus_H1_mean",

    "Motor_current_x_COMP0",
    "Motor_current_x_COMP1",
    "TP2_x_COMP0",
    "H1_x_COMP1",
]


STATIC_FEATURES = [
    f"{sensor}_mean"
    for sensor in ANALOG
]


# ============================================================
# LOAD CONFIG
# ============================================================

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8",
) as f:

    config = yaml.safe_load(f)


SEED = int(
    config["random_seed"]
)

PRE_EXCLUSION = int(
    config["task"][
        "pre_event_exclusion_minutes"
    ]
)

RECOVERY = int(
    config["task"][
        "post_event_recovery_minutes"
    ]
)

PERSISTENCE = int(
    config["alerting"][
        "persistence_minutes"
    ]
)

QUANTILES = [
    float(x)
    for x in config[
        "alerting"
    ][
        "training_normal_quantiles"
    ]
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 92)
print("AIRGUARD-LK — CHRONOLOGICAL FAULT-STATE BENCHMARK")
print("=" * 92)


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


ALL_FEATURES = [
    c
    for c in master.columns
    if (
        c != "timestamp"
        and
        not c.startswith("quality__")
        and
        not c.startswith("target__")
    )
]


for feature in CURRENT_FEATURES:

    if feature not in ALL_FEATURES:

        raise RuntimeError(
            f"Missing feature: {feature}"
        )


ELIGIBLE = (
    "quality__model_feature_eligible"
)


event_lookup = (
    events
    .set_index("event_id")
    .to_dict("index")
)


def event_info(event_id):

    item = event_lookup[event_id]

    return (
        pd.Timestamp(
            item["start_time"]
        ),
        pd.Timestamp(
            item["end_time"]
        ),
    )


print(
    f"Current features : "
    f"{len(CURRENT_FEATURES)}"
)

print(
    f"Temporal features: "
    f"{len(ALL_FEATURES)}"
)


# ============================================================
# HELPERS
# ============================================================

def ranking_metrics(
    y_true,
    scores,
):

    ap = average_precision_score(
        y_true,
        scores,
    )

    precision, recall, _ = (
        precision_recall_curve(
            y_true,
            scores,
        )
    )

    pr_auc = auc(
        recall,
        precision,
    )

    roc_auc = (
        roc_auc_score(
            y_true,
            scores,
        )
        if len(
            np.unique(y_true)
        ) == 2
        else np.nan
    )

    prevalence = float(
        np.mean(y_true)
    )

    lift = (
        ap / prevalence
        if prevalence > 0
        else np.nan
    )

    return {
        "pr_auc":
            float(pr_auc),

        "average_precision":
            float(ap),

        "roc_auc":
            float(roc_auc),

        "positive_prevalence":
            prevalence,

        "ap_lift_over_prevalence":
            float(lift),
    }


def persistent_alert_flags(
    timestamps,
    scores,
    threshold,
    persistence,
):

    timestamps = (
        pd.Series(
            pd.to_datetime(
                timestamps
            )
        )
        .reset_index(drop=True)
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    flags = np.zeros(
        len(scores),
        dtype=bool,
    )

    streak = 0
    previous_time = None

    for i in range(
        len(scores)
    ):

        current_time = (
            timestamps.iloc[i]
        )

        if previous_time is None:

            consecutive = True

        else:

            delta = (
                current_time
                -
                previous_time
            ).total_seconds()

            consecutive = (
                delta == 60
            )

        if not consecutive:

            streak = 0

        if scores[i] >= threshold:

            streak += 1

        else:

            streak = 0

        if streak >= persistence:

            flags[i] = True

        previous_time = (
            current_time
        )

    return flags


def count_alert_episodes(
    timestamps,
    flags,
):

    temp = pd.DataFrame({
        "timestamp":
            pd.to_datetime(
                timestamps
            ),

        "alert":
            flags,
    })

    alert_times = (
        temp.loc[
            temp["alert"],
            "timestamp",
        ]
        .sort_values()
        .reset_index(drop=True)
    )

    if len(alert_times) == 0:

        return 0

    episodes = 1

    for i in range(
        1,
        len(alert_times)
    ):

        delta = (
            alert_times.iloc[i]
            -
            alert_times.iloc[i - 1]
        ).total_seconds()

        if delta > 60:

            episodes += 1

    return episodes


def operational_metrics(
    evaluation,
    scores,
    threshold,
    event_start,
):

    temp = evaluation[
        [
            "timestamp",
            "role",
        ]
    ].copy()

    temp["score"] = scores

    temp = (
        temp
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    temp["alert"] = (
        persistent_alert_flags(
            temp["timestamp"],
            temp["score"],
            threshold,
            PERSISTENCE,
        )
    )


    # ========================================================
    # FALSE ALARMS
    # ========================================================

    negative = temp[
        temp["role"]
        == "evaluation_negative"
    ]

    false_alarm_episodes = (
        count_alert_episodes(
            negative["timestamp"],
            negative["alert"],
        )
    )

    negative_days = (
        len(negative)
        / 1440.0
    )

    false_alarms_per_day = (
        false_alarm_episodes
        / negative_days
        if negative_days > 0
        else np.nan
    )


    # ========================================================
    # EVENT DETECTION
    # ========================================================

    failure = temp[
        temp["role"]
        == "evaluation_failure"
    ]

    failure_alerts = failure[
        failure["alert"]
    ]

    detected = (
        len(failure_alerts) > 0
    )

    if detected:

        first_alert = (
            failure_alerts[
                "timestamp"
            ].min()
        )

        detection_delay = (
            first_alert
            -
            event_start
        ).total_seconds() / 60.0

    else:

        first_alert = pd.NaT
        detection_delay = np.nan


    # ========================================================
    # PRE-EVENT DIAGNOSTIC
    # ========================================================

    pre_event = temp[
        temp["role"]
        == "evaluation_pre_event_context"
    ]

    pre_alerts = pre_event[
        pre_event["alert"]
    ]

    if len(pre_alerts):

        earliest_pre_alert = (
            pre_alerts[
                "timestamp"
            ].min()
        )

        early_warning_minutes = (
            event_start
            -
            earliest_pre_alert
        ).total_seconds() / 60.0

    else:

        early_warning_minutes = np.nan


    return {
        "event_detected":
            int(detected),

        "detection_delay_minutes":
            float(detection_delay)
            if np.isfinite(
                detection_delay
            )
            else np.nan,

        "first_alert_timestamp":
            first_alert,

        "false_alarm_episodes":
            int(
                false_alarm_episodes
            ),

        "negative_days":
            float(
                negative_days
            ),

        "false_alarm_episodes_per_day":
            float(
                false_alarms_per_day
            ),

        "pre_event_warning_minutes":
            float(
                early_warning_minutes
            )
            if np.isfinite(
                early_warning_minutes
            )
            else np.nan,
    }


def build_lightgbm():

    return LGBMClassifier(
        objective="binary",

        n_estimators=350,

        learning_rate=0.03,

        num_leaves=15,

        max_depth=5,

        min_child_samples=20,

        subsample=0.8,

        colsample_bytree=0.8,

        reg_lambda=1.0,

        class_weight="balanced",

        random_state=SEED,

        n_jobs=-1,

        verbosity=-1,
    )


def build_isolation_forest():

    return IsolationForest(
        n_estimators=300,

        max_samples=8192,

        contamination="auto",

        random_state=SEED,

        n_jobs=-1,
    )


# ============================================================
# DEVELOPMENT FOLDS ONLY
# ============================================================

FOLDS = [
    {
        "name": "fault_F02",
        "training_events": [
            "F01",
        ],
        "evaluation_event": "F02",
    },

    {
        "name": "fault_F03",
        "training_events": [
            "F01",
            "F02",
        ],
        "evaluation_event": "F03",
    },
]


role_parts = []
ranking_rows = []
operational_rows = []
prediction_parts = []


# ============================================================
# BUILD + RUN EACH FOLD
# ============================================================

for fold in FOLDS:

    split_name = (
        fold["name"]
    )

    training_events = (
        fold["training_events"]
    )

    evaluation_event = (
        fold["evaluation_event"]
    )


    last_training_event = (
        training_events[-1]
    )

    _, last_training_end = (
        event_info(
            last_training_event
        )
    )


    evaluation_period_start = (
        last_training_end
        +
        pd.Timedelta(
            minutes=RECOVERY
        )
    )


    event_start, event_end = (
        event_info(
            evaluation_event
        )
    )


    pre_event_start = (
        event_start
        -
        pd.Timedelta(
            minutes=PRE_EXCLUSION
        )
    )


    # Causal convention:
    #
    # feature timestamp T represents data available
    # through the preceding minute.
    #
    # If fault begins at event_start, first feature
    # containing fault measurements is event_start + 1 min.
    evaluation_fault_end = (
        event_end
        +
        pd.Timedelta(
            minutes=1
        )
    )


    roles = pd.DataFrame({
        "timestamp":
            master["timestamp"],
    })

    roles["role"] = "outside"


    eligible = (
        master[ELIGIBLE] == 1
    )


    # ========================================================
    # TRAINING PERIOD
    # ========================================================

    train_period = (
        master["timestamp"]
        < evaluation_period_start
    )


    roles.loc[
        train_period
        &
        eligible,
        "role"
    ] = "train_negative"


    roles.loc[
        train_period
        &
        ~eligible,
        "role"
    ] = "train_ineligible"


    for event_id in training_events:

        start, end = (
            event_info(
                event_id
            )
        )


        pre_start = (
            start
            -
            pd.Timedelta(
                minutes=PRE_EXCLUSION
            )
        )


        fault_end = (
            end
            +
            pd.Timedelta(
                minutes=1
            )
        )


        recovery_end = (
            end
            +
            pd.Timedelta(
                minutes=RECOVERY
            )
        )


        # ----------------------------------------------------
        # PRE-EVENT CONTEXT
        # ----------------------------------------------------

        mask = (
            train_period
            &
            eligible
            &
            (
                master["timestamp"]
                > pre_start
            )
            &
            (
                master["timestamp"]
                <= start
            )
        )

        roles.loc[
            mask,
            "role"
        ] = "train_pre_event_context"


        # ----------------------------------------------------
        # DOCUMENTED FAULT STATE
        # ----------------------------------------------------

        mask = (
            train_period
            &
            eligible
            &
            (
                master["timestamp"]
                > start
            )
            &
            (
                master["timestamp"]
                <= fault_end
            )
        )

        roles.loc[
            mask,
            "role"
        ] = "train_failure"


        # ----------------------------------------------------
        # RECOVERY
        # ----------------------------------------------------

        mask = (
            train_period
            &
            (
                master["timestamp"]
                > fault_end
            )
            &
            (
                master["timestamp"]
                <= recovery_end
            )
        )

        roles.loc[
            mask,
            "role"
        ] = "train_recovery"


    # ========================================================
    # EVALUATION NORMAL
    # ========================================================

    mask = (
        eligible
        &
        (
            master["timestamp"]
            >= evaluation_period_start
        )
        &
        (
            master["timestamp"]
            <= pre_event_start
        )
    )

    roles.loc[
        mask,
        "role"
    ] = "evaluation_negative"


    # ========================================================
    # EVALUATION PRE-EVENT CONTEXT
    # ========================================================

    mask = (
        eligible
        &
        (
            master["timestamp"]
            > pre_event_start
        )
        &
        (
            master["timestamp"]
            <= event_start
        )
    )

    roles.loc[
        mask,
        "role"
    ] = "evaluation_pre_event_context"


    # ========================================================
    # EVALUATION FAULT
    # ========================================================

    mask = (
        eligible
        &
        (
            master["timestamp"]
            > event_start
        )
        &
        (
            master["timestamp"]
            <= evaluation_fault_end
        )
    )

    roles.loc[
        mask,
        "role"
    ] = "evaluation_failure"


    active = roles[
        roles["role"]
        != "outside"
    ].copy()

    active["split"] = (
        split_name
    )

    role_parts.append(
        active
    )


    # ========================================================
    # JOIN FEATURES
    # ========================================================

    data = master.merge(
        active[
            [
                "timestamp",
                "role",
            ]
        ],
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )


    train = data[
        data["role"].isin([
            "train_negative",
            "train_failure",
        ])
    ].copy()


    train_negative = train[
        train["role"]
        == "train_negative"
    ].copy()


    train_y = (
        train["role"]
        .eq(
            "train_failure"
        )
        .astype(int)
        .to_numpy()
    )


    evaluation = data[
        data["role"].isin([
            "evaluation_negative",
            "evaluation_pre_event_context",
            "evaluation_failure",
        ])
    ].copy()


    evaluation = (
        evaluation
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


    binary_mask = (
        evaluation["role"]
        .isin([
            "evaluation_negative",
            "evaluation_failure",
        ])
        .to_numpy()
    )


    binary_y = (
        evaluation.loc[
            binary_mask,
            "role",
        ]
        .eq(
            "evaluation_failure"
        )
        .astype(int)
        .to_numpy()
    )


    print("\n" + "=" * 92)

    print(
        f"{split_name}: "
        f"{training_events} → "
        f"{evaluation_event}"
    )

    print("=" * 92)

    print(
        f"Training rows: "
        f"{len(train):,}"
    )

    print(
        f"Training fault rows: "
        f"{train_y.sum():,}"
    )

    print(
        f"Evaluation normal rows: "
        f"{(binary_y == 0).sum():,}"
    )

    print(
        f"Evaluation fault rows: "
        f"{(binary_y == 1).sum():,}"
    )


    # ========================================================
    # STATIC BASELINE
    # ========================================================

    lower = (
        train_negative[
            STATIC_FEATURES
        ]
        .quantile(0.01)
    )

    upper = (
        train_negative[
            STATIC_FEATURES
        ]
        .quantile(0.99)
    )


    static_scores = (
        (
            evaluation[
                STATIC_FEATURES
            ]
            .lt(lower)
            |
            evaluation[
                STATIC_FEATURES
            ]
            .gt(upper)
        )
        .mean(axis=1)
        .to_numpy()
    )


    static_threshold = (
        1.0
        /
        len(
            STATIC_FEATURES
        )
    )


    ranking = ranking_metrics(
        binary_y,
        static_scores[
            binary_mask
        ],
    )


    operation = operational_metrics(
        evaluation,
        static_scores,
        static_threshold,
        event_start,
    )


    ranking_rows.append({
        "split":
            split_name,

        "validation_event":
            evaluation_event,

        "model":
            "Static Normal-Range",

        "feature_count":
            len(STATIC_FEATURES),

        **ranking,
    })


    operational_rows.append({
        "split":
            split_name,

        "validation_event":
            evaluation_event,

        "model":
            "Static Normal-Range",

        "threshold_method":
            "one_analogue_outside_1_99pct",

        "threshold_quantile":
            np.nan,

        "threshold_value":
            static_threshold,

        **operation,
    })


    # ========================================================
    # LEARNED MODELS
    # ========================================================

    model_specs = [
        (
            "LightGBM Current",
            "supervised",
            build_lightgbm(),
            CURRENT_FEATURES,
        ),

        (
            "LightGBM Temporal",
            "supervised",
            build_lightgbm(),
            ALL_FEATURES,
        ),

        (
            "Isolation Forest Temporal",
            "anomaly",
            build_isolation_forest(),
            ALL_FEATURES,
        ),
    ]


    for (
        model_name,
        model_type,
        model,
        features,
    ) in model_specs:


        print(
            f"Training "
            f"{model_name}..."
        )


        # ----------------------------------------------------
        # FIT
        # ----------------------------------------------------

        if model_type == "supervised":

            with warnings.catch_warnings():

                warnings.simplefilter(
                    "ignore"
                )

                model.fit(
                    train[
                        features
                    ],
                    train_y,
                )


            train_normal_scores = (
                model.predict_proba(
                    train_negative[
                        features
                    ]
                )[:, 1]
            )


            evaluation_scores = (
                model.predict_proba(
                    evaluation[
                        features
                    ]
                )[:, 1]
            )


        else:

            model.fit(
                train_negative[
                    features
                ]
            )


            train_normal_scores = (
                -
                model.decision_function(
                    train_negative[
                        features
                    ]
                )
            )


            evaluation_scores = (
                -
                model.decision_function(
                    evaluation[
                        features
                    ]
                )
            )


        # ----------------------------------------------------
        # RANKING
        # ----------------------------------------------------

        ranking = ranking_metrics(
            binary_y,
            evaluation_scores[
                binary_mask
            ],
        )


        ranking_rows.append({
            "split":
                split_name,

            "validation_event":
                evaluation_event,

            "model":
                model_name,

            "feature_count":
                len(features),

            **ranking,
        })


        # ----------------------------------------------------
        # OPERATIONAL THRESHOLDS
        # ----------------------------------------------------

        for quantile in QUANTILES:

            threshold = float(
                np.quantile(
                    train_normal_scores,
                    quantile,
                )
            )


            operation = operational_metrics(
                evaluation,
                evaluation_scores,
                threshold,
                event_start,
            )


            operational_rows.append({
                "split":
                    split_name,

                "validation_event":
                    evaluation_event,

                "model":
                    model_name,

                "threshold_method":
                    "train_normal_quantile",

                "threshold_quantile":
                    quantile,

                "threshold_value":
                    threshold,

                **operation,
            })


        # ----------------------------------------------------
        # SAVE SCORES
        # ----------------------------------------------------

        pred = evaluation[
            [
                "timestamp",
                "role",
            ]
        ].copy()

        pred["split"] = (
            split_name
        )

        pred[
            "validation_event"
        ] = evaluation_event

        pred["model"] = (
            model_name
        )

        pred["score"] = (
            evaluation_scores
        )

        prediction_parts.append(
            pred
        )


# ============================================================
# RESULTS
# ============================================================

roles_df = pd.concat(
    role_parts,
    ignore_index=True,
)

ranking_df = pd.DataFrame(
    ranking_rows
)

operational_df = pd.DataFrame(
    operational_rows
)

predictions_df = pd.concat(
    prediction_parts,
    ignore_index=True,
)


roles_df.to_parquet(
    ROOT
    / "data"
    / "processed"
    / "splits"
    / "fault_state_split_membership.parquet",
    index=False,
)


ranking_df.to_csv(
    METRIC_DIR
    / "fault_state_development_ranking.csv",
    index=False,
)


operational_df.to_csv(
    METRIC_DIR
    / "fault_state_development_operational.csv",
    index=False,
)


predictions_df.to_parquet(
    METRIC_DIR
    / "fault_state_development_predictions.parquet",
    index=False,
)


# ============================================================
# MACRO RANKING
# ============================================================

ranking_scoreboard = (
    ranking_df
    .groupby(
        "model",
        as_index=False,
    )
    .agg(
        mean_pr_auc=(
            "pr_auc",
            "mean",
        ),

        minimum_event_pr_auc=(
            "pr_auc",
            "min",
        ),

        maximum_event_pr_auc=(
            "pr_auc",
            "max",
        ),

        mean_average_precision=(
            "average_precision",
            "mean",
        ),

        mean_roc_auc=(
            "roc_auc",
            "mean",
        ),

        mean_ap_lift=(
            "ap_lift_over_prevalence",
            "mean",
        ),
    )
)


ranking_scoreboard.to_csv(
    TABLE_DIR
    / "fault_state_ranking_scoreboard.csv",
    index=False,
)


# ============================================================
# OPERATIONAL MACRO
# ============================================================

learned_operation = (
    operational_df[
        operational_df[
            "threshold_method"
        ]
        == "train_normal_quantile"
    ]
)


operational_scoreboard = (
    learned_operation
    .groupby(
        [
            "model",
            "threshold_quantile",
        ],
        as_index=False,
    )
    .agg(
        events_detected=(
            "event_detected",
            "sum",
        ),

        median_detection_delay_minutes=(
            "detection_delay_minutes",
            "median",
        ),

        mean_false_alarms_per_day=(
            "false_alarm_episodes_per_day",
            "mean",
        ),

        worst_false_alarms_per_day=(
            "false_alarm_episodes_per_day",
            "max",
        ),

        mean_pre_event_warning_minutes=(
            "pre_event_warning_minutes",
            "mean",
        ),
    )
)


operational_scoreboard[
    "events_total"
] = 2


operational_scoreboard.to_csv(
    TABLE_DIR
    / "fault_state_operational_scoreboard.csv",
    index=False,
)


# ============================================================
# STATIC OPERATION
# ============================================================

static_operation = (
    operational_df[
        operational_df["model"]
        == "Static Normal-Range"
    ][
        [
            "validation_event",
            "event_detected",
            "detection_delay_minutes",
            "false_alarm_episodes_per_day",
            "pre_event_warning_minutes",
        ]
    ]
)


static_operation.to_csv(
    TABLE_DIR
    / "fault_state_static_operational.csv",
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "development_validation_events": [
        "F02",
        "F03",
    ],

    "final_holdout_event_used":
        False,

    "pre_event_exclusion_minutes":
        PRE_EXCLUSION,

    "recovery_minutes":
        RECOVERY,

    "persistence_minutes":
        PERSISTENCE,

    "model_feature_count_temporal":
        len(ALL_FEATURES),

    "model_feature_count_current":
        len(CURRENT_FEATURES),
}


with open(
    METRIC_DIR
    / "fault_state_benchmark_config.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# PRINT
# ============================================================

pd.set_option(
    "display.max_columns",
    None,
)

pd.set_option(
    "display.width",
    240,
)


print("\n" + "=" * 92)
print("FAULT-STATE RANKING SCOREBOARD")
print("=" * 92)

print(
    ranking_scoreboard
    .sort_values(
        "mean_pr_auc",
        ascending=False,
    )
    .round(5)
    .to_string(index=False)
)


print("\n" + "=" * 92)
print("OPERATIONAL SCOREBOARD — 99.9% TRAIN-NORMAL THRESHOLD")
print("=" * 92)

view = (
    operational_scoreboard[
        operational_scoreboard[
            "threshold_quantile"
        ]
        == 0.999
    ]
)

print(
    view
    .sort_values(
        [
            "events_detected",
            "mean_false_alarms_per_day",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .round(5)
    .to_string(index=False)
)


print("\n" + "=" * 92)
print("STATIC BASELINE")
print("=" * 92)

print(
    static_operation
    .round(5)
    .to_string(index=False)
)


print("\nIMPORTANT:")
print(
    "F04 was not used anywhere "
    "in this benchmark."
)

print("\nFAULT-STATE BENCHMARK COMPLETE")
