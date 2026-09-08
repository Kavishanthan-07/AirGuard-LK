from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    IsolationForest,
)
from sklearn.metrics import (
    average_precision_score,
    auc,
    precision_recall_curve,
    roc_auc_score,
)
from lightgbm import LGBMClassifier


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MASTER_PATH = (
    ROOT
    / "data"
    / "processed"
    / "metropt3_model_master.parquet"
)

SPLIT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "splits"
    / "split_membership.parquet"
)

BOUNDARY_PATH = (
    ROOT
    / "artifacts"
    / "tables"
    / "metropt3_split_boundaries.csv"
)

EVENT_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "metropt3_failure_events.csv"
)

METRIC_DIR = ROOT / "artifacts" / "metrics"
TABLE_DIR = ROOT / "artifacts" / "tables"

METRIC_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

HORIZONS = [15, 30, 60, 120]

DEVELOPMENT_SPLITS = [
    "fold_F02",
    "fold_F03",
]

# Threshold is derived ONLY from training-negative scores.
THRESHOLD_QUANTILES = [
    0.990,
    0.995,
    0.999,
    0.9995,
]

PERSISTENCE_MINUTES = 3


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


# ============================================================
# LOAD
# ============================================================

print("=" * 90)
print("AIRGUARD-LK — NONLINEAR + ANOMALY DEVELOPMENT EXPERIMENT")
print("=" * 90)

master = pd.read_parquet(
    MASTER_PATH
)

master["timestamp"] = pd.to_datetime(
    master["timestamp"]
)


membership = pd.read_parquet(
    SPLIT_PATH
)

membership["timestamp"] = pd.to_datetime(
    membership["timestamp"]
)


boundaries = pd.read_csv(
    BOUNDARY_PATH,
    parse_dates=[
        "training_end",
        "evaluation_start",
        "evaluation_end",
    ],
)


events = pd.read_csv(
    EVENT_PATH,
    parse_dates=[
        "start_time",
        "end_time",
    ],
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


event_lookup = (
    events
    .set_index("event_id")
    ["start_time"]
    .to_dict()
)


validation_event_lookup = (
    boundaries[
        boundaries["type"]
        == "development"
    ]
    .set_index("split")
    ["evaluation_event"]
    .to_dict()
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
        "average_precision":
            float(ap),

        "pr_auc":
            float(pr_auc),

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
    persistence_minutes=3,
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

        current = (
            timestamps.iloc[i]
        )

        if previous_time is None:

            consecutive = True

        else:

            delta = (
                current
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

        if (
            streak
            >= persistence_minutes
        ):

            flags[i] = True

        previous_time = current

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

        gap = (
            alert_times.iloc[i]
            -
            alert_times.iloc[i - 1]
        ).total_seconds()

        if gap > 60:
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
            PERSISTENCE_MINUTES,
        )
    )


    # --------------------------------------------------------
    # FALSE ALARMS
    # --------------------------------------------------------

    negative = temp[
        temp["role"]
        == "validation_negative"
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


    # --------------------------------------------------------
    # TARGET WINDOW DETECTION
    # --------------------------------------------------------

    positive = temp[
        temp["role"]
        == "validation_positive"
    ]

    positive_alerts = positive[
        positive["alert"]
    ]

    detected = (
        len(positive_alerts) > 0
    )

    if detected:

        first_alert = (
            positive_alerts[
                "timestamp"
            ].min()
        )

        warning_minutes = (
            event_start
            -
            first_alert
        ).total_seconds() / 60.0

    else:

        first_alert = pd.NaT
        warning_minutes = np.nan


    # --------------------------------------------------------
    # EARLIEST ALERT IN FULL 6H PRE-EVENT ZONE
    # --------------------------------------------------------

    pre_event = temp[
        temp["role"].isin([
            "validation_ambiguous",
            "validation_positive",
        ])
    ]

    early_alerts = pre_event[
        pre_event["alert"]
    ]

    if len(early_alerts):

        earliest = (
            early_alerts[
                "timestamp"
            ].min()
        )

        earliest_warning = (
            event_start
            -
            earliest
        ).total_seconds() / 60.0

    else:

        earliest = pd.NaT
        earliest_warning = np.nan


    return {
        "event_detected":
            int(detected),

        "warning_minutes":
            float(warning_minutes)
            if np.isfinite(
                warning_minutes
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

        "earliest_warning_360m":
            float(
                earliest_warning
            )
            if np.isfinite(
                earliest_warning
            )
            else np.nan,
    }


# ============================================================
# MODEL BUILDERS
# ============================================================

def build_random_forest():

    return RandomForestClassifier(
        n_estimators=300,

        max_depth=12,

        min_samples_leaf=5,

        max_features="sqrt",

        class_weight=
            "balanced_subsample",

        n_jobs=-1,

        random_state=SEED,
    )


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
# STORAGE
# ============================================================

ranking_rows = []
operational_rows = []
prediction_parts = []


# ============================================================
# EXPERIMENT
# ============================================================

for split_name in DEVELOPMENT_SPLITS:

    validation_event = (
        validation_event_lookup[
            split_name
        ]
    )

    event_start = pd.Timestamp(
        event_lookup[
            validation_event
        ]
    )


    print("\n" + "=" * 90)
    print(
        f"{split_name} "
        f"→ validation event "
        f"{validation_event}"
    )
    print("=" * 90)


    for horizon in HORIZONS:

        print(
            f"\nHorizon: "
            f"{horizon} min"
        )


        split_membership = (
            membership[
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
            ][
                [
                    "timestamp",
                    "role",
                ]
            ]
        )


        data = master.merge(
            split_membership,
            on="timestamp",
            how="inner",
            validate="one_to_one",
        )


        train = data[
            data["role"].isin([
                "train_negative",
                "train_positive",
            ])
        ].copy()


        train_negative = train[
            train["role"]
            == "train_negative"
        ].copy()


        train_y = (
            train["role"]
            .eq("train_positive")
            .astype(int)
            .to_numpy()
        )


        evaluation = data[
            data["role"].isin([
                "validation_negative",
                "validation_ambiguous",
                "validation_positive",
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
                "validation_negative",
                "validation_positive",
            ])
            .to_numpy()
        )


        binary_y = (
            evaluation.loc[
                binary_mask,
                "role",
            ]
            .eq(
                "validation_positive"
            )
            .astype(int)
            .to_numpy()
        )


        print(
            f"Train rows: "
            f"{len(train):,} "
            f"| positives: "
            f"{train_y.sum():,}"
        )

        print(
            f"Validation binary rows: "
            f"{binary_mask.sum():,} "
            f"| positives: "
            f"{binary_y.sum():,}"
        )


        # ====================================================
        # MODEL DEFINITIONS
        # ====================================================

        model_specs = [
            (
                "Random Forest Temporal",
                "supervised",
                build_random_forest(),
                ALL_FEATURES,
            ),

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
                f"  Training "
                f"{model_name}..."
            )


            # ------------------------------------------------
            # FIT
            # ------------------------------------------------

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


                train_negative_scores = (
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

                # Anomaly model sees NORMAL data only.
                model.fit(
                    train_negative[
                        features
                    ]
                )


                # decision_function:
                # larger = more normal
                #
                # We negate it so:
                # larger = more anomalous.
                train_negative_scores = (
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


            # ------------------------------------------------
            # RANKING PERFORMANCE
            # ------------------------------------------------

            binary_scores = (
                evaluation_scores[
                    binary_mask
                ]
            )


            ranking = (
                ranking_metrics(
                    binary_y,
                    binary_scores,
                )
            )


            ranking_rows.append({
                "split":
                    split_name,

                "validation_event":
                    validation_event,

                "horizon_minutes":
                    horizon,

                "model":
                    model_name,

                "model_type":
                    model_type,

                "feature_count":
                    len(features),

                **ranking,
            })


            # ------------------------------------------------
            # OPERATIONAL THRESHOLDS
            # ------------------------------------------------

            for quantile in THRESHOLD_QUANTILES:

                threshold = float(
                    np.quantile(
                        train_negative_scores,
                        quantile,
                    )
                )


                operation = (
                    operational_metrics(
                        evaluation,
                        evaluation_scores,
                        threshold,
                        event_start,
                    )
                )


                operational_rows.append({
                    "split":
                        split_name,

                    "validation_event":
                        validation_event,

                    "horizon_minutes":
                        horizon,

                    "model":
                        model_name,

                    "threshold_quantile":
                        quantile,

                    "threshold_value":
                        threshold,

                    **operation,
                })


            # ------------------------------------------------
            # SAVE PREDICTIONS
            # ------------------------------------------------

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
            ] = validation_event

            pred[
                "horizon_minutes"
            ] = horizon

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
# SAVE RAW RESULTS
# ============================================================

ranking_df = pd.DataFrame(
    ranking_rows
)

operational_df = pd.DataFrame(
    operational_rows
)

predictions = pd.concat(
    prediction_parts,
    ignore_index=True,
)


ranking_df.to_csv(
    METRIC_DIR
    / "development_nonlinear_ranking.csv",
    index=False,
)


operational_df.to_csv(
    METRIC_DIR
    / "development_nonlinear_operational.csv",
    index=False,
)


predictions.to_parquet(
    METRIC_DIR
    / "development_nonlinear_predictions.parquet",
    index=False,
)


# ============================================================
# MACRO RANKING SCOREBOARD
# ============================================================

ranking_scoreboard = (
    ranking_df
    .groupby(
        [
            "model",
            "horizon_minutes",
        ],
        as_index=False,
    )
    .agg(
        mean_pr_auc=(
            "pr_auc",
            "mean",
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

        minimum_event_pr_auc=(
            "pr_auc",
            "min",
        ),

        maximum_event_pr_auc=(
            "pr_auc",
            "max",
        ),
    )
)


ranking_scoreboard.to_csv(
    TABLE_DIR
    / "development_nonlinear_ranking_scoreboard.csv",
    index=False,
)


# ============================================================
# OPERATIONAL SCOREBOARD
# ============================================================

operational_scoreboard = (
    operational_df
    .groupby(
        [
            "model",
            "horizon_minutes",
            "threshold_quantile",
        ],
        as_index=False,
    )
    .agg(
        events_detected=(
            "event_detected",
            "sum",
        ),

        median_warning_minutes=(
            "warning_minutes",
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

        mean_earliest_warning_360m=(
            "earliest_warning_360m",
            "mean",
        ),
    )
)


operational_scoreboard[
    "events_total"
] = len(
    DEVELOPMENT_SPLITS
)


operational_scoreboard.to_csv(
    TABLE_DIR
    / "development_nonlinear_operational_scoreboard.csv",
    index=False,
)


# ============================================================
# CONFIG RECORD
# ============================================================

config_record = {
    "development_splits":
        DEVELOPMENT_SPLITS,

    "horizons":
        HORIZONS,

    "threshold_quantiles":
        THRESHOLD_QUANTILES,

    "persistence_minutes":
        PERSISTENCE_MINUTES,

    "models": [
        "Random Forest Temporal",
        "LightGBM Current",
        "LightGBM Temporal",
        "Isolation Forest Temporal",
    ],

    "final_F04_used":
        False,
}


with open(
    METRIC_DIR
    / "development_nonlinear_config.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        config_record,
        f,
        indent=2,
    )


# ============================================================
# CONSOLE
# ============================================================

pd.set_option(
    "display.width",
    240,
)

pd.set_option(
    "display.max_columns",
    None,
)


print("\n" + "=" * 90)
print("MACRO RANKING SCOREBOARD")
print("=" * 90)

print(
    ranking_scoreboard
    .sort_values(
        [
            "horizon_minutes",
            "mean_pr_auc",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .round(5)
    .to_string(index=False)
)


print("\n" + "=" * 90)
print("OPERATIONAL SCOREBOARD — 99.9% TRAIN-NORMAL THRESHOLD")
print("=" * 90)

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
            "horizon_minutes",
            "events_detected",
            "mean_false_alarms_per_day",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )
    .round(5)
    .to_string(index=False)
)


print("\nIMPORTANT:")
print(
    "F04 was not used for fitting, "
    "threshold derivation, or evaluation."
)

print("\nEXPERIMENT COMPLETE")
