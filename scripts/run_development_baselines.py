from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    auc,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
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

SEED = 42

HORIZONS = [
    15,
    30,
    60,
    120,
]

DEVELOPMENT_SPLITS = [
    "fold_F02",
    "fold_F03",
]

PERSISTENCE_MINUTES = 3


# ============================================================
# FEATURE DEFINITIONS
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


# ------------------------------------------------------------
# Static baseline:
# only instantaneous analogue sensor averages
# ------------------------------------------------------------

STATIC_FEATURES = [
    f"{sensor}_mean"
    for sensor in ANALOG
]


# ------------------------------------------------------------
# Current-minute feature configuration
# ------------------------------------------------------------

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
# LOAD DATA
# ============================================================

print("=" * 86)
print("AIRGUARD-LK — DEVELOPMENT BASELINE EXPERIMENT")
print("=" * 86)

print("\nLoading model master...")

master = pd.read_parquet(
    MASTER_PATH
)

master["timestamp"] = pd.to_datetime(
    master["timestamp"]
)


print("Loading split membership...")

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


# ============================================================
# FEATURE SAFETY
# ============================================================

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
            f"Missing current feature: {feature}"
        )


print()
print(f"Current features : {len(CURRENT_FEATURES)}")
print(f"Temporal features: {len(ALL_FEATURES)}")


# ============================================================
# EVENT LOOKUP
# ============================================================

event_lookup = (
    events
    .set_index("event_id")
    ["start_time"]
    .to_dict()
)

validation_event_lookup = (
    boundaries[
        boundaries["type"] == "development"
    ]
    .set_index("split")
    ["evaluation_event"]
    .to_dict()
)


# ============================================================
# HELPERS
# ============================================================

def binary_metrics(
    y_true,
    scores,
    threshold,
):

    predictions = (
        scores >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    average_precision = (
        average_precision_score(
            y_true,
            scores,
        )
    )

    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            y_true,
            scores,
        )
    )

    pr_auc = auc(
        recall_curve,
        precision_curve,
    )

    if len(np.unique(y_true)) == 2:

        roc_auc = roc_auc_score(
            y_true,
            scores,
        )

    else:
        roc_auc = np.nan

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),

        "average_precision":
            float(average_precision),

        "pr_auc":
            float(pr_auc),

        "roc_auc":
            float(roc_auc),
    }


def persistent_alert_flags(
    timestamps,
    scores,
    threshold,
    persistence_minutes=3,
):

    timestamps = pd.Series(
        pd.to_datetime(timestamps)
    ).reset_index(drop=True)

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

    for i in range(len(scores)):

        current_time = timestamps.iloc[i]

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

        if streak >= persistence_minutes:
            flags[i] = True

        previous_time = current_time

    return flags


def count_alert_episodes(
    timestamps,
    alert_flags,
):

    alert_times = (
        pd.Series(
            pd.to_datetime(timestamps)
        )
        .loc[alert_flags]
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


def evaluate_event_behavior(
    eval_df,
    scores,
    threshold,
    event_start,
):

    temp = eval_df[
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
            persistence_minutes=
                PERSISTENCE_MINUTES,
        )
    )


    # --------------------------------------------------------
    # False alarms
    # --------------------------------------------------------

    negative = temp[
        temp["role"]
        == "validation_negative"
    ].copy()

    false_alarm_episodes = (
        count_alert_episodes(
            negative["timestamp"],
            negative["alert"].to_numpy(),
        )
    )

    observed_negative_days = (
        len(negative)
        / 1440.0
    )

    if observed_negative_days > 0:

        false_alarms_per_day = (
            false_alarm_episodes
            /
            observed_negative_days
        )

    else:

        false_alarms_per_day = np.nan


    # --------------------------------------------------------
    # Detection inside the official target window
    # --------------------------------------------------------

    positive = temp[
        temp["role"]
        == "validation_positive"
    ]

    positive_alerts = positive[
        positive["alert"]
    ]

    event_detected = (
        len(positive_alerts) > 0
    )

    if event_detected:

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
    # Diagnostic:
    # earliest alert anywhere inside 360-minute
    # pre-event neighborhood
    # --------------------------------------------------------

    pre_event = temp[
        (
            temp["timestamp"]
            < event_start
        )
        &
        (
            temp["role"].isin([
                "validation_ambiguous",
                "validation_positive",
            ])
        )
    ]

    early_alerts = pre_event[
        pre_event["alert"]
    ]

    if len(early_alerts):

        earliest_360_alert = (
            early_alerts[
                "timestamp"
            ].min()
        )

        earliest_360_warning = (
            event_start
            -
            earliest_360_alert
        ).total_seconds() / 60.0

    else:

        earliest_360_alert = pd.NaT
        earliest_360_warning = np.nan


    return {
        "event_detected":
            int(event_detected),

        "warning_minutes":
            warning_minutes,

        "first_alert_timestamp":
            first_alert,

        "earliest_alert_360m_timestamp":
            earliest_360_alert,

        "earliest_warning_360m":
            earliest_360_warning,

        "false_alarm_episodes":
            int(false_alarm_episodes),

        "observed_negative_days":
            float(observed_negative_days),

        "false_alarm_episodes_per_day":
            float(false_alarms_per_day),
    }


# ============================================================
# STATIC RANGE BASELINE
# ============================================================

def fit_static_ranges(
    train_negative,
):

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

    return lower, upper


def static_range_score(
    data,
    lower,
    upper,
):

    X = data[
        STATIC_FEATURES
    ]

    outside = (
        (X.lt(lower))
        |
        (X.gt(upper))
    )

    # Fraction of analogue signals outside
    # their training-normal 1%-99% ranges.
    score = (
        outside
        .mean(axis=1)
        .to_numpy(dtype=float)
    )

    return score


# Alert if at least one of the seven
# analogue signals violates its normal range.
STATIC_THRESHOLD = (
    1.0
    /
    len(STATIC_FEATURES)
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

def fit_logistic(
    train_df,
    train_y,
    features,
):

    model = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                C=1.0,
                solver="lbfgs",
                max_iter=2000,
            ),
        ),
    ])

    with warnings.catch_warnings():

        warnings.filterwarnings(
            "ignore",
            category=ConvergenceWarning,
        )

        model.fit(
            train_df[features],
            train_y,
        )

    return model


# ============================================================
# EXPERIMENT
# ============================================================

result_rows = []
prediction_parts = []


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

    print("\n" + "=" * 86)
    print(
        f"DEVELOPMENT SPLIT: "
        f"{split_name}"
    )
    print(
        f"Validation event: "
        f"{validation_event}"
    )
    print("=" * 86)


    for horizon in HORIZONS:

        print(
            f"\nHorizon: "
            f"{horizon} minutes"
        )

        membership_subset = (
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
            membership_subset,
            on="timestamp",
            how="inner",
            validate="one_to_one",
        )


        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        train = data[
            data["role"].isin([
                "train_negative",
                "train_positive",
            ])
        ].copy()

        train_y = (
            train["role"]
            .eq("train_positive")
            .astype(int)
            .to_numpy()
        )


        # ----------------------------------------------------
        # EVALUATION
        # ----------------------------------------------------

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


        binary_eval = evaluation[
            evaluation["role"].isin([
                "validation_negative",
                "validation_positive",
            ])
        ].copy()

        binary_y = (
            binary_eval["role"]
            .eq("validation_positive")
            .astype(int)
            .to_numpy()
        )


        print(
            f"Train: "
            f"{len(train):,} "
            f"({train_y.sum():,} positive)"
        )

        print(
            f"Binary validation: "
            f"{len(binary_eval):,} "
            f"({binary_y.sum():,} positive)"
        )


        # ====================================================
        # MODEL 1 — STATIC RANGE
        # ====================================================

        train_negative = train[
            train["role"]
            == "train_negative"
        ]

        lower, upper = (
            fit_static_ranges(
                train_negative
            )
        )

        eval_scores_full = (
            static_range_score(
                evaluation,
                lower,
                upper,
            )
        )

        binary_mask = (
            evaluation["role"]
            .isin([
                "validation_negative",
                "validation_positive",
            ])
            .to_numpy()
        )

        binary_scores = (
            eval_scores_full[
                binary_mask
            ]
        )

        metrics = binary_metrics(
            binary_y,
            binary_scores,
            STATIC_THRESHOLD,
        )

        event_metrics = (
            evaluate_event_behavior(
                evaluation,
                eval_scores_full,
                STATIC_THRESHOLD,
                event_start,
            )
        )

        result_rows.append({
            "split":
                split_name,

            "validation_event":
                validation_event,

            "horizon_minutes":
                horizon,

            "model":
                "Static Normal-Range",

            "feature_count":
                len(STATIC_FEATURES),

            "decision_threshold":
                STATIC_THRESHOLD,

            **metrics,
            **event_metrics,
        })


        pred = evaluation[
            [
                "timestamp",
                "role",
            ]
        ].copy()

        pred["split"] = split_name
        pred["horizon_minutes"] = horizon
        pred["model"] = (
            "Static Normal-Range"
        )
        pred["score"] = (
            eval_scores_full
        )

        prediction_parts.append(
            pred
        )


        # ====================================================
        # MODEL 2 — CURRENT-MINUTE LOGISTIC REGRESSION
        # ====================================================

        current_model = fit_logistic(
            train,
            train_y,
            CURRENT_FEATURES,
        )

        eval_scores_full = (
            current_model
            .predict_proba(
                evaluation[
                    CURRENT_FEATURES
                ]
            )[:, 1]
        )

        binary_scores = (
            eval_scores_full[
                binary_mask
            ]
        )

        metrics = binary_metrics(
            binary_y,
            binary_scores,
            0.5,
        )

        event_metrics = (
            evaluate_event_behavior(
                evaluation,
                eval_scores_full,
                0.5,
                event_start,
            )
        )

        result_rows.append({
            "split":
                split_name,

            "validation_event":
                validation_event,

            "horizon_minutes":
                horizon,

            "model":
                "Logistic Current",

            "feature_count":
                len(CURRENT_FEATURES),

            "decision_threshold":
                0.5,

            **metrics,
            **event_metrics,
        })


        pred = evaluation[
            [
                "timestamp",
                "role",
            ]
        ].copy()

        pred["split"] = split_name
        pred["horizon_minutes"] = horizon
        pred["model"] = (
            "Logistic Current"
        )
        pred["score"] = (
            eval_scores_full
        )

        prediction_parts.append(
            pred
        )


        # ====================================================
        # MODEL 3 — TEMPORAL LOGISTIC REGRESSION
        # ====================================================

        temporal_model = fit_logistic(
            train,
            train_y,
            ALL_FEATURES,
        )

        eval_scores_full = (
            temporal_model
            .predict_proba(
                evaluation[
                    ALL_FEATURES
                ]
            )[:, 1]
        )

        binary_scores = (
            eval_scores_full[
                binary_mask
            ]
        )

        metrics = binary_metrics(
            binary_y,
            binary_scores,
            0.5,
        )

        event_metrics = (
            evaluate_event_behavior(
                evaluation,
                eval_scores_full,
                0.5,
                event_start,
            )
        )

        result_rows.append({
            "split":
                split_name,

            "validation_event":
                validation_event,

            "horizon_minutes":
                horizon,

            "model":
                "Logistic Temporal",

            "feature_count":
                len(ALL_FEATURES),

            "decision_threshold":
                0.5,

            **metrics,
            **event_metrics,
        })


        pred = evaluation[
            [
                "timestamp",
                "role",
            ]
        ].copy()

        pred["split"] = split_name
        pred["horizon_minutes"] = horizon
        pred["model"] = (
            "Logistic Temporal"
        )
        pred["score"] = (
            eval_scores_full
        )

        prediction_parts.append(
            pred
        )


# ============================================================
# RESULTS
# ============================================================

results = pd.DataFrame(
    result_rows
)

results.to_csv(
    METRIC_DIR
    / "development_baseline_results.csv",
    index=False,
)


predictions = pd.concat(
    prediction_parts,
    ignore_index=True,
)

predictions.to_parquet(
    METRIC_DIR
    / "development_baseline_predictions.parquet",
    index=False,
)


# ============================================================
# MACRO DEVELOPMENT SCOREBOARD
#
# Each independent validation event gets equal weight.
# ============================================================

scoreboard = (
    results
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

        mean_precision=(
            "precision",
            "mean",
        ),

        mean_recall=(
            "recall",
            "mean",
        ),

        mean_f1=(
            "f1",
            "mean",
        ),

        events_detected=(
            "event_detected",
            "sum",
        ),

        median_warning_minutes=(
            "warning_minutes",
            "median",
        ),

        mean_false_alarm_episodes_per_day=(
            "false_alarm_episodes_per_day",
            "mean",
        ),

        mean_earliest_warning_360m=(
            "earliest_warning_360m",
            "mean",
        ),
    )
)


scoreboard[
    "events_total"
] = len(
    DEVELOPMENT_SPLITS
)


scoreboard = scoreboard[
    [
        "model",
        "horizon_minutes",

        "mean_pr_auc",
        "mean_average_precision",

        "mean_precision",
        "mean_recall",
        "mean_f1",

        "events_detected",
        "events_total",

        "median_warning_minutes",

        "mean_false_alarm_episodes_per_day",

        "mean_earliest_warning_360m",
    ]
]


scoreboard.to_csv(
    TABLE_DIR
    / "development_baseline_scoreboard.csv",
    index=False,
)


# ============================================================
# SAVE EXPERIMENT INFO
# ============================================================

experiment_info = {
    "development_splits":
        DEVELOPMENT_SPLITS,

    "candidate_horizons_minutes":
        HORIZONS,

    "models": [
        "Static Normal-Range",
        "Logistic Current",
        "Logistic Temporal",
    ],

    "static_features":
        len(STATIC_FEATURES),

    "current_features":
        len(CURRENT_FEATURES),

    "temporal_features":
        len(ALL_FEATURES),

    "persistence_minutes":
        PERSISTENCE_MINUTES,

    "logistic_class_weight":
        "balanced",

    "logistic_C":
        1.0,

    "decision_threshold_logistic":
        0.5,

    "final_holdout_used":
        False,
}


with open(
    METRIC_DIR
    / "development_baseline_config.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        experiment_info,
        f,
        indent=2
    )


# ============================================================
# CONSOLE REPORT
# ============================================================

pd.set_option(
    "display.max_columns",
    None,
)

pd.set_option(
    "display.width",
    220,
)

print("\n" + "=" * 86)
print("DEVELOPMENT SCOREBOARD")
print("=" * 86)

print(
    scoreboard
    .sort_values(
        [
            "horizon_minutes",
            "model",
        ]
    )
    .round(4)
    .to_string(index=False)
)


print("\n" + "=" * 86)
print("EVENT-BY-EVENT RESULTS")
print("=" * 86)

event_view = results[
    [
        "split",
        "validation_event",
        "horizon_minutes",
        "model",

        "pr_auc",
        "recall",
        "f1",

        "event_detected",
        "warning_minutes",

        "false_alarm_episodes_per_day",
    ]
]

print(
    event_view
    .sort_values(
        [
            "horizon_minutes",
            "model",
            "split",
        ]
    )
    .round(4)
    .to_string(index=False)
)


print("\nSaved:")
print(
    METRIC_DIR
    / "development_baseline_results.csv"
)

print(
    TABLE_DIR
    / "development_baseline_scoreboard.csv"
)

print("\nIMPORTANT:")
print(
    "F04 was not used in this experiment."
)

print("\nBASELINE EXPERIMENT COMPLETE")
