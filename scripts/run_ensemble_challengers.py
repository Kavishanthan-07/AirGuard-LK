from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import yaml

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
from xgboost import XGBClassifier


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

SPLIT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "splits"
    / "fault_state_split_membership.parquet"
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
    / "ensemble_challenger_protocol.yaml"
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
    exist_ok=True,
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIG
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

PERSISTENCE = int(
    config["alerting"][
        "persistence_minutes"
    ]
)

THRESHOLD_QUANTILES = [
    float(x)
    for x in config[
        "alerting"
    ][
        "threshold_quantiles"
    ]
]

DEVELOPMENT_SPLITS = [
    "fault_F02",
    "fault_F03",
]


# ============================================================
# CURRENT FEATURE SET
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


assert len(CURRENT_FEATURES) == 71


# ============================================================
# LOAD
# ============================================================

print("=" * 100)
print("AIRGUARD-LK — ENSEMBLE CHALLENGER DEVELOPMENT EXPERIMENT")
print("=" * 100)

print(
    "\nF04 is NOT used for challenger "
    "training or selection."
)


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


events = pd.read_csv(
    EVENT_PATH,
    parse_dates=[
        "start_time",
        "end_time",
    ],
)


event_start_lookup = (
    events
    .set_index("event_id")
    ["start_time"]
    .to_dict()
)


validation_event_lookup = {
    "fault_F02": "F02",
    "fault_F03": "F03",
}


# ============================================================
# HELPERS
# ============================================================

def empirical_percentile(
    reference_scores,
    scores,
):

    reference = np.sort(
        np.asarray(
            reference_scores,
            dtype=float,
        )
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    if len(reference) == 0:

        raise RuntimeError(
            "Empty percentile reference."
        )

    ranks = np.searchsorted(
        reference,
        scores,
        side="right",
    )

    return (
        ranks.astype(float)
        / len(reference)
    )


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

    roc_auc = roc_auc_score(
        y_true,
        scores,
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

    alerts = np.zeros(
        len(scores),
        dtype=bool,
    )

    streak = 0
    previous_timestamp = None


    for i in range(
        len(scores)
    ):

        timestamp = (
            timestamps.iloc[i]
        )


        if previous_timestamp is None:

            consecutive = True

        else:

            consecutive = (
                (
                    timestamp
                    -
                    previous_timestamp
                ).total_seconds()
                == 60
            )


        if not consecutive:

            streak = 0


        if scores[i] >= threshold:

            streak += 1

        else:

            streak = 0


        if streak >= PERSISTENCE:

            alerts[i] = True


        previous_timestamp = (
            timestamp
        )


    return alerts


def count_alert_episodes(
    timestamps,
    alerts,
):

    alert_times = (
        pd.Series(
            pd.to_datetime(
                timestamps
            )
        )
        .loc[
            np.asarray(
                alerts,
                dtype=bool,
            )
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
    risk_scores,
    threshold,
    event_start,
):

    temp = evaluation[
        [
            "timestamp",
            "role",
        ]
    ].copy()

    temp["risk"] = (
        risk_scores
    )

    temp = (
        temp
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


    temp["alert"] = (
        persistent_alert_flags(
            temp["timestamp"],
            temp["risk"],
            threshold,
        )
    )


    # --------------------------------------------------------
    # FALSE ALARMS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # FAULT DETECTION
    # --------------------------------------------------------

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


        delay = (
            first_alert
            -
            event_start
        ).total_seconds() / 60.0

    else:

        first_alert = pd.NaT
        delay = np.nan


    return {
        "event_detected":
            int(detected),

        "detection_delay_minutes":
            float(delay)
            if np.isfinite(delay)
            else np.nan,

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
    }


# ============================================================
# MODEL BUILDERS
# ============================================================

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


def build_random_forest():

    return RandomForestClassifier(
        n_estimators=400,

        max_depth=12,

        min_samples_leaf=5,

        max_features="sqrt",

        class_weight=
            "balanced_subsample",

        random_state=SEED,

        n_jobs=-1,
    )


def build_xgboost(
    scale_pos_weight,
):

    return XGBClassifier(
        objective="binary:logistic",

        n_estimators=400,

        learning_rate=0.03,

        max_depth=5,

        min_child_weight=5,

        subsample=0.8,

        colsample_bytree=0.8,

        reg_lambda=1.0,

        scale_pos_weight=
            scale_pos_weight,

        eval_metric="logloss",

        tree_method="hist",

        random_state=SEED,

        n_jobs=-1,
    )


def build_isolation_forest():

    return IsolationForest(
        n_estimators=400,

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
correlation_rows = []


# ============================================================
# DEVELOPMENT
# ============================================================

for split_name in DEVELOPMENT_SPLITS:

    print("\n" + "=" * 100)
    print(split_name)
    print("=" * 100)


    split_membership = (
        membership[
            membership["split"]
            == split_name
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
            "train_failure",
        ])
    ].copy()


    train_normal = train[
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


    validation_event = (
        validation_event_lookup[
            split_name
        ]
    )


    event_start = pd.Timestamp(
        event_start_lookup[
            validation_event
        ]
    )


    n_positive = int(
        train_y.sum()
    )

    n_negative = int(
        len(train_y)
        -
        n_positive
    )


    scale_pos_weight = (
        n_negative
        /
        max(
            n_positive,
            1,
        )
    )


    print(
        f"Train rows: "
        f"{len(train):,}"
    )

    print(
        f"Train positives: "
        f"{n_positive:,}"
    )

    print(
        f"Evaluation positives: "
        f"{binary_y.sum():,}"
    )


    # ========================================================
    # FIT BASE MODELS
    # ========================================================

    raw_train_normal = {}
    raw_evaluation = {}


    model_specs = [
        (
            "LightGBM Current",
            build_lightgbm(),
            "supervised",
        ),

        (
            "XGBoost Current",
            build_xgboost(
                scale_pos_weight
            ),
            "supervised",
        ),

        (
            "Random Forest Current",
            build_random_forest(),
            "supervised",
        ),

        (
            "Isolation Forest Current",
            build_isolation_forest(),
            "anomaly",
        ),
    ]


    for (
        model_name,
        model,
        model_type,
    ) in model_specs:


        print(
            f"Training "
            f"{model_name}..."
        )


        if model_type == "supervised":

            with warnings.catch_warnings():

                warnings.simplefilter(
                    "ignore"
                )


                model.fit(
                    train[
                        CURRENT_FEATURES
                    ],
                    train_y,
                )


            raw_train_normal[
                model_name
            ] = (
                model.predict_proba(
                    train_normal[
                        CURRENT_FEATURES
                    ]
                )[:, 1]
            )


            raw_evaluation[
                model_name
            ] = (
                model.predict_proba(
                    evaluation[
                        CURRENT_FEATURES
                    ]
                )[:, 1]
            )


        else:

            model.fit(
                train_normal[
                    CURRENT_FEATURES
                ]
            )


            raw_train_normal[
                model_name
            ] = (
                -
                model.decision_function(
                    train_normal[
                        CURRENT_FEATURES
                    ]
                )
            )


            raw_evaluation[
                model_name
            ] = (
                -
                model.decision_function(
                    evaluation[
                        CURRENT_FEATURES
                    ]
                )
            )


    # ========================================================
    # TRAIN-NORMAL PERCENTILE NORMALIZATION
    # ========================================================

    train_percentiles = {}
    eval_percentiles = {}


    for model_name in (
        raw_train_normal.keys()
    ):

        reference = (
            raw_train_normal[
                model_name
            ]
        )


        train_percentiles[
            model_name
        ] = empirical_percentile(
            reference,
            reference,
        )


        eval_percentiles[
            model_name
        ] = empirical_percentile(
            reference,
            raw_evaluation[
                model_name
            ],
        )


    # ========================================================
    # SCORE DIVERSITY / CORRELATION
    # ========================================================

    base_eval_df = pd.DataFrame({
        name:
            values
        for name, values
        in eval_percentiles.items()
    })


    correlation = (
        base_eval_df
        .corr(
            method="spearman"
        )
    )


    for row_model in (
        correlation.index
    ):

        for column_model in (
            correlation.columns
        ):

            if (
                row_model
                >= column_model
            ):

                continue


            correlation_rows.append({
                "split":
                    split_name,

                "model_a":
                    row_model,

                "model_b":
                    column_model,

                "spearman_correlation":
                    float(
                        correlation.loc[
                            row_model,
                            column_model,
                        ]
                    ),
            })


    # ========================================================
    # CANDIDATE RISKS
    # ========================================================

    candidate_train = {
        "LightGBM Current":
            train_percentiles[
                "LightGBM Current"
            ],

        "XGBoost Current":
            train_percentiles[
                "XGBoost Current"
            ],

        "Random Forest Current":
            train_percentiles[
                "Random Forest Current"
            ],

        "Isolation Forest Current":
            train_percentiles[
                "Isolation Forest Current"
            ],
    }


    candidate_eval = {
        "LightGBM Current":
            eval_percentiles[
                "LightGBM Current"
            ],

        "XGBoost Current":
            eval_percentiles[
                "XGBoost Current"
            ],

        "Random Forest Current":
            eval_percentiles[
                "Random Forest Current"
            ],

        "Isolation Forest Current":
            eval_percentiles[
                "Isolation Forest Current"
            ],
    }


    # --------------------------------------------------------
    # LGBM + XGB
    # --------------------------------------------------------

    candidate_train[
        "LGBM_XGB_50_50"
    ] = (
        0.5
        *
        train_percentiles[
            "LightGBM Current"
        ]
        +
        0.5
        *
        train_percentiles[
            "XGBoost Current"
        ]
    )


    candidate_eval[
        "LGBM_XGB_50_50"
    ] = (
        0.5
        *
        eval_percentiles[
            "LightGBM Current"
        ]
        +
        0.5
        *
        eval_percentiles[
            "XGBoost Current"
        ]
    )


    # --------------------------------------------------------
    # LGBM + Isolation Forest
    # --------------------------------------------------------

    candidate_train[
        "LGBM_IF_80_20"
    ] = (
        0.8
        *
        train_percentiles[
            "LightGBM Current"
        ]
        +
        0.2
        *
        train_percentiles[
            "Isolation Forest Current"
        ]
    )


    candidate_eval[
        "LGBM_IF_80_20"
    ] = (
        0.8
        *
        eval_percentiles[
            "LightGBM Current"
        ]
        +
        0.2
        *
        eval_percentiles[
            "Isolation Forest Current"
        ]
    )


    # --------------------------------------------------------
    # LGBM + XGB + Isolation Forest
    # --------------------------------------------------------

    candidate_train[
        "LGBM_XGB_IF_40_40_20"
    ] = (
        0.4
        *
        train_percentiles[
            "LightGBM Current"
        ]
        +
        0.4
        *
        train_percentiles[
            "XGBoost Current"
        ]
        +
        0.2
        *
        train_percentiles[
            "Isolation Forest Current"
        ]
    )


    candidate_eval[
        "LGBM_XGB_IF_40_40_20"
    ] = (
        0.4
        *
        eval_percentiles[
            "LightGBM Current"
        ]
        +
        0.4
        *
        eval_percentiles[
            "XGBoost Current"
        ]
        +
        0.2
        *
        eval_percentiles[
            "Isolation Forest Current"
        ]
    )


    # --------------------------------------------------------
    # Maximum-risk hybrid
    # --------------------------------------------------------

    candidate_train[
        "MAX_LGBM_IF"
    ] = np.maximum(
        train_percentiles[
            "LightGBM Current"
        ],
        train_percentiles[
            "Isolation Forest Current"
        ],
    )


    candidate_eval[
        "MAX_LGBM_IF"
    ] = np.maximum(
        eval_percentiles[
            "LightGBM Current"
        ],
        eval_percentiles[
            "Isolation Forest Current"
        ],
    )


    # ========================================================
    # EVALUATE CANDIDATES
    # ========================================================

    for candidate_name in (
        candidate_eval.keys()
    ):

        eval_risk = (
            candidate_eval[
                candidate_name
            ]
        )


        train_risk = (
            candidate_train[
                candidate_name
            ]
        )


        ranking = ranking_metrics(
            binary_y,
            eval_risk[
                binary_mask
            ],
        )


        ranking_rows.append({
            "split":
                split_name,

            "validation_event":
                validation_event,

            "candidate":
                candidate_name,

            **ranking,
        })


        for quantile in (
            THRESHOLD_QUANTILES
        ):

            threshold = float(
                np.quantile(
                    train_risk,
                    quantile,
                )
            )


            operation = (
                operational_metrics(
                    evaluation,
                    eval_risk,
                    threshold,
                    event_start,
                )
            )


            operational_rows.append({
                "split":
                    split_name,

                "validation_event":
                    validation_event,

                "candidate":
                    candidate_name,

                "threshold_quantile":
                    quantile,

                "threshold_value":
                    threshold,

                **operation,
            })


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

        pred["candidate"] = (
            candidate_name
        )

        pred["risk"] = (
            eval_risk
        )


        prediction_parts.append(
            pred
        )


# ============================================================
# TABLES
# ============================================================

ranking_df = pd.DataFrame(
    ranking_rows
)

operational_df = pd.DataFrame(
    operational_rows
)

correlation_df = pd.DataFrame(
    correlation_rows
)

predictions_df = pd.concat(
    prediction_parts,
    ignore_index=True,
)


ranking_df.to_csv(
    METRIC_DIR
    / "ensemble_challenger_ranking.csv",
    index=False,
)


operational_df.to_csv(
    METRIC_DIR
    / "ensemble_challenger_operational.csv",
    index=False,
)


correlation_df.to_csv(
    TABLE_DIR
    / "ensemble_base_model_correlations.csv",
    index=False,
)


predictions_df.to_parquet(
    METRIC_DIR
    / "ensemble_challenger_predictions.parquet",
    index=False,
)


# ============================================================
# MACRO RANKING
# ============================================================

ranking_scoreboard = (
    ranking_df
    .groupby(
        "candidate",
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
    / "ensemble_challenger_ranking_scoreboard.csv",
    index=False,
)


# ============================================================
# MACRO OPERATIONAL
# ============================================================

operational_scoreboard = (
    operational_df
    .groupby(
        [
            "candidate",
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
    )
)


operational_scoreboard[
    "events_total"
] = 2


operational_scoreboard.to_csv(
    TABLE_DIR
    / "ensemble_challenger_operational_scoreboard.csv",
    index=False,
)


# ============================================================
# QUALIFIED CONFIGURATIONS
# ============================================================

qualified = operational_scoreboard[
    (
        operational_scoreboard[
            "events_detected"
        ] == 2
    )
    &
    (
        operational_scoreboard[
            "median_detection_delay_minutes"
        ] <= 5
    )
].copy()


qualified = qualified.merge(
    ranking_scoreboard[
        [
            "candidate",
            "mean_pr_auc",
            "minimum_event_pr_auc",
        ]
    ],
    on="candidate",
    how="left",
)


qualified = (
    qualified
    .sort_values(
        [
            "worst_false_alarms_per_day",
            "minimum_event_pr_auc",
            "mean_pr_auc",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )
)


qualified.to_csv(
    TABLE_DIR
    / "ensemble_challenger_qualified.csv",
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "experiment":
        "ensemble_challenger",

    "development_events": [
        "F02",
        "F03",
    ],

    "F04_used_for_selection":
        False,

    "feature_count":
        len(
            CURRENT_FEATURES
        ),

    "score_normalization":
        "training_normal_empirical_percentile",

    "persistence_minutes":
        PERSISTENCE,

    "candidate_count":
        int(
            ranking_scoreboard.shape[0]
        ),

    "qualified_configuration_count":
        int(
            len(qualified)
        ),
}


with open(
    METRIC_DIR
    / "ensemble_challenger_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# DISPLAY
# ============================================================

pd.set_option(
    "display.max_columns",
    None,
)

pd.set_option(
    "display.width",
    260,
)


print("\n" + "=" * 100)
print("ENSEMBLE RANKING SCOREBOARD")
print("=" * 100)

print(
    ranking_scoreboard
    .sort_values(
        "mean_pr_auc",
        ascending=False,
    )
    .round(5)
    .to_string(index=False)
)


print("\n" + "=" * 100)
print("QUALIFIED OPERATIONAL CONFIGURATIONS")
print("=" * 100)

print(
    qualified
    .round(5)
    .to_string(index=False)
)


print("\n" + "=" * 100)
print("BASE MODEL SCORE CORRELATIONS")
print("=" * 100)

print(
    correlation_df
    .round(4)
    .to_string(index=False)
)


print("\nIMPORTANT:")
print(
    "F04 was not used in this "
    "challenger selection experiment."
)

print("\nENSEMBLE CHALLENGER EXPERIMENT COMPLETE")
