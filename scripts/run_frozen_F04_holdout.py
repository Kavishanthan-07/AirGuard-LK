from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import yaml

from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    auc,
    precision_recall_curve,
    roc_auc_score,
)


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
    / "frozen_final_model.yaml"
)

MODEL_DIR = ROOT / "artifacts" / "models"
METRIC_DIR = ROOT / "artifacts" / "metrics"
TABLE_DIR = ROOT / "artifacts" / "tables"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD FROZEN CONFIG
# ============================================================

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8",
) as f:

    config = yaml.safe_load(f)


with open(
    CONFIG_PATH,
    "rb",
) as f:

    config_sha256 = (
        hashlib.sha256(
            f.read()
        ).hexdigest()
    )


THRESHOLD_QUANTILE = float(
    config["alerting"][
        "threshold_quantile"
    ]
)

PERSISTENCE = int(
    config["alerting"][
        "persistence_minutes"
    ]
)

PRE_EXCLUSION = int(
    config["data"][
        "pre_event_exclusion_minutes"
    ]
)

RECOVERY = int(
    config["data"][
        "post_event_recovery_minutes"
    ]
)

SEED = int(
    config["random_seed"]
)


TRAINING_EVENTS = list(
    config["final_evaluation"][
        "training_events"
    ]
)

HOLDOUT_EVENT = (
    config["final_evaluation"][
        "holdout_event"
    ]
)


assert TRAINING_EVENTS == [
    "F01",
    "F02",
    "F03",
]

assert HOLDOUT_EVENT == "F04"

assert THRESHOLD_QUANTILE == 0.9995


# ============================================================
# FEATURE SET
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
# LOAD DATA
# ============================================================

print("=" * 92)
print("AIRGUARD-LK — FROZEN F04 CHRONOLOGICAL HOLDOUT")
print("=" * 92)

print()
print(
    f"Frozen config SHA256: "
    f"{config_sha256}"
)

print(
    f"Threshold quantile: "
    f"{THRESHOLD_QUANTILE}"
)

print(
    f"Persistence: "
    f"{PERSISTENCE} minutes"
)

print(
    f"Features: "
    f"{len(CURRENT_FEATURES)}"
)


master = pd.read_parquet(
    MASTER_PATH
)

master["timestamp"] = pd.to_datetime(
    master["timestamp"]
)

master = (
    master
    .sort_values("timestamp")
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
    events
    .sort_values("start_time")
    .reset_index(drop=True)
)


event_lookup = (
    events
    .set_index("event_id")
    .to_dict("index")
)


def event_info(event_id):

    item = event_lookup[
        event_id
    ]

    return (
        pd.Timestamp(
            item["start_time"]
        ),
        pd.Timestamp(
            item["end_time"]
        ),
    )


ELIGIBLE = (
    "quality__model_feature_eligible"
)


# ============================================================
# BUILD FINAL TRAIN / HOLDOUT ROLES
# ============================================================

last_training_event = (
    TRAINING_EVENTS[-1]
)

_, last_training_end = (
    event_info(
        last_training_event
    )
)


evaluation_start = (
    last_training_end
    +
    pd.Timedelta(
        minutes=RECOVERY
    )
)


holdout_start, holdout_end = (
    event_info(
        HOLDOUT_EVENT
    )
)


pre_event_start = (
    holdout_start
    -
    pd.Timedelta(
        minutes=PRE_EXCLUSION
    )
)


# Causal timestamp convention:
# event begins at T.
# Feature at T contains data only up to T.
# First feature containing fault measurements is T+1 min.
holdout_fault_end = (
    holdout_end
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


# ------------------------------------------------------------
# TRAINING PERIOD
# ------------------------------------------------------------

train_period = (
    master["timestamp"]
    < evaluation_start
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


for event_id in TRAINING_EVENTS:

    event_start, event_end = (
        event_info(
            event_id
        )
    )


    training_pre_start = (
        event_start
        -
        pd.Timedelta(
            minutes=PRE_EXCLUSION
        )
    )


    training_fault_end = (
        event_end
        +
        pd.Timedelta(
            minutes=1
        )
    )


    recovery_end = (
        event_end
        +
        pd.Timedelta(
            minutes=RECOVERY
        )
    )


    # Pre-event context excluded
    mask = (
        train_period
        &
        eligible
        &
        (
            master["timestamp"]
            > training_pre_start
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
    ] = "train_pre_event_context"


    # Documented failure
    mask = (
        train_period
        &
        eligible
        &
        (
            master["timestamp"]
            > event_start
        )
        &
        (
            master["timestamp"]
            <= training_fault_end
        )
    )

    roles.loc[
        mask,
        "role"
    ] = "train_failure"


    # Recovery excluded
    mask = (
        train_period
        &
        (
            master["timestamp"]
            > training_fault_end
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


# ------------------------------------------------------------
# HOLDOUT NORMAL PERIOD
# ------------------------------------------------------------

mask = (
    eligible
    &
    (
        master["timestamp"]
        >= evaluation_start
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
] = "holdout_negative"


# ------------------------------------------------------------
# HOLDOUT PRE-EVENT CONTEXT
# ------------------------------------------------------------

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
        <= holdout_start
    )
)

roles.loc[
    mask,
    "role"
] = "holdout_pre_event_context"


# ------------------------------------------------------------
# HOLDOUT DOCUMENTED FAILURE
# ------------------------------------------------------------

mask = (
    eligible
    &
    (
        master["timestamp"]
        > holdout_start
    )
    &
    (
        master["timestamp"]
        <= holdout_fault_end
    )
)

roles.loc[
    mask,
    "role"
] = "holdout_failure"


active_roles = roles[
    roles["role"]
    != "outside"
].copy()


data = master.merge(
    active_roles,
    on="timestamp",
    how="inner",
    validate="one_to_one",
)


# ============================================================
# TRAINING
# ============================================================

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


print()
print(
    f"Training rows: "
    f"{len(train):,}"
)

print(
    f"Training fault rows: "
    f"{train_y.sum():,}"
)

print(
    f"Training normal rows: "
    f"{len(train_normal):,}"
)


# Hard leakage assertion
assert (
    train["timestamp"].max()
    <
    evaluation_start
)


# ============================================================
# FIT FROZEN MODEL
# ============================================================

params = (
    config["model"][
        "parameters"
    ]
)


model = LGBMClassifier(
    objective=params[
        "objective"
    ],

    n_estimators=int(
        params[
            "n_estimators"
        ]
    ),

    learning_rate=float(
        params[
            "learning_rate"
        ]
    ),

    num_leaves=int(
        params[
            "num_leaves"
        ]
    ),

    max_depth=int(
        params[
            "max_depth"
        ]
    ),

    min_child_samples=int(
        params[
            "min_child_samples"
        ]
    ),

    subsample=float(
        params[
            "subsample"
        ]
    ),

    colsample_bytree=float(
        params[
            "colsample_bytree"
        ]
    ),

    reg_lambda=float(
        params[
            "reg_lambda"
        ]
    ),

    class_weight=params[
        "class_weight"
    ],

    random_state=SEED,

    n_jobs=-1,

    verbosity=-1,
)


model.fit(
    train[
        CURRENT_FEATURES
    ],
    train_y,
)


# ============================================================
# DERIVE FROZEN QUANTILE THRESHOLD
# ============================================================

train_normal_scores = (
    model.predict_proba(
        train_normal[
            CURRENT_FEATURES
        ]
    )[:, 1]
)


threshold = float(
    np.quantile(
        train_normal_scores,
        THRESHOLD_QUANTILE,
    )
)


print()
print(
    f"Final numeric threshold "
    f"from training normals: "
    f"{threshold:.10f}"
)


# ============================================================
# HOLDOUT
# ============================================================

holdout = data[
    data["role"].isin([
        "holdout_negative",
        "holdout_pre_event_context",
        "holdout_failure",
    ])
].copy()


holdout = (
    holdout
    .sort_values("timestamp")
    .reset_index(drop=True)
)


scores = (
    model.predict_proba(
        holdout[
            CURRENT_FEATURES
        ]
    )[:, 1]
)


holdout["score"] = scores


# ============================================================
# RANKING METRICS
# ============================================================

binary_mask = (
    holdout["role"]
    .isin([
        "holdout_negative",
        "holdout_failure",
    ])
    .to_numpy()
)


y_true = (
    holdout.loc[
        binary_mask,
        "role",
    ]
    .eq(
        "holdout_failure"
    )
    .astype(int)
    .to_numpy()
)


binary_scores = (
    scores[
        binary_mask
    ]
)


average_precision = (
    average_precision_score(
        y_true,
        binary_scores,
    )
)


precision_curve, recall_curve, _ = (
    precision_recall_curve(
        y_true,
        binary_scores,
    )
)


pr_auc = auc(
    recall_curve,
    precision_curve,
)


roc_auc = roc_auc_score(
    y_true,
    binary_scores,
)


prevalence = float(
    y_true.mean()
)


ap_lift = (
    average_precision
    /
    prevalence
)


# ============================================================
# PERSISTENT ALERTS
# ============================================================

timestamps = (
    holdout[
        "timestamp"
    ]
    .reset_index(drop=True)
)


alert = np.zeros(
    len(holdout),
    dtype=bool,
)


streak = 0
previous_time = None


for i in range(
    len(holdout)
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


    if streak >= PERSISTENCE:

        alert[i] = True


    previous_time = (
        current_time
    )


holdout["alert"] = (
    alert
)


# ============================================================
# FALSE ALARMS
# ============================================================

negative = holdout[
    holdout["role"]
    == "holdout_negative"
].copy()


negative_alert_times = (
    negative.loc[
        negative["alert"],
        "timestamp",
    ]
    .sort_values()
    .reset_index(drop=True)
)


false_alarm_episodes = 0


if len(
    negative_alert_times
) > 0:

    false_alarm_episodes = 1

    for i in range(
        1,
        len(
            negative_alert_times
        )
    ):

        delta = (
            negative_alert_times.iloc[i]
            -
            negative_alert_times.iloc[
                i - 1
            ]
        ).total_seconds()

        if delta > 60:

            false_alarm_episodes += 1


negative_days = (
    len(negative)
    / 1440.0
)


false_alarms_per_day = (
    false_alarm_episodes
    /
    negative_days
    if negative_days > 0
    else np.nan
)


# ============================================================
# EVENT DETECTION
# ============================================================

failure = holdout[
    holdout["role"]
    == "holdout_failure"
]


failure_alerts = failure[
    failure["alert"]
]


event_detected = (
    len(
        failure_alerts
    ) > 0
)


if event_detected:

    first_failure_alert = (
        failure_alerts[
            "timestamp"
        ].min()
    )

    detection_delay_minutes = (
        first_failure_alert
        -
        holdout_start
    ).total_seconds() / 60.0

else:

    first_failure_alert = pd.NaT
    detection_delay_minutes = np.nan


# ============================================================
# PRE-EVENT WARNING DIAGNOSTIC
# ============================================================

pre_event = holdout[
    holdout["role"]
    == "holdout_pre_event_context"
]


pre_event_alerts = pre_event[
    pre_event["alert"]
]


if len(
    pre_event_alerts
) > 0:

    first_pre_event_alert = (
        pre_event_alerts[
            "timestamp"
        ].min()
    )

    pre_event_warning_minutes = (
        holdout_start
        -
        first_pre_event_alert
    ).total_seconds() / 60.0

else:

    first_pre_event_alert = pd.NaT
    pre_event_warning_minutes = np.nan


# ============================================================
# FIRST ALERT ACROSS PRE-EVENT + FAILURE
# ============================================================

event_neighborhood = holdout[
    holdout["role"].isin([
        "holdout_pre_event_context",
        "holdout_failure",
    ])
]


event_neighborhood_alerts = (
    event_neighborhood[
        event_neighborhood[
            "alert"
        ]
    ]
)


if len(
    event_neighborhood_alerts
) > 0:

    first_neighborhood_alert = (
        event_neighborhood_alerts[
            "timestamp"
        ].min()
    )

    first_alert_relative_to_event = (
        first_neighborhood_alert
        -
        holdout_start
    ).total_seconds() / 60.0

else:

    first_neighborhood_alert = pd.NaT
    first_alert_relative_to_event = np.nan


# ============================================================
# RAPID-DETECTION CHECKPOINTS
# ============================================================

checkpoint_results = {}


for minutes in [
    5,
    15,
    30,
    60,
]:

    deadline = (
        holdout_start
        +
        pd.Timedelta(
            minutes=minutes
        )
    )

    detected = bool(
        (
            failure[
                "timestamp"
            ]
            <= deadline
        )
        &
        (
            failure[
                "alert"
            ]
        )
    ).any()

    checkpoint_results[
        f"detected_within_{minutes}m"
    ] = int(
        detected
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature":
        CURRENT_FEATURES,

    "gain":
        model.booster_
        .feature_importance(
            importance_type="gain"
        ),

    "split":
        model.booster_
        .feature_importance(
            importance_type="split"
        ),
})


importance = (
    importance
    .sort_values(
        "gain",
        ascending=False,
    )
    .reset_index(drop=True)
)


importance.to_csv(
    TABLE_DIR
    / "final_F04_feature_importance.csv",
    index=False,
)


# ============================================================
# SAVE MODEL + PREDICTIONS
# ============================================================

joblib.dump(
    {
        "model": model,
        "features": CURRENT_FEATURES,
        "threshold_quantile":
            THRESHOLD_QUANTILE,
        "threshold_value":
            threshold,
        "persistence_minutes":
            PERSISTENCE,
        "config_sha256":
            config_sha256,
    },

    MODEL_DIR
    / "airguard_F04_frozen_model.joblib",
)


holdout.to_parquet(
    METRIC_DIR
    / "final_F04_predictions.parquet",
    index=False,
)


# ============================================================
# FINAL RESULT
# ============================================================

result = {
    "holdout_event":
        HOLDOUT_EVENT,

    "training_events":
        TRAINING_EVENTS,

    "frozen_config_sha256":
        config_sha256,

    "model":
        "LightGBM Current",

    "feature_count":
        len(
            CURRENT_FEATURES
        ),

    "threshold_quantile":
        THRESHOLD_QUANTILE,

    "threshold_value":
        threshold,

    "persistence_minutes":
        PERSISTENCE,

    "holdout_normal_minutes":
        int(
            len(negative)
        ),

    "holdout_failure_minutes":
        int(
            len(failure)
        ),

    "positive_prevalence":
        prevalence,

    "pr_auc":
        float(pr_auc),

    "average_precision":
        float(
            average_precision
        ),

    "roc_auc":
        float(
            roc_auc
        ),

    "ap_lift_over_prevalence":
        float(
            ap_lift
        ),

    "event_detected":
        int(
            event_detected
        ),

    "detection_delay_minutes":
        float(
            detection_delay_minutes
        )
        if np.isfinite(
            detection_delay_minutes
        )
        else None,

    "false_alarm_episodes":
        int(
            false_alarm_episodes
        ),

    "false_alarm_episodes_per_day":
        float(
            false_alarms_per_day
        ),

    "pre_event_warning_minutes":
        float(
            pre_event_warning_minutes
        )
        if np.isfinite(
            pre_event_warning_minutes
        )
        else None,

    "first_alert_relative_to_event_minutes":
        float(
            first_alert_relative_to_event
        )
        if np.isfinite(
            first_alert_relative_to_event
        )
        else None,

    **checkpoint_results,
}


with open(
    METRIC_DIR
    / "final_F04_holdout_result.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        result,
        f,
        indent=2,
    )


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 92)
print("FROZEN F04 HOLDOUT RESULT")
print("=" * 92)


for key, value in (
    result.items()
):

    print(
        f"{key}: {value}"
    )


print("\nTOP 15 FEATURES BY GAIN")
print("-" * 92)

print(
    importance.head(15)
    .to_string(index=False)
)


print("\n" + "=" * 92)

print(
    "FINAL HOLDOUT EVALUATION COMPLETE"
)

print(
    "Do not tune the model based on "
    "this result."
)
