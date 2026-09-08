from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

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

MODEL_PATH = (
    ROOT
    / "artifacts"
    / "models"
    / "airguard_F04_frozen_model.joblib"
)

OUTPUT_PATH = (
    ROOT
    / "artifacts"
    / "metrics"
    / "final_F04_current_eligibility_sensitivity.json"
)


print("=" * 94)
print("AIRGUARD-LK — POST-HOC CURRENT-FEATURE ELIGIBILITY SENSITIVITY")
print("=" * 94)

print(
    "\nIMPORTANT: No model fitting or threshold "
    "selection occurs in this analysis."
)


# ============================================================
# LOAD FROZEN MODEL
# ============================================================

bundle = joblib.load(
    MODEL_PATH
)

model = bundle["model"]
features = bundle["features"]

threshold = float(
    bundle["threshold_value"]
)

persistence = int(
    bundle["persistence_minutes"]
)


print(
    f"\nFrozen threshold: {threshold:.10f}"
)

print(
    f"Frozen persistence: {persistence}"
)

print(
    f"Frozen feature count: {len(features)}"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_parquet(
    MASTER_PATH
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = (
    df.sort_values("timestamp")
      .reset_index(drop=True)
)


events = pd.read_csv(
    EVENT_PATH,
    parse_dates=[
        "start_time",
        "end_time",
    ],
)


lookup = (
    events
    .set_index("event_id")
    .to_dict("index")
)


def event_info(event_id):

    item = lookup[event_id]

    return (
        pd.Timestamp(
            item["start_time"]
        ),
        pd.Timestamp(
            item["end_time"]
        ),
    )


# ============================================================
# FROZEN BOUNDARIES
# ============================================================

_, f03_end = event_info(
    "F03"
)

f04_start, f04_end = (
    event_info(
        "F04"
    )
)


evaluation_start = (
    f03_end
    +
    pd.Timedelta(
        minutes=360
    )
)


pre_event_start = (
    f04_start
    -
    pd.Timedelta(
        minutes=360
    )
)


fault_end = (
    f04_end
    +
    pd.Timedelta(
        minutes=1
    )
)


# ============================================================
# MODEL-SPECIFIC CURRENT ELIGIBILITY
#
# Unlike the temporal model, the final LightGBM Current model
# does not require 120 minutes of continuous history.
#
# We still reject:
#   - the minute receiving a >30 s raw gap
#   - low-sample minutes
#   - any row missing a selected current feature
# ============================================================

feature_complete = (
    df[features]
    .notna()
    .all(axis=1)
)


current_eligible = (
    feature_complete
    &
    (
        df[
            "quality__low_sample_minute"
        ] == 0
    )
    &
    (
        df[
            "quality__raw_gap_over_30s"
        ] == 0
    )
)


strict_eligible = (
    df[
        "quality__model_feature_eligible"
    ] == 1
)


df[
    "current_feature_eligible"
] = current_eligible


# ============================================================
# ROLES
# ============================================================

df["sensitivity_role"] = (
    "outside"
)


normal_mask = (
    current_eligible
    &
    (
        df["timestamp"]
        >= evaluation_start
    )
    &
    (
        df["timestamp"]
        <= pre_event_start
    )
)


pre_mask = (
    current_eligible
    &
    (
        df["timestamp"]
        > pre_event_start
    )
    &
    (
        df["timestamp"]
        <= f04_start
    )
)


failure_mask = (
    current_eligible
    &
    (
        df["timestamp"]
        > f04_start
    )
    &
    (
        df["timestamp"]
        <= fault_end
    )
)


df.loc[
    normal_mask,
    "sensitivity_role"
] = "holdout_negative"


df.loc[
    pre_mask,
    "sensitivity_role"
] = "holdout_pre_event_context"


df.loc[
    failure_mask,
    "sensitivity_role"
] = "holdout_failure"


holdout = df[
    df["sensitivity_role"]
    != "outside"
].copy()


holdout = (
    holdout
    .sort_values("timestamp")
    .reset_index(drop=True)
)


# ============================================================
# SCORE USING FROZEN MODEL ONLY
# ============================================================

holdout["score"] = (
    model.predict_proba(
        holdout[features]
    )[:, 1]
)


# ============================================================
# PERSISTENCE
# ============================================================

alerts = np.zeros(
    len(holdout),
    dtype=bool,
)


streak = 0
previous_timestamp = None


for i, row in holdout.iterrows():

    timestamp = row["timestamp"]


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


    if row["score"] >= threshold:

        streak += 1

    else:

        streak = 0


    if streak >= persistence:

        alerts[i] = True


    previous_timestamp = timestamp


holdout["alert"] = alerts


# ============================================================
# BINARY RANKING
# ============================================================

binary = holdout[
    holdout[
        "sensitivity_role"
    ].isin([
        "holdout_negative",
        "holdout_failure",
    ])
].copy()


y_true = (
    binary[
        "sensitivity_role"
    ]
    .eq(
        "holdout_failure"
    )
    .astype(int)
    .to_numpy()
)


scores = (
    binary["score"]
    .to_numpy()
)


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
    y_true.mean()
)


# ============================================================
# FALSE ALARMS
# ============================================================

normal = holdout[
    holdout[
        "sensitivity_role"
    ]
    == "holdout_negative"
].copy()


normal_alert_times = (
    normal.loc[
        normal["alert"],
        "timestamp",
    ]
    .sort_values()
    .reset_index(drop=True)
)


false_alarm_episodes = 0


if len(normal_alert_times):

    false_alarm_episodes = 1

    for i in range(
        1,
        len(normal_alert_times)
    ):

        if (
            normal_alert_times.iloc[i]
            -
            normal_alert_times.iloc[i - 1]
        ).total_seconds() > 60:

            false_alarm_episodes += 1


normal_days = (
    len(normal)
    / 1440.0
)


false_alarms_per_day = (
    false_alarm_episodes
    / normal_days
    if normal_days > 0
    else np.nan
)


# ============================================================
# EVENT DETECTION
# ============================================================

failure = holdout[
    holdout[
        "sensitivity_role"
    ]
    == "holdout_failure"
].copy()


failure_alerts = failure[
    failure["alert"]
]


event_detected = (
    len(failure_alerts) > 0
)


if event_detected:

    first_alert = (
        failure_alerts[
            "timestamp"
        ].min()
    )

    detection_delay = (
        first_alert
        -
        f04_start
    ).total_seconds() / 60.0

else:

    first_alert = pd.NaT
    detection_delay = np.nan


# ============================================================
# COVERAGE COMPARISON
# ============================================================

event_clock_rows = df[
    (
        df["timestamp"]
        > f04_start
    )
    &
    (
        df["timestamp"]
        <= fault_end
    )
]


strict_fault_rows = int(
    strict_eligible.loc[
        event_clock_rows.index
    ].sum()
)


current_fault_rows = int(
    current_eligible.loc[
        event_clock_rows.index
    ].sum()
)


# ============================================================
# RESULT
# ============================================================

result = {
    "analysis_type":
        "post_hoc_eligibility_sensitivity",

    "model_retrained":
        False,

    "threshold_changed":
        False,

    "model":
        "LightGBM Current",

    "frozen_threshold":
        threshold,

    "persistence_minutes":
        persistence,

    "f04_event_clock_rows":
        int(
            len(event_clock_rows)
        ),

    "strict_primary_fault_rows":
        strict_fault_rows,

    "current_feature_eligible_fault_rows":
        current_fault_rows,

    "additional_fault_rows_scored":
        int(
            current_fault_rows
            -
            strict_fault_rows
        ),

    "holdout_normal_minutes":
        int(
            len(normal)
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
        float(ap),

    "roc_auc":
        float(roc_auc),

    "event_detected":
        int(event_detected),

    "detection_delay_minutes":
        float(detection_delay)
        if np.isfinite(
            detection_delay
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
}


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        result,
        f,
        indent=2,
    )


print("\n" + "=" * 94)
print("SENSITIVITY RESULT")
print("=" * 94)


for key, value in result.items():

    print(
        f"{key}: {value}"
    )


print("\nThis does NOT replace the frozen primary F04 result.")
