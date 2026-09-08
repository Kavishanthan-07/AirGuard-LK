from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

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

FROZEN_MODEL_PATH = (
    ROOT
    / "artifacts"
    / "models"
    / "airguard_F04_frozen_model.joblib"
)

TABLE_DIR = ROOT / "artifacts" / "tables"
FIGURE_DIR = ROOT / "artifacts" / "figures"
METRIC_DIR = ROOT / "artifacts" / "metrics"

for p in [
    TABLE_DIR,
    FIGURE_DIR,
    METRIC_DIR,
]:
    p.mkdir(
        parents=True,
        exist_ok=True,
    )


SEED = 42


# ============================================================
# 71 CURRENT / STATE-AWARE FEATURES
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
    CURRENT_FEATURES.extend([
        f"{sensor}_mean",
        f"{sensor}_last",
        f"{sensor}_transition_sum",
    ])


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

print("=" * 100)
print("AIRGUARD-LK — SHAP + EVENT-LEVEL EXPLAINABILITY")
print("=" * 100)


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


event_lookup = (
    events
    .set_index("event_id")
    .to_dict("index")
)


bundle = joblib.load(
    FROZEN_MODEL_PATH
)


frozen_model = bundle["model"]
frozen_features = bundle["features"]

threshold = float(
    bundle["threshold_value"]
)

persistence = int(
    bundle["persistence_minutes"]
)


assert frozen_features == CURRENT_FEATURES


print(
    f"\nFrozen threshold: "
    f"{threshold:.10f}"
)

print(
    f"Persistence: "
    f"{persistence} minutes"
)

print(
    f"Feature count: "
    f"{len(CURRENT_FEATURES)}"
)


# ============================================================
# EXACT DEVELOPMENT LIGHTGBM
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


# ============================================================
# SHAP HELPER
# ============================================================

def get_shap_values(
    model,
    X,
):

    explainer = shap.TreeExplainer(
        model
    )

    values = explainer.shap_values(
        X,
        check_additivity=False,
    )


    # Older SHAP binary-classifier API
    if isinstance(
        values,
        list,
    ):
        values = values[-1]


    values = np.asarray(
        values
    )


    # Handle possible
    # samples x features x classes
    if (
        values.ndim == 3
        and
        values.shape[-1] == 2
    ):
        values = values[:, :, 1]


    if values.shape != X.shape:
        raise RuntimeError(
            f"Unexpected SHAP shape "
            f"{values.shape}; "
            f"expected {X.shape}"
        )


    return values


# ============================================================
# CURRENT-FEATURE ELIGIBILITY
# ============================================================

feature_complete = (
    master[
        CURRENT_FEATURES
    ]
    .notna()
    .all(axis=1)
)


master[
    "current_feature_eligible"
] = (
    feature_complete
    &
    (
        master[
            "quality__low_sample_minute"
        ] == 0
    )
    &
    (
        master[
            "quality__raw_gap_over_30s"
        ] == 0
    )
)


# ============================================================
# PERSISTENCE
# ============================================================

def persistent_alert(
    timestamps,
    scores,
    threshold_value,
    persistence_minutes,
):

    timestamps = pd.Series(
        pd.to_datetime(
            timestamps
        )
    ).reset_index(
        drop=True
    )

    scores = np.asarray(
        scores
    )

    alert = np.zeros(
        len(scores),
        dtype=bool,
    )

    streak = 0
    previous = None


    for i in range(
        len(scores)
    ):

        ts = timestamps.iloc[i]


        if previous is not None:

            if (
                ts - previous
            ).total_seconds() != 60:

                streak = 0


        if scores[i] >= threshold_value:
            streak += 1

        else:
            streak = 0


        if streak >= persistence_minutes:
            alert[i] = True


        previous = ts


    return alert


# ============================================================
# EXPLAIN DEVELOPMENT EVENTS F02 / F03
# ============================================================

event_importance_parts = []
event_score_summary = []


for split_name, event_id in [
    (
        "fault_F02",
        "F02",
    ),
    (
        "fault_F03",
        "F03",
    ),
]:

    print(
        "\n"
        + "-" * 100
    )

    print(
        f"Explaining {event_id}"
    )


    split_roles = (
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
        split_roles,
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


    y_train = (
        train["role"]
        .eq(
            "train_failure"
        )
        .astype(int)
        .to_numpy()
    )


    model = build_lightgbm()

    model.fit(
        train[
            CURRENT_FEATURES
        ],
        y_train,
    )


    failure = (
        data[
            data["role"]
            == "evaluation_failure"
        ]
        .sort_values("timestamp")
        .copy()
    )


    X_failure = (
        failure[
            CURRENT_FEATURES
        ]
    )


    scores = model.predict_proba(
        X_failure
    )[:, 1]


    shap_values = get_shap_values(
        model,
        X_failure,
    )


    mean_abs = np.mean(
        np.abs(
            shap_values
        ),
        axis=0,
    )

    mean_signed = np.mean(
        shap_values,
        axis=0,
    )


    part = pd.DataFrame({
        "event":
            event_id,

        "feature":
            CURRENT_FEATURES,

        "mean_abs_shap":
            mean_abs,

        "mean_signed_shap":
            mean_signed,

        "mean_feature_value":
            X_failure.mean(
                axis=0
            ).to_numpy(),
    })


    event_importance_parts.append(
        part
    )


    event_score_summary.append({
        "event":
            event_id,

        "rows_explained":
            len(failure),

        "mean_fault_score":
            float(
                np.mean(scores)
            ),

        "median_fault_score":
            float(
                np.median(scores)
            ),

        "max_fault_score":
            float(
                np.max(scores)
            ),
    })


# ============================================================
# F04 — FROZEN MODEL ONLY
# POST-HOC EXPLANATION
# ============================================================

f04_start = pd.Timestamp(
    event_lookup[
        "F04"
    ]["start_time"]
)

f04_end = pd.Timestamp(
    event_lookup[
        "F04"
    ]["end_time"]
)


f04_failure = master[
    (
        master["timestamp"]
        > f04_start
    )
    &
    (
        master["timestamp"]
        <=
        f04_end
        + pd.Timedelta(
            minutes=1
        )
    )
    &
    (
        master[
            "current_feature_eligible"
        ]
    )
].copy()


f04_failure = (
    f04_failure
    .sort_values("timestamp")
    .reset_index(drop=True)
)


X_f04 = f04_failure[
    CURRENT_FEATURES
]


f04_scores = (
    frozen_model.predict_proba(
        X_f04
    )[:, 1]
)


f04_shap = get_shap_values(
    frozen_model,
    X_f04,
)


f04_failure["score"] = (
    f04_scores
)


f04_failure["alert"] = (
    persistent_alert(
        f04_failure["timestamp"],
        f04_scores,
        threshold,
        persistence,
    )
)


mean_abs = np.mean(
    np.abs(f04_shap),
    axis=0,
)

mean_signed = np.mean(
    f04_shap,
    axis=0,
)


f04_part = pd.DataFrame({
    "event":
        "F04",

    "feature":
        CURRENT_FEATURES,

    "mean_abs_shap":
        mean_abs,

    "mean_signed_shap":
        mean_signed,

    "mean_feature_value":
        X_f04.mean(
            axis=0
        ).to_numpy(),
})


event_importance_parts.append(
    f04_part
)


event_score_summary.append({
    "event":
        "F04",

    "rows_explained":
        len(f04_failure),

    "mean_fault_score":
        float(
            np.mean(f04_scores)
        ),

    "median_fault_score":
        float(
            np.median(f04_scores)
        ),

    "max_fault_score":
        float(
            np.max(f04_scores)
        ),
})


# ============================================================
# SAVE EVENT SHAP TABLES
# ============================================================

event_importance = pd.concat(
    event_importance_parts,
    ignore_index=True,
)


event_importance.to_csv(
    TABLE_DIR
    / "shap_event_feature_importance.csv",
    index=False,
)


score_summary_df = pd.DataFrame(
    event_score_summary
)


score_summary_df.to_csv(
    TABLE_DIR
    / "shap_event_score_summary.csv",
    index=False,
)


# ============================================================
# PRINT TOP FEATURES PER EVENT
# ============================================================

for event_id in [
    "F02",
    "F03",
    "F04",
]:

    top = (
        event_importance[
            event_importance[
                "event"
            ]
            == event_id
        ]
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .head(12)
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        f"{event_id} — TOP SHAP FEATURES"
    )

    print(
        "=" * 100
    )

    print(
        top[
            [
                "feature",
                "mean_abs_shap",
                "mean_signed_shap",
            ]
        ]
        .round(5)
        .to_string(
            index=False
        )
    )


# ============================================================
# EVENT COMPARISON FIGURE
# ============================================================

overall_event_top = (
    event_importance
    .groupby(
        "feature",
        as_index=False,
    )["mean_abs_shap"]
    .mean()
    .sort_values(
        "mean_abs_shap",
        ascending=False,
    )
    .head(12)[
        "feature"
    ]
    .tolist()
)


comparison = (
    event_importance[
        event_importance[
            "feature"
        ].isin(
            overall_event_top
        )
    ]
    .pivot(
        index="feature",
        columns="event",
        values="mean_abs_shap",
    )
    .reindex(
        overall_event_top
    )
)


fig, ax = plt.subplots(
    figsize=(12, 7)
)

comparison.plot.bar(
    ax=ax
)

ax.set_title(
    "AirGuard-LK: Event-Level SHAP Importance"
)

ax.set_xlabel(
    "Feature"
)

ax.set_ylabel(
    "Mean |SHAP| (raw LightGBM output)"
)

ax.tick_params(
    axis="x",
    rotation=55,
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    / "shap_event_comparison.png",
    dpi=200,
)

plt.close(fig)


# ============================================================
# F04 RISK TIMELINE
# ============================================================

first_alert_time = None

if f04_failure["alert"].any():

    first_alert_time = (
        f04_failure.loc[
            f04_failure["alert"],
            "timestamp",
        ]
        .min()
    )


clock_index = pd.date_range(
    start=(
        f04_start
        + pd.Timedelta(
            minutes=1
        )
    ),
    end=(
        f04_end
        + pd.Timedelta(
            minutes=1
        )
    ),
    freq="1min",
)


timeline = (
    f04_failure[
        [
            "timestamp",
            "score",
            "alert",
        ]
    ]
    .set_index(
        "timestamp"
    )
    .reindex(
        clock_index
    )
)


fig, ax = plt.subplots(
    figsize=(12, 5)
)


ax.plot(
    timeline.index,
    timeline["score"],
    linewidth=1.8,
    label="Frozen LightGBM risk score",
)


ax.axhline(
    threshold,
    linestyle="--",
    linewidth=1.5,
    label=(
        f"Frozen threshold "
        f"({threshold:.3f})"
    ),
)


ax.axvline(
    f04_start,
    linestyle=":",
    linewidth=1.2,
    label="F04 documented start",
)


if first_alert_time is not None:

    ax.axvline(
        first_alert_time,
        linestyle="-.",
        linewidth=1.5,
        label=(
            "First persistent alert "
            f"(+{int((first_alert_time - f04_start).total_seconds()/60)} min)"
        ),
    )


ax.set_title(
    "F04 Holdout: AirGuard-LK Risk Development"
)

ax.set_ylabel(
    "Fault score"
)

ax.set_xlabel(
    "Time"
)

ax.set_ylim(
    -0.03,
    1.03,
)

ax.legend(
    loc="best"
)

fig.autofmt_xdate()

fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    / "f04_risk_timeline.png",
    dpi=220,
)

plt.close(fig)


# ============================================================
# F04 SHAP CHECKPOINTS
# ============================================================

checkpoint_offsets = [
    5,
    30,
    60,
    90,
    120,
    140,
    148,
    150,
    170,
]


# Use the most important F04
# features across the whole event.
f04_top_features = (
    f04_part
    .sort_values(
        "mean_abs_shap",
        ascending=False,
    )
    .head(10)[
        "feature"
    ]
    .tolist()
)


feature_indices = [
    CURRENT_FEATURES.index(
        feature
    )
    for feature in f04_top_features
]


checkpoint_rows = []


for offset in checkpoint_offsets:

    target_time = (
        f04_start
        +
        pd.Timedelta(
            minutes=offset
        )
    )


    candidate = f04_failure[
        f04_failure["timestamp"]
        >= target_time
    ]


    if len(candidate) == 0:
        continue


    row_index = candidate.index[0]

    row = f04_failure.loc[
        row_index
    ]


    for feature, feature_index in zip(
        f04_top_features,
        feature_indices,
    ):

        checkpoint_rows.append({
            "requested_offset_minutes":
                offset,

            "timestamp":
                row["timestamp"],

            "actual_offset_minutes":
                (
                    row["timestamp"]
                    -
                    f04_start
                ).total_seconds()
                / 60.0,

            "score":
                float(
                    row["score"]
                ),

            "feature":
                feature,

            "feature_value":
                float(
                    row[
                        feature
                    ]
                ),

            "shap_value":
                float(
                    f04_shap[
                        row_index,
                        feature_index,
                    ]
                ),
        })


checkpoint_df = pd.DataFrame(
    checkpoint_rows
)


checkpoint_df.to_csv(
    TABLE_DIR
    / "f04_shap_checkpoints.csv",
    index=False,
)


# ============================================================
# F04 SHAP CHECKPOINT HEATMAP
# ============================================================

heat = (
    checkpoint_df
    .pivot(
        index="feature",
        columns="actual_offset_minutes",
        values="shap_value",
    )
    .reindex(
        f04_top_features
    )
)


fig, ax = plt.subplots(
    figsize=(11, 7)
)


max_abs = np.nanmax(
    np.abs(
        heat.to_numpy()
    )
)


if not np.isfinite(max_abs) or max_abs == 0:
    max_abs = 1.0


im = ax.imshow(
    heat.to_numpy(),
    aspect="auto",
    cmap="coolwarm",
    vmin=-max_abs,
    vmax=max_abs,
)


ax.set_yticks(
    np.arange(
        len(heat.index)
    )
)

ax.set_yticklabels(
    heat.index
)


ax.set_xticks(
    np.arange(
        len(heat.columns)
    )
)

ax.set_xticklabels([
    f"+{int(x)} min"
    for x in heat.columns
])


ax.set_title(
    "F04: How Feature Contributions Changed as the Fault Developed"
)

ax.set_xlabel(
    "Minutes after documented F04 start"
)

ax.set_ylabel(
    "Feature"
)


cbar = fig.colorbar(
    im,
    ax=ax,
)

cbar.set_label(
    "Signed SHAP value\n(+ increases fault score)"
)


fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    / "f04_shap_checkpoint_heatmap.png",
    dpi=220,
)

plt.close(fig)


# ============================================================
# F04 TOP FEATURE PHYSICAL TRAJECTORIES
# Standardised relative to 6 h pre-event baseline
# ============================================================

pre_start = (
    f04_start
    -
    pd.Timedelta(
        minutes=360
    )
)


pre_event = master[
    (
        master["timestamp"]
        > pre_start
    )
    &
    (
        master["timestamp"]
        <= f04_start
    )
    &
    (
        master[
            "current_feature_eligible"
        ]
    )
].copy()


top_physical_features = (
    f04_part
    .sort_values(
        "mean_abs_shap",
        ascending=False,
    )
    .head(5)[
        "feature"
    ]
    .tolist()
)


baseline_mean = (
    pre_event[
        top_physical_features
    ]
    .mean()
)


baseline_std = (
    pre_event[
        top_physical_features
    ]
    .std()
)


baseline_std = baseline_std.mask(
    baseline_std < 1e-8,
    1.0,
)


trajectory = f04_failure[
    [
        "timestamp",
        *top_physical_features,
    ]
].copy()


for feature in top_physical_features:

    trajectory[
        f"{feature}_z"
    ] = (
        (
            trajectory[
                feature
            ]
            -
            baseline_mean[
                feature
            ]
        )
        /
        baseline_std[
            feature
        ]
    )


trajectory.to_csv(
    TABLE_DIR
    / "f04_top_feature_trajectories.csv",
    index=False,
)


fig, ax = plt.subplots(
    figsize=(12, 6)
)


for feature in top_physical_features:

    ax.plot(
        trajectory[
            "timestamp"
        ],
        trajectory[
            f"{feature}_z"
        ],
        label=feature,
        linewidth=1.4,
    )


if first_alert_time is not None:

    ax.axvline(
        first_alert_time,
        linestyle="--",
        linewidth=1.4,
        label="First persistent alert",
    )


ax.axhline(
    0,
    linewidth=0.8,
)


ax.set_title(
    "F04: Strongest Model Features Relative to Pre-Event Baseline"
)

ax.set_ylabel(
    "Standard deviations from 6 h pre-event baseline"
)

ax.set_xlabel(
    "Time"
)

ax.legend(
    loc="best"
)

fig.autofmt_xdate()

fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    / "f04_top_feature_trajectories.png",
    dpi=220,
)

plt.close(fig)


# ============================================================
# FINAL MODEL HOLDOUT SHAP IMPORTANCE
#
# Post-hoc explanation sample:
# current-eligible F04 normal history + F04 fault
# ============================================================

f03_end = pd.Timestamp(
    event_lookup[
        "F03"
    ]["end_time"]
)


evaluation_start = (
    f03_end
    +
    pd.Timedelta(
        minutes=360
    )
)


f04_pre_start = (
    f04_start
    -
    pd.Timedelta(
        minutes=360
    )
)


holdout_normal = master[
    (
        master["timestamp"]
        >= evaluation_start
    )
    &
    (
        master["timestamp"]
        <= f04_pre_start
    )
    &
    (
        master[
            "current_feature_eligible"
        ]
    )
].copy()


if len(holdout_normal) > 4000:

    holdout_normal = (
        holdout_normal
        .sample(
            4000,
            random_state=SEED,
        )
        .sort_values("timestamp")
    )


holdout_explain = pd.concat([
    holdout_normal,
    f04_failure,
])


X_holdout_explain = (
    holdout_explain[
        CURRENT_FEATURES
    ]
)


holdout_shap = get_shap_values(
    frozen_model,
    X_holdout_explain,
)


holdout_importance = pd.DataFrame({
    "feature":
        CURRENT_FEATURES,

    "mean_abs_shap":
        np.mean(
            np.abs(
                holdout_shap
            ),
            axis=0,
        ),
})


holdout_importance = (
    holdout_importance
    .sort_values(
        "mean_abs_shap",
        ascending=False,
    )
)


holdout_importance.to_csv(
    TABLE_DIR
    / "final_model_shap_holdout_importance.csv",
    index=False,
)


top15 = (
    holdout_importance
    .head(15)
    .sort_values(
        "mean_abs_shap",
        ascending=True,
    )
)


fig, ax = plt.subplots(
    figsize=(9, 7)
)


ax.barh(
    top15[
        "feature"
    ],
    top15[
        "mean_abs_shap"
    ],
)


ax.set_title(
    "Frozen LightGBM: SHAP Importance on F04 Evaluation Data"
)

ax.set_xlabel(
    "Mean |SHAP|"
)


fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    / "final_model_shap_holdout_bar.png",
    dpi=220,
)

plt.close(fig)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "analysis":
        "post_hoc_shap_explainability",

    "model_changed":
        False,

    "threshold_changed":
        False,

    "frozen_threshold":
        threshold,

    "persistence_minutes":
        persistence,

    "f04_rows_explained":
        int(
            len(f04_failure)
        ),

    "f04_first_persistent_alert":
        (
            str(first_alert_time)
            if first_alert_time is not None
            else None
        ),

    "f04_detection_delay_minutes":
        (
            float(
                (
                    first_alert_time
                    -
                    f04_start
                ).total_seconds()
                / 60.0
            )
            if first_alert_time is not None
            else None
        ),

    "f04_top_features":
        f04_top_features,

    "note":
        (
            "SHAP values are model explanations, "
            "not evidence of physical causation."
        ),
}


with open(
    METRIC_DIR
    / "shap_explainability_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print(
    "\n"
    + "=" * 100
)

print(
    "F04 SHAP CHECKPOINTS"
)

print(
    "=" * 100
)


print(
    checkpoint_df[
        [
            "actual_offset_minutes",
            "score",
            "feature",
            "feature_value",
            "shap_value",
        ]
    ]
    .round(5)
    .to_string(
        index=False
    )
)


print(
    "\n"
    + "=" * 100
)

print(
    "FINAL HOLDOUT SHAP TOP 15"
)

print(
    "=" * 100
)


print(
    holdout_importance
    .head(15)
    .round(5)
    .to_string(
        index=False
    )
)


print(
    "\nSaved figures:"
)

for filename in [
    "shap_event_comparison.png",
    "f04_risk_timeline.png",
    "f04_shap_checkpoint_heatmap.png",
    "f04_top_feature_trajectories.png",
    "final_model_shap_holdout_bar.png",
]:
    print(
        FIGURE_DIR
        / filename
    )


print(
    "\nSHAP EXPLAINABILITY ANALYSIS COMPLETE"
)
