from pathlib import Path

import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "metropt3_features_1min.parquet"
)

QUALITY_PATH = (
    ROOT
    / "data"
    / "processed"
    / "metropt3_quality_1min.parquet"
)

LABEL_PATH = (
    ROOT
    / "data"
    / "interim"
    / "metropt3_1min_labelled.parquet"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "metropt3_model_master.parquet"
)

METRIC_PATH = (
    ROOT
    / "artifacts"
    / "metrics"
    / "metropt3_model_master.json"
)


print("=" * 80)
print("AIRGUARD-LK — BUILD MODEL MASTER TABLE")
print("=" * 80)


# ============================================================
# LOAD
# ============================================================

features = pd.read_parquet(
    FEATURE_PATH
)

quality = pd.read_parquet(
    QUALITY_PATH
)

labels_full = pd.read_parquet(
    LABEL_PATH
)


# ============================================================
# LABEL / METADATA COLUMNS ONLY
# ============================================================

label_columns = [
    "timestamp",

    "failure_active",
    "failure_event_id",

    "pre_15m",
    "pre_30m",
    "pre_60m",
    "pre_120m",
    "pre_360m",

    "post_60m",
    "post_120m",
    "post_360m",

    "next_event_id",
    "previous_event_id",

    "minutes_to_failure",
    "minutes_since_failure_end",

    "event_phase",
]

labels = labels_full[
    label_columns
].copy()


# ============================================================
# ALIGNMENT CHECKS
# ============================================================

assert features["timestamp"].is_unique
assert quality["timestamp"].is_unique
assert labels["timestamp"].is_unique

assert len(features) == len(quality)
assert len(features) == len(labels)

assert features[
    "timestamp"
].equals(
    quality["timestamp"]
)

assert features[
    "timestamp"
].equals(
    labels["timestamp"]
)


# ============================================================
# PREFIX QUALITY COLUMNS
# ============================================================

quality = quality.rename(
    columns={
        c: f"quality__{c}"
        for c in quality.columns
        if c != "timestamp"
    }
)


# ============================================================
# PREFIX LABEL/METADATA COLUMNS
#
# This gives us a mechanical leakage guard:
#
# feature columns  = no special prefix
# target metadata  = target__
# quality metadata = quality__
# ============================================================

labels = labels.rename(
    columns={
        c: f"target__{c}"
        for c in labels.columns
        if c != "timestamp"
    }
)


# ============================================================
# MERGE
# ============================================================

master = (
    features
    .merge(
        quality,
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )
    .merge(
        labels,
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )
)


# ============================================================
# FEATURE COLUMN DEFINITION
# ============================================================

feature_columns = [
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

quality_columns = [
    c
    for c in master.columns
    if c.startswith(
        "quality__"
    )
]

target_columns = [
    c
    for c in master.columns
    if c.startswith(
        "target__"
    )
]


# ============================================================
# ELIGIBILITY
# ============================================================

master[
    "quality__model_feature_eligible"
] = (
    (
        master[
            "quality__history_120m_complete"
        ] == 1
    )
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
).astype("int8")


# ============================================================
# LEAKAGE SAFETY CHECK
# ============================================================

for forbidden in [
    "minutes_to_failure",
    "next_event",
    "previous_event",
    "event_phase",
    "pre_",
    "post_",
    "failure_active",
]:

    matches = [
        c
        for c in feature_columns
        if forbidden.lower()
        in c.lower()
    ]

    if matches:
        raise RuntimeError(
            "Potential target leakage "
            f"found in features: {matches}"
        )


# ============================================================
# SAVE
# ============================================================

master.to_parquet(
    OUTPUT_PATH,
    index=False
)


summary = {
    "rows":
        int(len(master)),

    "model_feature_count":
        int(len(feature_columns)),

    "quality_metadata_count":
        int(len(quality_columns) + 1),

    "target_metadata_count":
        int(len(target_columns)),

    "eligible_rows":
        int(
            master[
                "quality__model_feature_eligible"
            ].sum()
        ),

    "target_columns_in_feature_matrix":
        False,
}


METRIC_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    METRIC_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


print("\nMODEL MASTER SUMMARY")
print("-" * 80)

for key, value in summary.items():
    print(
        f"{key}: {value}"
    )

print("\nExample feature columns:")
for column in feature_columns[:20]:
    print(" -", column)

print("\nExample target columns:")
for column in target_columns[:10]:
    print(" -", column)

print("\nSaved:")
print(OUTPUT_PATH)

print("\nMODEL MASTER BUILD COMPLETE")
