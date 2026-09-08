from pathlib import Path

import json
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    ROOT
    / "data"
    / "raw"
    / "metropt3"
    / "MetroPT3(AirCompressor).csv"
)

OUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "metropt3_1min.parquet"
)

METRIC_PATH = (
    ROOT
    / "artifacts"
    / "metrics"
    / "metropt3_1min_build.json"
)

TABLE_PATH = (
    ROOT
    / "artifacts"
    / "tables"
    / "metropt3_1min_quality.csv"
)

OUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

METRIC_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

TABLE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


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

USECOLS = [
    "timestamp",
    *ANALOG,
    *DIGITAL,
]


print("=" * 78)
print("AIRGUARD-LK — BUILD GAP-SAFE 1-MINUTE TABLE")
print("=" * 78)

print("\nLoading raw dataset...")

df = pd.read_csv(
    CSV_PATH,
    usecols=USECOLS
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = (
    df.sort_values("timestamp")
      .reset_index(drop=True)
)

print(f"Raw rows: {len(df):,}")


# ============================================================
# RAW TIME-GAP INFORMATION
# ============================================================

df["gap_from_previous_s"] = (
    df["timestamp"]
    .diff()
    .dt.total_seconds()
)

# Wall-clock minute
df["minute"] = (
    df["timestamp"]
    .dt.floor("1min")
)


# ============================================================
# DIGITAL TRANSITIONS AT RAW RESOLUTION
# ============================================================

for sensor in DIGITAL:

    changed = (
        df[sensor]
        .ne(df[sensor].shift())
    )

    # First sample is not a real transition
    changed.iloc[0] = False

    # Do not count a transition across a large acquisition gap
    changed = (
        changed
        &
        (
            df["gap_from_previous_s"]
            .fillna(0)
            <= 30
        )
    )

    df[
        f"{sensor}_transition"
    ] = changed.astype(np.int8)


# ============================================================
# ENGINEERING CONSISTENCY FEATURES
# ============================================================

df["COMP_MPG_mismatch"] = (
    df["COMP"] != df["MPG"]
).astype(np.int8)

# Normal relationship is usually opposite.
# Equality therefore represents an inconsistency.
df["COMP_DV_equal"] = (
    df["COMP"] == df["DV_eletric"]
).astype(np.int8)

df["TP3_Reservoir_abs_diff"] = (
    df["TP3"]
    -
    df["Reservoirs"]
).abs()


# ============================================================
# AGGREGATION DICTIONARY
# ============================================================

agg = {}

# Analog signals:
# retain central level, variation, extremes, and latest value.
for sensor in ANALOG:

    agg[sensor] = [
        "mean",
        "std",
        "min",
        "max",
        "last",
    ]

# Digital signals:
# mean = fraction of minute active
# last = state at end of minute
for sensor in DIGITAL:

    agg[sensor] = [
        "mean",
        "last",
    ]

for sensor in DIGITAL:

    agg[
        f"{sensor}_transition"
    ] = "sum"

agg["COMP_MPG_mismatch"] = "mean"
agg["COMP_DV_equal"] = "mean"

agg["TP3_Reservoir_abs_diff"] = [
    "mean",
    "max",
]

agg["gap_from_previous_s"] = "max"


print("\nAggregating to one-minute resolution...")

minute = (
    df
    .groupby(
        "minute",
        sort=True
    )
    .agg(agg)
)


# ============================================================
# FLATTEN COLUMN NAMES
# ============================================================

minute.columns = [
    "_".join(
        [
            str(part)
            for part in col
            if str(part)
        ]
    )
    if isinstance(col, tuple)
    else str(col)
    for col in minute.columns
]

minute = minute.reset_index()

minute = minute.rename(
    columns={
        "minute": "timestamp"
    }
)


# ============================================================
# SAMPLE COUNTS
# ============================================================

sample_count = (
    df.groupby("minute")
      .size()
      .rename("sample_count")
      .reset_index()
      .rename(
          columns={"minute": "timestamp"}
      )
)

minute = minute.merge(
    sample_count,
    on="timestamp",
    how="left"
)


# ============================================================
# CLEAN STANDARD-DEVIATION NaNs
#
# A minute with one sample has undefined std.
# Set to zero because no within-minute variation was observed.
# ============================================================

std_cols = [
    c
    for c in minute.columns
    if c.endswith("_std")
]

minute[std_cols] = (
    minute[std_cols]
    .fillna(0.0)
)


# ============================================================
# MINUTE-LEVEL GAP / SEGMENT ANALYSIS
# ============================================================

minute["minute_delta_s"] = (
    minute["timestamp"]
    .diff()
    .dt.total_seconds()
)

# New segment if:
# 1. no preceding minute,
# 2. wall-clock minute discontinuity,
# 3. incoming raw-data gap exceeded 30 seconds.
segment_break = (
    minute["minute_delta_s"].isna()
    |
    (minute["minute_delta_s"] > 60)
    |
    (
        minute["gap_from_previous_s_max"]
        > 30
    )
)

minute["segment_id"] = (
    segment_break
    .cumsum()
    .astype(np.int32)
)


# ============================================================
# QUALITY FEATURES
# ============================================================

# Nominal expectation is ~6 samples per minute.
# This is descriptive only, not a hard validity rule.

minute["sampling_density"] = (
    minute["sample_count"]
    / 6.0
)

minute["low_sample_minute"] = (
    minute["sample_count"] < 3
).astype(np.int8)

minute["raw_gap_over_30s"] = (
    minute["gap_from_previous_s_max"]
    > 30
).astype(np.int8)


# ============================================================
# BASIC QUALITY TABLE
# ============================================================

quality = pd.DataFrame({
    "metric": [
        "raw_rows",
        "minute_rows",
        "segments",
        "low_sample_minutes",
        "minutes_with_raw_gap_over_30s",
        "median_samples_per_minute",
        "mean_samples_per_minute",
        "min_samples_per_minute",
        "max_samples_per_minute",
    ],

    "value": [
        len(df),
        len(minute),
        minute["segment_id"].nunique(),
        int(
            minute[
                "low_sample_minute"
            ].sum()
        ),
        int(
            minute[
                "raw_gap_over_30s"
            ].sum()
        ),
        float(
            minute[
                "sample_count"
            ].median()
        ),
        float(
            minute[
                "sample_count"
            ].mean()
        ),
        int(
            minute[
                "sample_count"
            ].min()
        ),
        int(
            minute[
                "sample_count"
            ].max()
        ),
    ]
})

quality.to_csv(
    TABLE_PATH,
    index=False
)


# ============================================================
# SAVE TABLE
# ============================================================

minute.to_parquet(
    OUT_PATH,
    index=False
)


# ============================================================
# BUILD SUMMARY
# ============================================================

summary = {
    "raw_rows": int(len(df)),
    "minute_rows": int(len(minute)),
    "columns": int(minute.shape[1]),

    "timestamp_start":
        str(minute["timestamp"].min()),

    "timestamp_end":
        str(minute["timestamp"].max()),

    "segments":
        int(
            minute[
                "segment_id"
            ].nunique()
        ),

    "low_sample_minutes":
        int(
            minute[
                "low_sample_minute"
            ].sum()
        ),

    "minutes_with_raw_gap_over_30s":
        int(
            minute[
                "raw_gap_over_30s"
            ].sum()
        ),

    "median_samples_per_minute":
        float(
            minute[
                "sample_count"
            ].median()
        ),
}

with open(
    METRIC_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# CONSOLE
# ============================================================

print("\n" + "=" * 78)
print("1-MINUTE TABLE SUMMARY")
print("=" * 78)

print(
    quality.to_string(
        index=False
    )
)

print("\nShape:")
print(minute.shape)

print("\nFirst 20 columns:")
for column in minute.columns[:20]:
    print(" -", column)

print("\nSaved:")
print(OUT_PATH)

print("\nBUILD COMPLETE")
