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

TABLE_DIR = ROOT / "artifacts" / "tables"
METRIC_DIR = ROOT / "artifacts" / "metrics"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("AIRGUARD-LK — METROPT-3 DATASET AUDIT")
print("=" * 70)

print("\nLoading dataset...")
df = pd.read_csv(CSV_PATH)

print(f"Original shape: {df.shape}")

# --------------------------------------------------
# Remove exported CSV index
# --------------------------------------------------

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print(f"Shape after removing index column: {df.shape}")

# --------------------------------------------------
# Parse timestamps
# --------------------------------------------------

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

invalid_timestamps = int(df["timestamp"].isna().sum())

df = (
    df.sort_values("timestamp")
      .reset_index(drop=True)
)

# --------------------------------------------------
# Basic audit
# --------------------------------------------------

n_rows = len(df)

duplicate_rows = int(
    df.duplicated().sum()
)

duplicate_timestamps = int(
    df["timestamp"].duplicated().sum()
)

total_missing = int(
    df.isna().sum().sum()
)

start_time = df["timestamp"].min()
end_time = df["timestamp"].max()

# --------------------------------------------------
# Sampling interval
# --------------------------------------------------

delta = (
    df["timestamp"]
    .diff()
    .dt.total_seconds()
)

positive_delta = delta[
    delta > 0
].dropna()

cadence = {
    "median_seconds": float(positive_delta.median()),
    "mean_seconds": float(positive_delta.mean()),
    "min_seconds": float(positive_delta.min()),
    "p01_seconds": float(positive_delta.quantile(0.01)),
    "p05_seconds": float(positive_delta.quantile(0.05)),
    "p95_seconds": float(positive_delta.quantile(0.95)),
    "p99_seconds": float(positive_delta.quantile(0.99)),
    "max_seconds": float(positive_delta.max()),
}

# --------------------------------------------------
# Missing values
# --------------------------------------------------

missing_table = pd.DataFrame({
    "column": df.columns,
    "missing_count": [
        int(df[c].isna().sum())
        for c in df.columns
    ],
})

missing_table["missing_pct"] = (
    missing_table["missing_count"]
    / n_rows
    * 100
)

missing_table.to_csv(
    TABLE_DIR / "metropt3_missing_values.csv",
    index=False
)

# --------------------------------------------------
# Column summary
# --------------------------------------------------

column_summary = []

for col in df.columns:

    column_summary.append({
        "column": col,
        "dtype": str(df[col].dtype),
        "unique_values": int(
            df[col].nunique(dropna=True)
        ),
        "missing_values": int(
            df[col].isna().sum()
        ),
    })

column_summary = pd.DataFrame(
    column_summary
)

column_summary.to_csv(
    TABLE_DIR / "metropt3_column_summary.csv",
    index=False
)

# --------------------------------------------------
# Sensor statistics
# --------------------------------------------------

numeric_cols = (
    df.select_dtypes(include=np.number)
    .columns
    .tolist()
)

sensor_summary = (
    df[numeric_cols]
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )
    .T
)

sensor_summary["unique_values"] = (
    df[numeric_cols]
    .nunique()
)

sensor_summary["missing_values"] = (
    df[numeric_cols]
    .isna()
    .sum()
)

sensor_summary.to_csv(
    TABLE_DIR / "metropt3_sensor_summary.csv"
)

# --------------------------------------------------
# Low-cardinality / digital signals
# --------------------------------------------------

low_cardinality = []

for col in numeric_cols:

    n_unique = int(df[col].nunique())

    if n_unique <= 20:
        values = sorted(
            df[col]
            .dropna()
            .unique()
            .tolist()
        )

        low_cardinality.append({
            "column": col,
            "unique_count": n_unique,
            "values": values,
        })

pd.DataFrame(
    low_cardinality
).to_csv(
    TABLE_DIR / "metropt3_low_cardinality_signals.csv",
    index=False
)

# --------------------------------------------------
# Timestamp gaps
# --------------------------------------------------

gap_thresholds = [
    10,
    20,
    30,
    60,
    120,
    300,
    600,
    3600,
]

gap_counts = {}

for seconds in gap_thresholds:

    gap_counts[
        f"gaps_over_{seconds}s"
    ] = int(
        (positive_delta > seconds).sum()
    )

largest_gaps = (
    pd.DataFrame({
        "timestamp": df["timestamp"],
        "delta_seconds": delta,
    })
    .sort_values(
        "delta_seconds",
        ascending=False
    )
    .head(30)
)

largest_gaps.to_csv(
    TABLE_DIR / "metropt3_largest_time_gaps.csv",
    index=False
)

# --------------------------------------------------
# Sampling interval distribution
# --------------------------------------------------

delta_distribution = (
    positive_delta
    .value_counts()
    .sort_index()
    .rename_axis("delta_seconds")
    .reset_index(name="count")
)

delta_distribution.to_csv(
    TABLE_DIR / "metropt3_sampling_distribution.csv",
    index=False
)

# --------------------------------------------------
# Final JSON
# --------------------------------------------------

audit = {
    "dataset": "MetroPT-3",
    "rows": int(n_rows),
    "columns_after_index_removal": int(df.shape[1]),

    "timestamp_start": str(start_time),
    "timestamp_end": str(end_time),

    "invalid_timestamps": invalid_timestamps,
    "duplicate_rows": duplicate_rows,
    "duplicate_timestamps": duplicate_timestamps,

    "total_missing_values": total_missing,

    "numeric_signal_count": len(numeric_cols),

    "sampling": cadence,
    "timestamp_gap_counts": gap_counts,
}

with open(
    METRIC_DIR / "metropt3_dataset_audit.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        audit,
        f,
        indent=2
    )

# --------------------------------------------------
# Console output
# --------------------------------------------------

print("\n" + "=" * 70)
print("AUDIT SUMMARY")
print("=" * 70)

print(f"\nRows: {n_rows:,}")
print(f"Columns: {df.shape[1]}")

print("\nTime range:")
print("Start:", start_time)
print("End:  ", end_time)

print("\nMissing values:")
print(f"{total_missing:,}")

print("\nInvalid timestamps:")
print(f"{invalid_timestamps:,}")

print("\nDuplicate rows:")
print(f"{duplicate_rows:,}")

print("\nDuplicate timestamps:")
print(f"{duplicate_timestamps:,}")

print("\nSampling interval:")
for key, value in cadence.items():
    print(
        f"{key:20s}: "
        f"{value:.3f}"
    )

print("\nTime gaps:")
for key, value in gap_counts.items():
    print(
        f"{key:20s}: "
        f"{value:,}"
    )

print("\nCandidate digital / low-cardinality signals:")

for item in low_cardinality:
    print(
        f"{item['column']:20s} "
        f"unique={item['unique_count']:3d} "
        f"values={item['values']}"
    )

print("\nSaved audit artifacts to:")
print(TABLE_DIR)
print(METRIC_DIR)

print("\nAUDIT COMPLETE")
