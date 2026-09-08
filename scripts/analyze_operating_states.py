from pathlib import Path

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
TABLE_DIR.mkdir(parents=True, exist_ok=True)

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

ANALOG = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
]

print("=" * 78)
print("AIRGUARD-LK — COMPRESSOR OPERATING-STATE AUDIT")
print("=" * 78)

usecols = [
    "timestamp",
    *DIGITAL,
    *ANALOG,
]

print("\nLoading dataset...")

df = pd.read_csv(
    CSV_PATH,
    usecols=usecols
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = (
    df.sort_values("timestamp")
      .reset_index(drop=True)
)

print(f"Rows: {len(df):,}")


# ============================================================
# 1. DIGITAL SIGNAL RELATIONSHIPS
# ============================================================

relationships = []

pairs = [
    ("COMP", "MPG"),
    ("COMP", "DV_eletric"),
    ("MPG", "DV_eletric"),
    ("Towers", "Pressure_switch"),
    ("Oil_level", "Caudal_impulses"),
]

for a, b in pairs:

    equal_fraction = float(
        (df[a] == df[b]).mean()
    )

    opposite_fraction = float(
        (df[a] == (1 - df[b])).mean()
    )

    correlation = float(
        df[[a, b]]
        .corr()
        .iloc[0, 1]
    )

    relationships.append({
        "signal_a": a,
        "signal_b": b,
        "equal_fraction": equal_fraction,
        "opposite_fraction": opposite_fraction,
        "correlation": correlation,
    })

relationship_df = pd.DataFrame(
    relationships
)

relationship_df.to_csv(
    TABLE_DIR
    / "metropt3_digital_relationships.csv",
    index=False
)


# ============================================================
# 2. COMMON DIGITAL STATE COMBINATIONS
# ============================================================

state_counts = (
    df[DIGITAL]
    .value_counts()
    .reset_index(name="count")
)

state_counts["fraction"] = (
    state_counts["count"]
    / len(df)
)

state_counts.to_csv(
    TABLE_DIR
    / "metropt3_operating_state_combinations.csv",
    index=False
)


# ============================================================
# 3. DIGITAL TRANSITION BEHAVIOUR
# ============================================================

transition_rows = []

total_duration_days = (
    df["timestamp"].max()
    -
    df["timestamp"].min()
).total_seconds() / 86400

for sensor in DIGITAL:

    changed = (
        df[sensor]
        .ne(df[sensor].shift())
    )

    transition_count = int(
        changed.sum() - 1
    )

    transition_rows.append({
        "sensor": sensor,
        "transition_count": transition_count,
        "transitions_per_calendar_day":
            transition_count
            / total_duration_days,
        "active_fraction": float(
            df[sensor].mean()
        ),
    })

transition_df = pd.DataFrame(
    transition_rows
)

transition_df.to_csv(
    TABLE_DIR
    / "metropt3_digital_transition_summary.csv",
    index=False
)


# ============================================================
# 4. GAP-AWARE RUN LENGTHS
# ============================================================

run_rows = []

time_gap = (
    df["timestamp"]
    .diff()
    .dt.total_seconds()
)

for sensor in DIGITAL:

    boundary = (
        df[sensor].ne(
            df[sensor].shift()
        )
        |
        (time_gap > 30)
        |
        time_gap.isna()
    )

    run_id = boundary.cumsum()

    temp = pd.DataFrame({
        "timestamp": df["timestamp"],
        "value": df[sensor],
        "run_id": run_id,
    })

    runs = (
        temp.groupby(
            ["run_id", "value"],
            as_index=False
        )
        .agg(
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            samples=("timestamp", "size"),
        )
    )

    runs["duration_seconds"] = (
        (
            runs["end"]
            -
            runs["start"]
        )
        .dt.total_seconds()
    )

    # Add nominal one sample interval so a
    # single-sample run is not treated as 0 s.
    runs["duration_seconds"] += 10

    for value in [0.0, 1.0]:

        subset = runs[
            runs["value"] == value
        ]

        if len(subset) == 0:
            continue

        run_rows.append({
            "sensor": sensor,
            "state": int(value),
            "runs": int(len(subset)),
            "median_duration_s": float(
                subset[
                    "duration_seconds"
                ].median()
            ),
            "mean_duration_s": float(
                subset[
                    "duration_seconds"
                ].mean()
            ),
            "p95_duration_s": float(
                subset[
                    "duration_seconds"
                ].quantile(0.95)
            ),
            "max_duration_s": float(
                subset[
                    "duration_seconds"
                ].max()
            ),
        })

run_df = pd.DataFrame(
    run_rows
)

run_df.to_csv(
    TABLE_DIR
    / "metropt3_digital_run_lengths.csv",
    index=False
)


# ============================================================
# 5. ANALOG BEHAVIOUR BY MAIN COMPRESSOR STATE
# ============================================================

state_sensor_rows = []

for comp_state in [0.0, 1.0]:

    subset = df[
        df["COMP"] == comp_state
    ]

    for sensor in ANALOG:

        state_sensor_rows.append({
            "COMP_state": int(comp_state),
            "sensor": sensor,
            "samples": len(subset),
            "mean": float(
                subset[sensor].mean()
            ),
            "median": float(
                subset[sensor].median()
            ),
            "std": float(
                subset[sensor].std()
            ),
            "p05": float(
                subset[sensor].quantile(0.05)
            ),
            "p95": float(
                subset[sensor].quantile(0.95)
            ),
        })

state_sensor_df = pd.DataFrame(
    state_sensor_rows
)

state_sensor_df.to_csv(
    TABLE_DIR
    / "metropt3_analog_by_comp_state.csv",
    index=False
)


# ============================================================
# 6. ENGINEERING RELATIONSHIPS
# ============================================================

engineering = {
    "tp3_reservoir_correlation":
        float(
            df[
                ["TP3", "Reservoirs"]
            ]
            .corr()
            .iloc[0, 1]
        ),

    "tp3_reservoir_mean_abs_difference_bar":
        float(
            (
                df["TP3"]
                -
                df["Reservoirs"]
            )
            .abs()
            .mean()
        ),

    "tp3_reservoir_p95_abs_difference_bar":
        float(
            (
                df["TP3"]
                -
                df["Reservoirs"]
            )
            .abs()
            .quantile(0.95)
        ),
}

engineering_df = pd.DataFrame(
    [
        {
            "metric": key,
            "value": value,
        }
        for key, value
        in engineering.items()
    ]
)

engineering_df.to_csv(
    TABLE_DIR
    / "metropt3_engineering_relationships.csv",
    index=False
)


# ============================================================
# PRINT IMPORTANT RESULTS
# ============================================================

print("\n" + "=" * 78)
print("DIGITAL SIGNAL RELATIONSHIPS")
print("=" * 78)

print(
    relationship_df
    .round(5)
    .to_string(index=False)
)

print("\n" + "=" * 78)
print("TOP 15 OPERATING STATES")
print("=" * 78)

print(
    state_counts
    .head(15)
    .to_string(index=False)
)

print("\n" + "=" * 78)
print("DIGITAL TRANSITION SUMMARY")
print("=" * 78)

print(
    transition_df
    .round(3)
    .to_string(index=False)
)

print("\n" + "=" * 78)
print("ENGINEERING RELATIONSHIPS")
print("=" * 78)

print(
    engineering_df
    .round(6)
    .to_string(index=False)
)

print("\nOPERATING-STATE AUDIT COMPLETE")
