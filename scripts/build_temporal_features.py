from pathlib import Path
import json
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "metropt3_1min.parquet"
)

OUTPUT_PATH = (
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

METRIC_PATH = (
    ROOT
    / "artifacts"
    / "metrics"
    / "metropt3_feature_build.json"
)

TABLE_PATH = (
    ROOT
    / "artifacts"
    / "tables"
    / "metropt3_feature_manifest.csv"
)

OUTPUT_PATH.parent.mkdir(
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

WINDOWS = [
    5,
    15,
    30,
    60,
    120,
]

TREND_WINDOWS = [
    15,
    60,
]

VOLATILITY_WINDOWS = [
    15,
    60,
]


print("=" * 80)
print("AIRGUARD-LK — BUILD PAST-ONLY TEMPORAL FEATURES")
print("=" * 80)

print("\nLoading one-minute base table...")

df = pd.read_parquet(
    INPUT_PATH
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = (
    df.sort_values(
        ["segment_id", "timestamp"]
    )
    .reset_index(drop=True)
)

print(f"Rows: {len(df):,}")
print(f"Columns: {df.shape[1]}")


# ============================================================
# CONTINUOUS HISTORY POSITION
# ============================================================

df["segment_position"] = (
    df.groupby("segment_id")
      .cumcount()
      + 1
)

# A 120-minute rolling window including current minute
# requires 120 consecutive minute rows.
df["history_120m_complete"] = (
    df["segment_position"] >= 120
).astype(np.int8)


# ============================================================
# BASE FEATURE TABLE
#
# Only physical/system measurements are included.
# Data-quality fields are kept separately.
# ============================================================

feature_df = pd.DataFrame({
    "timestamp": df["timestamp"],
})


# ------------------------------------------------------------
# Current-minute analogue features
# ------------------------------------------------------------

for sensor in ANALOG:

    for statistic in [
        "mean",
        "std",
        "min",
        "max",
        "last",
    ]:

        column = (
            f"{sensor}_{statistic}"
        )

        feature_df[column] = df[column]


# ------------------------------------------------------------
# Current-minute digital features
# ------------------------------------------------------------

for sensor in DIGITAL:

    for statistic in [
        "mean",
        "last",
    ]:

        column = (
            f"{sensor}_{statistic}"
        )

        feature_df[column] = df[column]

    transition_column = (
        f"{sensor}_transition_sum"
    )

    feature_df[
        transition_column
    ] = df[
        transition_column
    ]


# ------------------------------------------------------------
# Current-minute engineering consistency features
# ------------------------------------------------------------

ENGINEERING_BASE = [
    "COMP_MPG_mismatch_mean",
    "COMP_DV_equal_mean",
    "TP3_Reservoir_abs_diff_mean",
    "TP3_Reservoir_abs_diff_max",
]

for column in ENGINEERING_BASE:

    feature_df[column] = df[column]


# ============================================================
# DERIVED CURRENT-MINUTE ENGINEERING FEATURES
# ============================================================

feature_df[
    "TP3_minus_Reservoirs_mean"
] = (
    df["TP3_mean"]
    -
    df["Reservoirs_mean"]
)

feature_df[
    "TP2_minus_TP3_mean"
] = (
    df["TP2_mean"]
    -
    df["TP3_mean"]
)

feature_df[
    "H1_minus_TP3_mean"
] = (
    df["H1_mean"]
    -
    df["TP3_mean"]
)

feature_df[
    "TP2_minus_H1_mean"
] = (
    df["TP2_mean"]
    -
    df["H1_mean"]
)


# ============================================================
# HELPER FOR SEGMENT-SAFE ROLLING
# ============================================================

def rolling_transform(
    values,
    segment_ids,
    window,
    operation,
):
    temp = pd.DataFrame({
        "segment_id": segment_ids,
        "value": values,
    })

    grouped = temp.groupby(
        "segment_id",
        sort=False
    )["value"]

    if operation == "mean":

        result = grouped.transform(
            lambda s:
            s.rolling(
                window=window,
                min_periods=window
            ).mean()
        )

    elif operation == "std":

        result = grouped.transform(
            lambda s:
            s.rolling(
                window=window,
                min_periods=window
            ).std()
        )

    elif operation == "min":

        result = grouped.transform(
            lambda s:
            s.rolling(
                window=window,
                min_periods=window
            ).min()
        )

    elif operation == "max":

        result = grouped.transform(
            lambda s:
            s.rolling(
                window=window,
                min_periods=window
            ).max()
        )

    elif operation == "sum":

        result = grouped.transform(
            lambda s:
            s.rolling(
                window=window,
                min_periods=window
            ).sum()
        )

    else:
        raise ValueError(
            f"Unknown operation: {operation}"
        )

    return result


# ============================================================
# ANALOG TEMPORAL FEATURES
# ============================================================

print("\nBuilding analogue temporal features...")

for sensor in ANALOG:

    source = df[
        f"{sensor}_mean"
    ]

    # --------------------------------------------------------
    # Rolling level + variability
    # --------------------------------------------------------

    for window in WINDOWS:

        feature_df[
            f"{sensor}_mean_{window}m"
        ] = rolling_transform(
            source,
            df["segment_id"],
            window,
            "mean",
        )

        feature_df[
            f"{sensor}_std_{window}m"
        ] = rolling_transform(
            source,
            df["segment_id"],
            window,
            "std",
        )

    # --------------------------------------------------------
    # Trend / change relative to earlier observation
    # --------------------------------------------------------

    grouped = source.groupby(
        df["segment_id"],
        sort=False
    )

    for window in TREND_WINDOWS:

        previous = grouped.shift(
            window
        )

        feature_df[
            f"{sensor}_delta_{window}m"
        ] = (
            source
            -
            previous
        )

    # --------------------------------------------------------
    # Recent within-minute volatility
    # --------------------------------------------------------

    minute_std = df[
        f"{sensor}_std"
    ]

    for window in VOLATILITY_WINDOWS:

        feature_df[
            f"{sensor}_minute_volatility_{window}m"
        ] = rolling_transform(
            minute_std,
            df["segment_id"],
            window,
            "mean",
        )


# ============================================================
# DIGITAL TEMPORAL FEATURES
# ============================================================

print("Building digital temporal features...")

for sensor in DIGITAL:

    state_source = df[
        f"{sensor}_mean"
    ]

    transition_source = df[
        f"{sensor}_transition_sum"
    ]

    for window in WINDOWS:

        # Duty cycle
        feature_df[
            f"{sensor}_duty_{window}m"
        ] = rolling_transform(
            state_source,
            df["segment_id"],
            window,
            "mean",
        )

        # Number of state transitions
        feature_df[
            f"{sensor}_transitions_{window}m"
        ] = rolling_transform(
            transition_source,
            df["segment_id"],
            window,
            "sum",
        )


# ============================================================
# ENGINEERING RELATIONSHIP HISTORY
# ============================================================

print("Building engineering relationship features...")

for window in WINDOWS:

    feature_df[
        f"COMP_MPG_mismatch_{window}m"
    ] = rolling_transform(
        df[
            "COMP_MPG_mismatch_mean"
        ],
        df["segment_id"],
        window,
        "mean",
    )

    feature_df[
        f"COMP_DV_equal_{window}m"
    ] = rolling_transform(
        df[
            "COMP_DV_equal_mean"
        ],
        df["segment_id"],
        window,
        "mean",
    )

    feature_df[
        f"TP3_Reservoir_abs_diff_{window}m"
    ] = rolling_transform(
        df[
            "TP3_Reservoir_abs_diff_mean"
        ],
        df["segment_id"],
        window,
        "mean",
    )


# ============================================================
# COMPRESSOR-STATE INTERACTION FEATURES
#
# These preserve an engineering distinction between the
# two observed COMP regimes.
# ============================================================

feature_df[
    "Motor_current_x_COMP0"
] = (
    df["Motor_current_mean"]
    *
    (1.0 - df["COMP_mean"])
)

feature_df[
    "Motor_current_x_COMP1"
] = (
    df["Motor_current_mean"]
    *
    df["COMP_mean"]
)

feature_df[
    "TP2_x_COMP0"
] = (
    df["TP2_mean"]
    *
    (1.0 - df["COMP_mean"])
)

feature_df[
    "H1_x_COMP1"
] = (
    df["H1_mean"]
    *
    df["COMP_mean"]
)


# ============================================================
# QUALITY / CONTINUITY TABLE
#
# These are NOT model features.
# ============================================================

quality_df = pd.DataFrame({
    "timestamp":
        df["timestamp"],

    "segment_id":
        df["segment_id"],

    "segment_position":
        df["segment_position"],

    "history_120m_complete":
        df["history_120m_complete"],

    "sample_count":
        df["sample_count"],

    "sampling_density":
        df["sampling_density"],

    "low_sample_minute":
        df["low_sample_minute"],

    "raw_gap_over_30s":
        df["raw_gap_over_30s"],
})


# ============================================================
# FEATURE MANIFEST
# ============================================================

manifest_rows = []

for column in feature_df.columns:

    if column == "timestamp":
        continue

    if "_duty_" in column:
        family = "digital_duty_cycle"

    elif "_transitions_" in column:
        family = "digital_transition"

    elif "mismatch" in column:
        family = "state_consistency"

    elif "COMP_DV_equal" in column:
        family = "state_consistency"

    elif "Reservoir_abs_diff" in column:
        family = "pressure_relationship"

    elif "_delta_" in column:
        family = "analogue_trend"

    elif "_volatility_" in column:
        family = "analogue_volatility"

    elif "_COMP" in column:
        family = "state_interaction"

    elif any(
        column.startswith(
            f"{sensor}_"
        )
        for sensor in ANALOG
    ):
        family = "analogue"

    elif any(
        column.startswith(
            f"{sensor}_"
        )
        for sensor in DIGITAL
    ):
        family = "digital"

    else:
        family = "engineering"

    manifest_rows.append({
        "feature": column,
        "family": family,
    })


manifest = pd.DataFrame(
    manifest_rows
)

manifest.to_csv(
    TABLE_PATH,
    index=False
)


# ============================================================
# SAVE
# ============================================================

feature_df.to_parquet(
    OUTPUT_PATH,
    index=False
)

quality_df.to_parquet(
    QUALITY_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

feature_columns = [
    c
    for c in feature_df.columns
    if c != "timestamp"
]

rows_complete = int(
    quality_df[
        "history_120m_complete"
    ].sum()
)

complete_feature_rows = int(
    feature_df.loc[
        quality_df[
            "history_120m_complete"
        ] == 1,
        feature_columns,
    ]
    .notna()
    .all(axis=1)
    .sum()
)


summary = {
    "rows": int(
        len(feature_df)
    ),

    "feature_count": int(
        len(feature_columns)
    ),

    "max_history_minutes": 120,

    "continuous_segments": int(
        quality_df[
            "segment_id"
        ].nunique()
    ),

    "rows_with_120m_history":
        rows_complete,

    "rows_with_120m_history_and_all_features":
        complete_feature_rows,

    "data_quality_columns_used_as_model_features":
        False,
}


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


print("\n" + "=" * 80)
print("FEATURE BUILD SUMMARY")
print("=" * 80)

for key, value in summary.items():
    print(
        f"{key}: {value}"
    )

print("\nFeature families:")

print(
    manifest[
        "family"
    ]
    .value_counts()
    .to_string()
)

print("\nSaved features:")
print(OUTPUT_PATH)

print("\nSaved quality metadata:")
print(QUALITY_PATH)

print("\nFEATURE BUILD COMPLETE")
