from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PRED_PATH = (
    ROOT
    / "artifacts"
    / "metrics"
    / "final_F04_predictions.parquet"
)

RESULT_PATH = (
    ROOT
    / "artifacts"
    / "metrics"
    / "final_F04_holdout_result.json"
)

OUTPUT_PATH = (
    ROOT
    / "artifacts"
    / "tables"
    / "final_F04_temporal_error_analysis.csv"
)


df = pd.read_parquet(
    PRED_PATH
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)


with open(
    RESULT_PATH,
    "r",
    encoding="utf-8",
) as f:

    result = json.load(f)


threshold = float(
    result["threshold_value"]
)


EVENT_START = pd.Timestamp(
    "2020-07-15 14:30:00"
)


failure = df[
    df["role"]
    == "holdout_failure"
].copy()


failure[
    "minutes_after_event_start"
] = (
    (
        failure["timestamp"]
        -
        EVENT_START
    )
    .dt.total_seconds()
    / 60.0
)


bins = [
    0,
    5,
    15,
    30,
    60,
    90,
    120,
    150,
    180,
    240,
    300,
]


labels = [
    "0-5",
    "5-15",
    "15-30",
    "30-60",
    "60-90",
    "90-120",
    "120-150",
    "150-180",
    "180-240",
    "240-300",
]


failure["time_bin"] = pd.cut(
    failure[
        "minutes_after_event_start"
    ],
    bins=bins,
    labels=labels,
    right=True,
    include_lowest=True,
)


rows = []


for time_bin, group in (
    failure
    .groupby(
        "time_bin",
        observed=True,
    )
):

    scores = group["score"]

    rows.append({
        "time_bin":
            str(time_bin),

        "observed_minutes":
            int(len(group)),

        "median_score":
            float(
                scores.median()
            ),

        "mean_score":
            float(
                scores.mean()
            ),

        "p90_score":
            float(
                scores.quantile(
                    0.90
                )
            ),

        "max_score":
            float(
                scores.max()
            ),

        "minutes_above_threshold":
            int(
                (
                    scores
                    >= threshold
                ).sum()
            ),

        "fraction_above_threshold":
            float(
                (
                    scores
                    >= threshold
                ).mean()
            ),

        "frozen_threshold":
            threshold,
    })


summary = pd.DataFrame(
    rows
)


summary.to_csv(
    OUTPUT_PATH,
    index=False,
)


print("=" * 100)
print("F04 TEMPORAL ERROR ANALYSIS")
print("=" * 100)

print(
    summary
    .round(5)
    .to_string(
        index=False
    )
)


# ============================================================
# FIRST RAW SCORE CROSSING
# ============================================================

above = failure[
    failure["score"]
    >= threshold
]


print()
print("Frozen threshold:")
print(threshold)


if len(above):

    first = (
        above.iloc[0]
    )

    print()
    print(
        "First individual score "
        "above threshold:"
    )

    print(
        first["timestamp"]
    )

    print(
        "Minutes after event start:",
        first[
            "minutes_after_event_start"
        ],
    )

    print(
        "Score:",
        first["score"],
    )


# ============================================================
# HIGHEST SCORES BEFORE FIRST ALERT
# ============================================================

pre_150 = failure[
    failure[
        "minutes_after_event_start"
    ] < 150
]


print()
print(
    "Highest 15 scores during "
    "first 150 minutes:"
)

print(
    pre_150[
        [
            "timestamp",
            "minutes_after_event_start",
            "score",
        ]
    ]
    .sort_values(
        "score",
        ascending=False,
    )
    .head(15)
    .to_string(
        index=False
    )
)


print()
print("Saved:")
print(OUTPUT_PATH)
