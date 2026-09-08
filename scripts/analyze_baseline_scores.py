from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PRED_PATH = (
    ROOT
    / "artifacts"
    / "metrics"
    / "development_baseline_predictions.parquet"
)

TABLE_DIR = (
    ROOT
    / "artifacts"
    / "tables"
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 86)
print("AIRGUARD-LK — BASELINE SCORE DISTRIBUTION ANALYSIS")
print("=" * 86)


pred = pd.read_parquet(
    PRED_PATH
)

pred["timestamp"] = pd.to_datetime(
    pred["timestamp"]
)


rows = []


for (
    split,
    horizon,
    model,
), group in pred.groupby(
    [
        "split",
        "horizon_minutes",
        "model",
    ]
):

    for role in [
        "validation_negative",
        "validation_ambiguous",
        "validation_positive",
    ]:

        values = (
            group.loc[
                group["role"] == role,
                "score",
            ]
            .dropna()
        )

        if len(values) == 0:
            continue

        rows.append({
            "split":
                split,

            "horizon_minutes":
                horizon,

            "model":
                model,

            "role":
                role,

            "samples":
                len(values),

            "mean":
                float(values.mean()),

            "median":
                float(values.median()),

            "p50":
                float(
                    values.quantile(0.50)
                ),

            "p75":
                float(
                    values.quantile(0.75)
                ),

            "p90":
                float(
                    values.quantile(0.90)
                ),

            "p95":
                float(
                    values.quantile(0.95)
                ),

            "p99":
                float(
                    values.quantile(0.99)
                ),

            "max":
                float(values.max()),
        })


summary = pd.DataFrame(
    rows
)

summary.to_csv(
    TABLE_DIR
    / "development_baseline_score_distributions.csv",
    index=False
)


# ============================================================
# POSITIVE VS NEGATIVE RANKING DIAGNOSTIC
# ============================================================

diagnostic_rows = []


for (
    split,
    horizon,
    model,
), group in pred.groupby(
    [
        "split",
        "horizon_minutes",
        "model",
    ]
):

    negative = (
        group.loc[
            group["role"]
            == "validation_negative",
            "score",
        ]
        .dropna()
    )

    positive = (
        group.loc[
            group["role"]
            == "validation_positive",
            "score",
        ]
        .dropna()
    )

    if (
        len(negative) == 0
        or
        len(positive) == 0
    ):
        continue

    negative_quantiles = {
        "neg_p95":
            negative.quantile(0.95),

        "neg_p99":
            negative.quantile(0.99),

        "neg_p995":
            negative.quantile(0.995),

        "neg_p999":
            negative.quantile(0.999),
    }

    positive_above = {}

    for name, threshold in (
        negative_quantiles.items()
    ):

        positive_above[
            f"positive_fraction_above_{name}"
        ] = float(
            (
                positive
                >= threshold
            ).mean()
        )


    diagnostic_rows.append({
        "split":
            split,

        "horizon_minutes":
            horizon,

        "model":
            model,

        "positive_median":
            float(
                positive.median()
            ),

        "negative_median":
            float(
                negative.median()
            ),

        **{
            k: float(v)
            for k, v
            in negative_quantiles.items()
        },

        **positive_above,
    })


diagnostics = pd.DataFrame(
    diagnostic_rows
)

diagnostics.to_csv(
    TABLE_DIR
    / "development_baseline_ranking_diagnostics.csv",
    index=False
)


# ============================================================
# CONSOLE
# ============================================================

pd.set_option(
    "display.width",
    220,
)

pd.set_option(
    "display.max_columns",
    None,
)


print("\n" + "=" * 86)
print("LOGISTIC TEMPORAL SCORE DISTRIBUTIONS")
print("=" * 86)

view = summary[
    summary["model"]
    == "Logistic Temporal"
]

print(
    view[
        [
            "split",
            "horizon_minutes",
            "role",
            "samples",
            "median",
            "p90",
            "p95",
            "p99",
            "max",
        ]
    ]
    .round(5)
    .to_string(index=False)
)


print("\n" + "=" * 86)
print("TEMPORAL LOGISTIC RANKING DIAGNOSTIC")
print("=" * 86)

view = diagnostics[
    diagnostics["model"]
    == "Logistic Temporal"
]

print(
    view
    .round(5)
    .to_string(index=False)
)


print("\nSaved:")
print(
    TABLE_DIR
    / "development_baseline_score_distributions.csv"
)

print(
    TABLE_DIR
    / "development_baseline_ranking_diagnostics.csv"
)

print("\nANALYSIS COMPLETE")
