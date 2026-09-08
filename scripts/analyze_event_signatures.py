from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    ROOT
    / "data"
    / "raw"
    / "metropt3"
    / "MetroPT3(AirCompressor).csv"
)

EVENT_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "metropt3_failure_events.csv"
)

TABLE_DIR = ROOT / "artifacts" / "tables"
FIGURE_DIR = ROOT / "artifacts" / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


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


print("=" * 76)
print("AIRGUARD-LK — EVENT-CENTERED EDA")
print("=" * 76)

print("\nLoading MetroPT-3...")

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

events = pd.read_csv(
    EVENT_PATH,
    parse_dates=["start_time", "end_time"]
)

print(f"Loaded {len(df):,} rows")
print(f"Events: {len(events)}")


# ============================================================
# PERIOD HELPERS
# ============================================================

def select_period(data, start, end):
    return data[
        (data["timestamp"] >= start)
        &
        (data["timestamp"] < end)
    ]


def analog_stats(series):
    if len(series) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "q25": np.nan,
            "q75": np.nan,
        }

    return {
        "n": len(series),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "q25": float(series.quantile(0.25)),
        "q75": float(series.quantile(0.75)),
    }


analog_rows = []
digital_rows = []


# ============================================================
# EVENT STATISTICS
# ============================================================

for _, event in events.iterrows():

    event_id = event["event_id"]

    start = event["start_time"]
    end = event["end_time"]

    # Reference behaviour well before the leak.
    baseline = select_period(
        df,
        start - pd.Timedelta(hours=24),
        start - pd.Timedelta(hours=2),
    )

    # Two separate pre-fault hours.
    pre_120_to_60 = select_period(
        df,
        start - pd.Timedelta(hours=2),
        start - pd.Timedelta(hours=1),
    )

    pre_60 = select_period(
        df,
        start - pd.Timedelta(hours=1),
        start,
    )

    failure = select_period(
        df,
        start,
        end + pd.Timedelta(seconds=1),
    )

    periods = {
        "baseline_24h_to_2h": baseline,
        "pre_120_to_60m": pre_120_to_60,
        "pre_60m": pre_60,
        "failure": failure,
    }

    # --------------------------------------------------------
    # ANALOG SIGNALS
    # --------------------------------------------------------

    for sensor in ANALOG:

        base_stats = analog_stats(
            baseline[sensor]
        )

        base_iqr = (
            base_stats["q75"]
            -
            base_stats["q25"]
        )

        for period_name, period_df in periods.items():

            stats = analog_stats(
                period_df[sensor]
            )

            robust_shift = np.nan

            if (
                period_name != "baseline_24h_to_2h"
                and
                np.isfinite(base_iqr)
                and
                base_iqr > 1e-9
            ):
                robust_shift = (
                    stats["median"]
                    -
                    base_stats["median"]
                ) / base_iqr

            analog_rows.append({
                "event_id": event_id,
                "sensor": sensor,
                "period": period_name,

                "n": stats["n"],
                "mean": stats["mean"],
                "median": stats["median"],
                "std": stats["std"],
                "q25": stats["q25"],
                "q75": stats["q75"],

                "baseline_median":
                    base_stats["median"],

                "baseline_iqr":
                    base_iqr,

                "robust_median_shift_vs_baseline":
                    robust_shift,
            })

    # --------------------------------------------------------
    # DIGITAL SIGNALS
    # --------------------------------------------------------

    for sensor in DIGITAL:

        baseline_active = (
            float(baseline[sensor].mean())
            if len(baseline)
            else np.nan
        )

        for period_name, period_df in periods.items():

            active_fraction = (
                float(period_df[sensor].mean())
                if len(period_df)
                else np.nan
            )

            digital_rows.append({
                "event_id": event_id,
                "sensor": sensor,
                "period": period_name,

                "samples": len(period_df),

                "active_fraction":
                    active_fraction,

                "baseline_active_fraction":
                    baseline_active,

                "delta_vs_baseline":
                    (
                        active_fraction
                        -
                        baseline_active
                    )
                    if (
                        np.isfinite(active_fraction)
                        and
                        np.isfinite(baseline_active)
                    )
                    else np.nan,
            })


analog_summary = pd.DataFrame(
    analog_rows
)

digital_summary = pd.DataFrame(
    digital_rows
)

analog_summary.to_csv(
    TABLE_DIR
    / "metropt3_event_analog_summary.csv",
    index=False
)

digital_summary.to_csv(
    TABLE_DIR
    / "metropt3_event_digital_summary.csv",
    index=False
)


# ============================================================
# COMPACT PRE-60-MINUTE EFFECT TABLE
# ============================================================

pre60_effect = (
    analog_summary[
        analog_summary["period"] == "pre_60m"
    ]
    .pivot(
        index="event_id",
        columns="sensor",
        values="robust_median_shift_vs_baseline",
    )
)

print("\n" + "=" * 76)
print("PRE-60 MIN ANALOG ROBUST SHIFTS")
print("0 = similar to baseline")
print("positive = higher than baseline")
print("negative = lower than baseline")
print("=" * 76)

print(
    pre60_effect
    .round(3)
    .to_string()
)


digital_pre60 = (
    digital_summary[
        digital_summary["period"] == "pre_60m"
    ]
    .pivot(
        index="event_id",
        columns="sensor",
        values="delta_vs_baseline",
    )
)

print("\n" + "=" * 76)
print("PRE-60 MIN DIGITAL DUTY-CYCLE CHANGES")
print("=" * 76)

print(
    digital_pre60
    .round(3)
    .to_string()
)


# ============================================================
# EVENT-CENTERED VISUALIZATIONS
# Visualization uses 1-minute aggregation only.
# This does NOT modify the ML dataset.
# ============================================================

for _, event in events.iterrows():

    event_id = event["event_id"]
    start = event["start_time"]
    end = event["end_time"]

    plot_start = (
        start
        - pd.Timedelta(hours=24)
    )

    plot_end = (
        end
        + pd.Timedelta(hours=2)
    )

    window = df[
        (df["timestamp"] >= plot_start)
        &
        (df["timestamp"] <= plot_end)
    ].copy()

    window = (
        window
        .set_index("timestamp")
    )

    # --------------------------------------------------------
    # Analog plot
    # --------------------------------------------------------

    analog_1m = (
        window[ANALOG]
        .resample("1min")
        .mean()
    )

    fig, axes = plt.subplots(
        len(ANALOG),
        1,
        figsize=(14, 14),
        sharex=True,
    )

    for ax, sensor in zip(
        axes,
        ANALOG
    ):

        ax.plot(
            analog_1m.index,
            analog_1m[sensor],
            linewidth=0.8,
        )

        ax.axvline(
            start,
            linestyle="--",
            linewidth=1,
        )

        ax.axvspan(
            start,
            end,
            alpha=0.15,
        )

        ax.set_ylabel(sensor)
        ax.grid(
            alpha=0.2
        )

    axes[0].set_title(
        f"{event_id} — Analog Signals\n"
        "24 h before leak → failure interval → 2 h after"
    )

    axes[-1].set_xlabel(
        "Timestamp"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / f"{event_id}_analog_event_window.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Digital plot
    # --------------------------------------------------------

    # Mean within each minute = fraction of that minute
    # for which the digital state was active.
    digital_1m = (
        window[DIGITAL]
        .resample("1min")
        .mean()
    )

    fig, axes = plt.subplots(
        len(DIGITAL),
        1,
        figsize=(14, 14),
        sharex=True,
    )

    for ax, sensor in zip(
        axes,
        DIGITAL
    ):

        ax.plot(
            digital_1m.index,
            digital_1m[sensor],
            linewidth=0.8,
        )

        ax.axvline(
            start,
            linestyle="--",
            linewidth=1,
        )

        ax.axvspan(
            start,
            end,
            alpha=0.15,
        )

        ax.set_ylabel(sensor)
        ax.set_ylim(-0.05, 1.05)

        ax.grid(
            alpha=0.2
        )

    axes[0].set_title(
        f"{event_id} — Digital/State Signals\n"
        "24 h before leak → failure interval → 2 h after"
    )

    axes[-1].set_xlabel(
        "Timestamp"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / f"{event_id}_digital_event_window.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


print("\nSaved tables:")
print(
    TABLE_DIR
    / "metropt3_event_analog_summary.csv"
)
print(
    TABLE_DIR
    / "metropt3_event_digital_summary.csv"
)

print("\nSaved event figures to:")
print(FIGURE_DIR)

print("\nEVENT-CENTERED EDA COMPLETE")
