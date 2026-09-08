from pathlib import Path
import pandas as pd
import numpy as np

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

OUTPUT = (
    ROOT
    / "artifacts"
    / "tables"
    / "metropt3_event_coverage.csv"
)

print("=" * 72)
print("AIRGUARD-LK — FAILURE EVENT COVERAGE AUDIT")
print("=" * 72)

print("\nLoading timestamps...")

df = pd.read_csv(
    CSV_PATH,
    usecols=["timestamp"]
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

WINDOWS_MINUTES = [
    15,
    30,
    60,
    120,
    360,
    1440,
]

rows = []

for _, event in events.iterrows():

    event_id = event["event_id"]
    start = event["start_time"]
    end = event["end_time"]

    result = {
        "event_id": event_id,
        "failure_type": event["failure_type"],
        "start_time": start,
        "end_time": end,
        "duration_hours": (
            end - start
        ).total_seconds() / 3600,
    }

    # ----------------------------------------------
    # Samples during documented failure
    # ----------------------------------------------

    failure_mask = (
        (df["timestamp"] >= start)
        &
        (df["timestamp"] <= end)
    )

    failure_times = (
        df.loc[failure_mask, "timestamp"]
    )

    result["failure_samples"] = int(
        len(failure_times)
    )

    if len(failure_times) >= 2:

        failure_gaps = (
            failure_times
            .diff()
            .dt.total_seconds()
            .dropna()
        )

        result["failure_max_gap_s"] = float(
            failure_gaps.max()
        )

    else:

        result["failure_max_gap_s"] = np.nan

    # ----------------------------------------------
    # Pre-fault warning windows
    # ----------------------------------------------

    for minutes in WINDOWS_MINUTES:

        window_start = (
            start
            - pd.Timedelta(minutes=minutes)
        )

        mask = (
            (df["timestamp"] >= window_start)
            &
            (df["timestamp"] < start)
        )

        times = df.loc[
            mask,
            "timestamp"
        ]

        result[
            f"pre_{minutes}m_samples"
        ] = int(len(times))

        if len(times) >= 2:

            gaps = (
                times.diff()
                .dt.total_seconds()
                .dropna()
            )

            result[
                f"pre_{minutes}m_max_gap_s"
            ] = float(gaps.max())

        else:

            result[
                f"pre_{minutes}m_max_gap_s"
            ] = np.nan

        # Approximate coverage against observed
        # nominal cadence of one sample / 10 sec.

        expected = int(
            minutes * 60 / 10
        )

        result[
            f"pre_{minutes}m_expected"
        ] = expected

        result[
            f"pre_{minutes}m_coverage_pct"
        ] = (
            len(times)
            / expected
            * 100
            if expected > 0
            else np.nan
        )

    # ----------------------------------------------
    # Closest timestamps before/after event start
    # ----------------------------------------------

    before = df.loc[
        df["timestamp"] < start,
        "timestamp"
    ]

    after = df.loc[
        df["timestamp"] >= start,
        "timestamp"
    ]

    if len(before):

        previous_ts = before.iloc[-1]

        result[
            "last_sample_before_start"
        ] = previous_ts

        result[
            "gap_before_event_start_s"
        ] = (
            start - previous_ts
        ).total_seconds()

    if len(after):

        next_ts = after.iloc[0]

        result[
            "first_sample_at_or_after_start"
        ] = next_ts

        result[
            "gap_after_event_start_s"
        ] = (
            next_ts - start
        ).total_seconds()

    rows.append(result)

coverage = pd.DataFrame(rows)

coverage.to_csv(
    OUTPUT,
    index=False
)

print("\nEVENT COVERAGE")
print("-" * 72)

display_cols = [
    "event_id",
    "duration_hours",
    "failure_samples",
    "pre_60m_samples",
    "pre_60m_coverage_pct",
    "pre_120m_samples",
    "pre_120m_coverage_pct",
    "gap_before_event_start_s",
]

print(
    coverage[display_cols]
    .to_string(index=False)
)

print("\nSaved:")
print(OUTPUT)
