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

EVENT_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "metropt3_failure_events.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "metropt3_1min_labelled.parquet"
)

TABLE_DIR = (
    ROOT
    / "artifacts"
    / "tables"
)

METRIC_DIR = (
    ROOT
    / "artifacts"
    / "metrics"
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRIC_DIR.mkdir(
    parents=True,
    exist_ok=True
)


PRE_HORIZONS = [
    15,
    30,
    60,
    120,
    360,
]

POST_HORIZONS = [
    60,
    120,
    360,
]


print("=" * 78)
print("AIRGUARD-LK — MASTER EVENT LABEL CONSTRUCTION")
print("=" * 78)


# ============================================================
# LOAD
# ============================================================

print("\nLoading one-minute table...")

df = pd.read_parquet(
    INPUT_PATH
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
    parse_dates=[
        "start_time",
        "end_time",
    ]
)

events = (
    events.sort_values("start_time")
          .reset_index(drop=True)
)


print(f"Minute rows: {len(df):,}")
print(f"Documented events: {len(events)}")


# ============================================================
# INITIALIZE EVENT COLUMNS
# ============================================================

df["failure_active"] = np.int8(0)

df["failure_event_id"] = pd.Series(
    pd.NA,
    index=df.index,
    dtype="string",
)

df["next_event_id"] = pd.Series(
    pd.NA,
    index=df.index,
    dtype="string",
)

df["previous_event_id"] = pd.Series(
    pd.NA,
    index=df.index,
    dtype="string",
)

df["minutes_to_failure"] = np.nan
df["minutes_since_failure_end"] = np.nan


# ============================================================
# INITIALIZE PRE-FAILURE FLAGS
# ============================================================

for horizon in PRE_HORIZONS:

    df[
        f"pre_{horizon}m"
    ] = np.int8(0)


# ============================================================
# INITIALIZE POST-FAILURE FLAGS
# ============================================================

for horizon in POST_HORIZONS:

    df[
        f"post_{horizon}m"
    ] = np.int8(0)


# ============================================================
# EVENT LABELS
# ============================================================

for _, event in events.iterrows():

    event_id = str(
        event["event_id"]
    )

    start = pd.Timestamp(
        event["start_time"]
    )

    end = pd.Timestamp(
        event["end_time"]
    )

    # --------------------------------------------------------
    # ACTIVE FAILURE INTERVAL
    # --------------------------------------------------------

    failure_mask = (
        (df["timestamp"] >= start)
        &
        (df["timestamp"] <= end)
    )

    df.loc[
        failure_mask,
        "failure_active"
    ] = 1

    df.loc[
        failure_mask,
        "failure_event_id"
    ] = event_id


    # --------------------------------------------------------
    # PRE-FAILURE WINDOWS
    #
    # Labels use future event timestamps because they are
    # SUPERVISED TARGETS.
    #
    # Future information must NEVER appear in model features.
    # --------------------------------------------------------

    for horizon in PRE_HORIZONS:

        pre_start = (
            start
            - pd.Timedelta(
                minutes=horizon
            )
        )

        mask = (
            (df["timestamp"] >= pre_start)
            &
            (df["timestamp"] < start)
        )

        df.loc[
            mask,
            f"pre_{horizon}m"
        ] = 1


    # --------------------------------------------------------
    # POST-FAILURE WINDOWS
    #
    # These are marked rather than automatically discarded.
    # We will later decide which recovery buffer is appropriate.
    # --------------------------------------------------------

    for horizon in POST_HORIZONS:

        post_end = (
            end
            + pd.Timedelta(
                minutes=horizon
            )
        )

        mask = (
            (df["timestamp"] > end)
            &
            (df["timestamp"] <= post_end)
        )

        df.loc[
            mask,
            f"post_{horizon}m"
        ] = 1


# ============================================================
# MINUTES TO NEXT FAILURE
#
# We intentionally calculate this independently from the
# binary horizon labels for later survival/risk analysis.
# ============================================================

for _, event in events.iterrows():

    event_id = str(
        event["event_id"]
    )

    start = pd.Timestamp(
        event["start_time"]
    )

    delta_minutes = (
        (
            start
            -
            df["timestamp"]
        )
        .dt.total_seconds()
        / 60.0
    )

    candidate = (
        delta_minutes >= 0
    )

    update = (
        candidate
        &
        (
            df[
                "minutes_to_failure"
            ].isna()
            |
            (
                delta_minutes
                <
                df[
                    "minutes_to_failure"
                ]
            )
        )
    )

    df.loc[
        update,
        "minutes_to_failure"
    ] = delta_minutes[
        update
    ]

    df.loc[
        update,
        "next_event_id"
    ] = event_id


# ============================================================
# MINUTES SINCE PREVIOUS FAILURE END
# ============================================================

for _, event in events.iterrows():

    event_id = str(
        event["event_id"]
    )

    end = pd.Timestamp(
        event["end_time"]
    )

    delta_minutes = (
        (
            df["timestamp"]
            -
            end
        )
        .dt.total_seconds()
        / 60.0
    )

    candidate = (
        delta_minutes >= 0
    )

    update = (
        candidate
        &
        (
            df[
                "minutes_since_failure_end"
            ].isna()
            |
            (
                delta_minutes
                <
                df[
                    "minutes_since_failure_end"
                ]
            )
        )
    )

    df.loc[
        update,
        "minutes_since_failure_end"
    ] = delta_minutes[
        update
    ]

    df.loc[
        update,
        "previous_event_id"
    ] = event_id


# ============================================================
# HUMAN-READABLE EXCLUSIVE PHASE
#
# This column is for EDA/reporting.
# ML targets remain the individual horizon columns.
# ============================================================

df["event_phase"] = "normal_or_other"

df.loc[
    df["post_360m"] == 1,
    "event_phase"
] = "post_failure_360m"

df.loc[
    df["pre_360m"] == 1,
    "event_phase"
] = "pre_360_to_120m"

df.loc[
    df["pre_120m"] == 1,
    "event_phase"
] = "pre_120_to_60m"

df.loc[
    df["pre_60m"] == 1,
    "event_phase"
] = "pre_60_to_30m"

df.loc[
    df["pre_30m"] == 1,
    "event_phase"
] = "pre_30_to_15m"

df.loc[
    df["pre_15m"] == 1,
    "event_phase"
] = "pre_15m"

df.loc[
    df["failure_active"] == 1,
    "event_phase"
] = "failure"


# ============================================================
# EVENT-SPECIFIC LABEL COUNTS
# ============================================================

event_rows = []

for _, event in events.iterrows():

    event_id = str(
        event["event_id"]
    )

    start = pd.Timestamp(
        event["start_time"]
    )

    end = pd.Timestamp(
        event["end_time"]
    )

    row = {
        "event_id": event_id,
        "start_time": start,
        "end_time": end,
    }

    failure_mask = (
        (df["timestamp"] >= start)
        &
        (df["timestamp"] <= end)
    )

    row[
        "failure_minutes_observed"
    ] = int(
        failure_mask.sum()
    )

    for horizon in PRE_HORIZONS:

        window_start = (
            start
            -
            pd.Timedelta(
                minutes=horizon
            )
        )

        mask = (
            (df["timestamp"] >= window_start)
            &
            (df["timestamp"] < start)
        )

        row[
            f"pre_{horizon}m_observed_minutes"
        ] = int(
            mask.sum()
        )

    event_rows.append(
        row
    )


event_counts = pd.DataFrame(
    event_rows
)

event_counts.to_csv(
    TABLE_DIR
    / "metropt3_event_label_counts.csv",
    index=False
)


# ============================================================
# GLOBAL LABEL COUNTS
# ============================================================

global_rows = []

global_rows.append({
    "label": "failure_active",
    "positive_minutes": int(
        df[
            "failure_active"
        ].sum()
    ),
    "positive_fraction": float(
        df[
            "failure_active"
        ].mean()
    ),
})

for horizon in PRE_HORIZONS:

    column = (
        f"pre_{horizon}m"
    )

    global_rows.append({
        "label": column,
        "positive_minutes": int(
            df[column].sum()
        ),
        "positive_fraction": float(
            df[column].mean()
        ),
    })

for horizon in POST_HORIZONS:

    column = (
        f"post_{horizon}m"
    )

    global_rows.append({
        "label": column,
        "positive_minutes": int(
            df[column].sum()
        ),
        "positive_fraction": float(
            df[column].mean()
        ),
    })


global_counts = pd.DataFrame(
    global_rows
)

global_counts.to_csv(
    TABLE_DIR
    / "metropt3_global_label_counts.csv",
    index=False
)


# ============================================================
# EVENT PHASE COUNTS
# ============================================================

phase_counts = (
    df["event_phase"]
    .value_counts()
    .rename_axis(
        "event_phase"
    )
    .reset_index(
        name="minutes"
    )
)

phase_counts[
    "fraction"
] = (
    phase_counts["minutes"]
    / len(df)
)

phase_counts.to_csv(
    TABLE_DIR
    / "metropt3_event_phase_counts.csv",
    index=False
)


# ============================================================
# SANITY CHECKS
# ============================================================

# Nested pre-failure windows must be logically consistent.

assert (
    df.loc[
        df["pre_15m"] == 1,
        "pre_30m"
    ]
    == 1
).all()

assert (
    df.loc[
        df["pre_30m"] == 1,
        "pre_60m"
    ]
    == 1
).all()

assert (
    df.loc[
        df["pre_60m"] == 1,
        "pre_120m"
    ]
    == 1
).all()

assert (
    df.loc[
        df["pre_120m"] == 1,
        "pre_360m"
    ]
    == 1
).all()

# No pre-failure sample should simultaneously be inside a
# documented failure interval.

for horizon in PRE_HORIZONS:

    overlap = (
        (
            df[
                f"pre_{horizon}m"
            ] == 1
        )
        &
        (
            df[
                "failure_active"
            ] == 1
        )
    )

    assert not overlap.any()


# ============================================================
# SAVE
# ============================================================

df.to_parquet(
    OUTPUT_PATH,
    index=False
)


summary = {
    "rows": int(
        len(df)
    ),

    "documented_failure_events":
        int(len(events)),

    "failure_minutes":
        int(
            df[
                "failure_active"
            ].sum()
        ),

    "pre_15m_minutes":
        int(
            df[
                "pre_15m"
            ].sum()
        ),

    "pre_30m_minutes":
        int(
            df[
                "pre_30m"
            ].sum()
        ),

    "pre_60m_minutes":
        int(
            df[
                "pre_60m"
            ].sum()
        ),

    "pre_120m_minutes":
        int(
            df[
                "pre_120m"
            ].sum()
        ),

    "pre_360m_minutes":
        int(
            df[
                "pre_360m"
            ].sum()
        ),
}

with open(
    METRIC_DIR
    / "metropt3_label_build.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print("\n" + "=" * 78)
print("EVENT-SPECIFIC LABEL COUNTS")
print("=" * 78)

print(
    event_counts
    .to_string(
        index=False
    )
)

print("\n" + "=" * 78)
print("GLOBAL LABEL COUNTS")
print("=" * 78)

print(
    global_counts
    .to_string(
        index=False
    )
)

print("\n" + "=" * 78)
print("EVENT PHASE COUNTS")
print("=" * 78)

print(
    phase_counts
    .to_string(
        index=False
    )
)

print("\nSaved:")
print(OUTPUT_PATH)

print("\nLABEL BUILD COMPLETE")
