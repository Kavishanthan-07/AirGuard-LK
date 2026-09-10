
"""
AirGuard-LK — Final Duration-Based Escalation Analysis
======================================================

Purpose
-------
Convert the existing consensus anomaly detector from a sensitive EARLY WATCH
signal into a more selective WARNING state by requiring the consensus anomaly
condition to remain active continuously for a fixed duration.

NO model retraining.
NO threshold tuning.
NO feature changes.

Input
-----
artifacts/consensus_early_watch/consensus_scored_holdout.csv

The input must contain:
    timestamp
    consensus_persistent

Predeclared duration grid
-------------------------
30 minutes
60 minutes
120 minutes

Definition
----------
A WARNING begins only after the raw consensus-persistent condition has stayed
TRUE continuously for the required duration.

Example:
    consensus-persistent starts 10:00
    60-minute rule
    -> WARNING starts at 11:00 if the condition remained continuously TRUE

After escalation:
    -> merge WARNING episodes separated by <=30 minutes
    -> 6-hour cooldown
    -> strict early = NEW WARNING starts 24 h to 2 h before event

Development selection
---------------------
Use F01-F03 only:
    1) keep candidates with >=2/3 strict-new-early detections, if any
    2) minimize false WARNING episodes/day
    3) minimize WARNING occupancy
    4) prefer longer duration on ties

Then apply the selected duration unchanged to F04.

Scientific status
-----------------
This is a post-hoc operational sensitivity analysis. F04 has already been
examined and is not treated as a pristine holdout.

Outputs
-------
artifacts/duration_escalation/
    result.json
    duration_summary.csv
    duration_event_results.csv
    selected_candidate.json
    selected_F04_result.json
    selected_warning_episodes.csv
    selected_warning_timeline.csv
    f04_duration_comparison.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EVENTS = {
    "F01": (
        pd.Timestamp("2020-04-18 00:00:00"),
        pd.Timestamp("2020-04-18 23:59:00"),
    ),
    "F02": (
        pd.Timestamp("2020-05-29 23:30:00"),
        pd.Timestamp("2020-05-30 06:00:00"),
    ),
    "F03": (
        pd.Timestamp("2020-06-05 10:00:00"),
        pd.Timestamp("2020-06-07 14:30:00"),
    ),
    "F04": (
        pd.Timestamp("2020-07-15 14:30:00"),
        pd.Timestamp("2020-07-15 19:00:00"),
    ),
}

DURATIONS_MIN = [30, 60, 120]

BIN_MINUTES = 5
MERGE_MINUTES = 30
COOLDOWN_HOURS = 6
EARLY_START_HOURS = 24
EARLY_END_HOURS = 2

F04_EARLY_WINDOW_START = (
    EVENTS["F04"][0] - pd.Timedelta(hours=EARLY_START_HOURS)
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=Path(
            "artifacts/consensus_early_watch/"
            "consensus_scored_holdout.csv"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/duration_escalation"),
    )
    return p.parse_args()


def sustained_warning_flags(
    timestamps: pd.DatetimeIndex,
    persistent: np.ndarray,
    required_minutes: int,
) -> np.ndarray:
    """
    Turn on WARNING only after the consensus-persistent signal has remained
    continuously true for the requested duration.
    """
    required_steps = int(required_minutes / BIN_MINUTES)
    if required_steps < 1:
        raise ValueError("required duration too short")

    out = np.zeros(len(persistent), dtype=bool)
    streak = 0
    prev_t = None

    for i, (t, flag) in enumerate(zip(timestamps, persistent)):
        contiguous = (
            prev_t is not None
            and (t - prev_t).total_seconds() == BIN_MINUTES * 60
        )

        if not contiguous:
            streak = 0

        if bool(flag):
            streak += 1
        else:
            streak = 0

        if streak >= required_steps:
            out[i] = True

        prev_t = t

    return out


def raw_episodes(
    timestamps: pd.DatetimeIndex,
    flags: np.ndarray,
) -> list[dict]:
    episodes = []
    start = None
    end = None
    prev_t = None

    for t, flag in zip(timestamps, flags):
        contiguous = (
            prev_t is not None
            and (t - prev_t).total_seconds() == BIN_MINUTES * 60
        )

        if bool(flag):
            if start is None or not contiguous:
                if start is not None:
                    episodes.append({"start": start, "end": end})
                start = t
            end = t
        else:
            if start is not None:
                episodes.append({"start": start, "end": end})
                start = None
                end = None

        prev_t = t

    if start is not None:
        episodes.append({"start": start, "end": end})

    return episodes


def merge_episodes(episodes: list[dict]) -> list[dict]:
    if not episodes:
        return []

    gap = pd.Timedelta(minutes=MERGE_MINUTES)
    merged = [episodes[0].copy()]

    for ep in episodes[1:]:
        current = merged[-1]
        if ep["start"] - current["end"] <= gap:
            current["end"] = max(current["end"], ep["end"])
        else:
            merged.append(ep.copy())

    return merged


def apply_cooldown(episodes: list[dict]) -> list[dict]:
    if not episodes:
        return []

    kept = []
    blocked_until = pd.Timestamp.min
    cooldown = pd.Timedelta(hours=COOLDOWN_HOURS)

    for ep in episodes:
        if ep["start"] < blocked_until:
            continue
        kept.append(ep.copy())
        blocked_until = ep["end"] + cooldown

    return kept


def episodes_frame(episodes: list[dict]) -> pd.DataFrame:
    rows = []
    for i, ep in enumerate(episodes, start=1):
        rows.append({
            "episode_id": i,
            "start": ep["start"],
            "end": ep["end"],
            "duration_min": (
                (ep["end"] - ep["start"]).total_seconds() / 60
                + BIN_MINUTES
            ),
        })

    return pd.DataFrame(
        rows,
        columns=["episode_id", "start", "end", "duration_min"],
    )


def overlap(a_start, a_end, b_start, b_end) -> bool:
    return a_end >= b_start and a_start < b_end


def event_metrics(
    event_id: str,
    episodes: pd.DataFrame,
) -> dict:
    start, end = EVENTS[event_id]

    early_start = start - pd.Timedelta(hours=EARLY_START_HOURS)
    early_end = start - pd.Timedelta(hours=EARLY_END_HOURS)

    strict = episodes[
        (episodes["start"] >= early_start)
        & (episodes["start"] < early_end)
    ].sort_values("start")

    late = episodes[
        episodes.apply(
            lambda r: overlap(
                r["start"], r["end"], early_end, end
            ),
            axis=1,
        )
    ].sort_values("start")

    strict_detected = len(strict) > 0
    late_detected = len(late) > 0

    first_early = (
        strict.iloc[0]["start"] if strict_detected else pd.NaT
    )
    lead = (
        (start - first_early).total_seconds() / 60
        if strict_detected
        else np.nan
    )

    first_late = pd.NaT
    late_relative = np.nan

    if late_detected:
        row = late.iloc[0]
        first_late = max(row["start"], early_end)
        late_relative = (
            first_late - start
        ).total_seconds() / 60

    if strict_detected:
        status = "STRICT_EARLY"
    elif late_detected:
        status = "LATE"
    else:
        status = "MISSED"

    return {
        "event_id": event_id,
        "event_start": start,
        "status": status,
        "strict_new_early_detected": int(strict_detected),
        "first_strict_early_warning": first_early,
        "strict_early_lead_minutes": lead,
        "late_detected": int(late_detected),
        "first_late_warning": first_late,
        "late_warning_relative_to_start_minutes": late_relative,
    }


def event_related_mask(
    timestamps: pd.DatetimeIndex,
    event_ids: list[str],
) -> np.ndarray:
    mask = np.zeros(len(timestamps), dtype=bool)

    for event_id in event_ids:
        start, end = EVENTS[event_id]
        related_start = start - pd.Timedelta(hours=EARLY_START_HOURS)
        mask |= (
            (timestamps >= related_start)
            & (timestamps <= end)
        )

    return mask


def episode_related(
    row: pd.Series,
    event_ids: list[str],
) -> bool:
    for event_id in event_ids:
        start, end = EVENTS[event_id]
        related_start = start - pd.Timedelta(hours=EARLY_START_HOURS)
        if overlap(row["start"], row["end"], related_start, end):
            return True
    return False


def policy_flags(
    timestamps: pd.DatetimeIndex,
    episodes: pd.DataFrame,
) -> np.ndarray:
    flags = np.zeros(len(timestamps), dtype=bool)

    for _, ep in episodes.iterrows():
        flags |= (
            (timestamps >= ep["start"])
            & (timestamps <= ep["end"])
        )

    return flags


def nuisance_metrics(
    timestamps: pd.DatetimeIndex,
    episodes: pd.DataFrame,
    flags: np.ndarray,
    event_ids: list[str],
    end_time: pd.Timestamp | None = None,
) -> dict:
    if end_time is None:
        scope = np.ones(len(timestamps), dtype=bool)
        scoped_eps = episodes.copy()
    else:
        scope = timestamps < end_time
        scoped_eps = episodes[
            episodes["start"] < end_time
        ].copy()

    ts = timestamps[scope]
    scoped_flags = flags[scope]

    related = event_related_mask(ts, event_ids)
    normal = ~related

    false_eps = scoped_eps[
        ~scoped_eps.apply(
            lambda r: episode_related(r, event_ids),
            axis=1,
        )
    ]

    normal_days = normal.sum() * BIN_MINUTES / (60 * 24)
    fa_day = (
        len(false_eps) / normal_days
        if normal_days > 0
        else np.nan
    )

    return {
        "false_warning_episodes": int(len(false_eps)),
        "normal_exposure_days": float(normal_days),
        "false_warning_episodes_per_day": float(fa_day),
        "warning_time_pct": 100.0 * float(scoped_flags.mean()),
    }


def evaluate_duration(
    df: pd.DataFrame,
    duration_min: int,
):
    timestamps = pd.DatetimeIndex(df["timestamp"])
    persistent = df["consensus_persistent"].astype(int).astype(bool).to_numpy()

    warning_raw = sustained_warning_flags(
        timestamps,
        persistent,
        duration_min,
    )

    episodes = raw_episodes(timestamps, warning_raw)
    episodes = merge_episodes(episodes)
    episodes = apply_cooldown(episodes)
    episodes_df = episodes_frame(episodes)

    flags = policy_flags(timestamps, episodes_df)

    event_df = pd.DataFrame(
        [event_metrics(event_id, episodes_df) for event_id in EVENTS]
    )

    dev_events = event_df[
        event_df["event_id"].isin(["F01", "F02", "F03"])
    ]

    dev_early = int(
        dev_events["strict_new_early_detected"].sum()
    )
    all_early = int(
        event_df["strict_new_early_detected"].sum()
    )

    dev_nuisance = nuisance_metrics(
        timestamps=timestamps,
        episodes=episodes_df,
        flags=flags,
        event_ids=["F01", "F02", "F03"],
        end_time=F04_EARLY_WINDOW_START,
    )

    all_nuisance = nuisance_metrics(
        timestamps=timestamps,
        episodes=episodes_df,
        flags=flags,
        event_ids=["F01", "F02", "F03", "F04"],
    )

    summary = {
        "duration_minutes": duration_min,
        "dev_strict_early_events": dev_early,
        "all_strict_early_events": all_early,
        "dev_false_warnings_per_day": dev_nuisance[
            "false_warning_episodes_per_day"
        ],
        "dev_warning_time_pct": dev_nuisance[
            "warning_time_pct"
        ],
        "all_false_warnings_per_day": all_nuisance[
            "false_warning_episodes_per_day"
        ],
        "all_warning_time_pct": all_nuisance[
            "warning_time_pct"
        ],
        "total_warning_episodes": len(episodes_df),
    }

    return summary, event_df, episodes_df, flags


def select_candidate(table: pd.DataFrame) -> pd.Series:
    eligible = table[
        table["dev_strict_early_events"] >= 2
    ].copy()

    if len(eligible) == 0:
        eligible = table.copy()

    ranked = eligible.sort_values(
        [
            "dev_strict_early_events",
            "dev_false_warnings_per_day",
            "dev_warning_time_pct",
            "duration_minutes",
        ],
        ascending=[False, True, True, False],
    )

    return ranked.iloc[0]


def json_value(v):
    if isinstance(v, pd.Timestamp):
        return v.isoformat(sep=" ")
    if isinstance(v, np.generic):
        v = v.item()
    if pd.isna(v):
        return None
    return v


def save_f04_plot(
    df: pd.DataFrame,
    result_cache: dict,
    out_path: Path,
) -> None:
    start, end = EVENTS["F04"]

    lo = start - pd.Timedelta(hours=30)
    hi = end + pd.Timedelta(hours=8)

    view = df[
        (df["timestamp"] >= lo)
        & (df["timestamp"] <= hi)
    ].copy()

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.step(
        view["timestamp"],
        view["consensus_persistent"].astype(int),
        where="post",
        linewidth=1.2,
        label="Consensus EARLY WATCH",
    )

    levels = {
        30: 1.10,
        60: 1.20,
        120: 1.30,
    }

    for duration in DURATIONS_MIN:
        flags = result_cache[duration]["flags"]
        mask = (
            (df["timestamp"] >= lo)
            & (df["timestamp"] <= hi)
        ).to_numpy()

        ax.step(
            view["timestamp"],
            flags[mask].astype(int) * levels[duration],
            where="post",
            linewidth=1.3,
            label=f"{duration}-min WARNING",
        )

    ax.axvspan(
        start - pd.Timedelta(hours=24),
        start - pd.Timedelta(hours=2),
        alpha=0.10,
        label="Strict early window",
    )

    ax.axvspan(
        start - pd.Timedelta(hours=2),
        start,
        alpha=0.08,
        label="Late pre-fault",
    )

    ax.axvspan(
        start,
        end,
        alpha=0.08,
        label="F04",
    )

    ax.set_ylim(-0.05, 1.42)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No", "Active"])
    ax.set_title("F04 — Duration-Based Escalation from Early Watch to Warning")
    ax.set_xlabel("Time")
    ax.set_ylabel("Operational state")
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    df = pd.read_csv(
        args.input,
        parse_dates=["timestamp"],
    ).sort_values("timestamp").reset_index(drop=True)

    required = {"timestamp", "consensus_persistent"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    summaries = []
    event_tables = []
    cache = {}

    for duration in DURATIONS_MIN:
        summary, event_df, episodes_df, flags = evaluate_duration(
            df,
            duration,
        )

        summaries.append(summary)

        event_df = event_df.copy()
        event_df["duration_minutes"] = duration
        event_tables.append(event_df)

        cache[duration] = {
            "episodes": episodes_df,
            "flags": flags,
        }

    summary_df = pd.DataFrame(summaries)
    event_df_all = pd.concat(event_tables, ignore_index=True)

    selected = select_candidate(summary_df)
    selected_duration = int(selected["duration_minutes"])

    selected_events = event_df_all[
        event_df_all["duration_minutes"] == selected_duration
    ].copy()

    selected_f04 = selected_events[
        selected_events["event_id"] == "F04"
    ].iloc[0]

    selected_episodes = cache[selected_duration]["episodes"]
    selected_flags = cache[selected_duration]["flags"]

    selected_timeline = df[["timestamp", "consensus_persistent"]].copy()
    selected_timeline["selected_warning"] = selected_flags.astype(int)

    result = {
        "analysis": "AirGuard-LK final duration-based escalation analysis",
        "status": "post-hoc operational sensitivity",
        "model_retrained": False,
        "thresholds_changed": False,
        "duration_grid_minutes": DURATIONS_MIN,
        "selection_uses_F04_outcome": False,
        "selection_rule": [
            "retain candidates with >=2/3 F01-F03 strict-new-early warnings if possible",
            "maximize F01-F03 strict-new-early warnings",
            "minimize development false-warning episodes/day",
            "minimize development warning occupancy",
            "prefer longer duration on tie",
        ],
        "selected_duration_minutes": selected_duration,
        "selected_development": {
            k: json_value(v)
            for k, v in selected.to_dict().items()
        },
        "selected_event_results": [
            {
                k: json_value(v)
                for k, v in row.items()
            }
            for row in selected_events.to_dict(orient="records")
        ],
        "selected_F04": {
            k: json_value(v)
            for k, v in selected_f04.to_dict().items()
        },
    }

    summary_df.to_csv(
        args.out_dir / "duration_summary.csv",
        index=False,
    )

    event_df_all.to_csv(
        args.out_dir / "duration_event_results.csv",
        index=False,
    )

    selected_episodes.to_csv(
        args.out_dir / "selected_warning_episodes.csv",
        index=False,
    )

    selected_timeline.to_csv(
        args.out_dir / "selected_warning_timeline.csv",
        index=False,
    )

    with open(
        args.out_dir / "result.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2)

    with open(
        args.out_dir / "selected_candidate.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "selected_duration_minutes": selected_duration,
                "development_metrics": {
                    k: json_value(v)
                    for k, v in selected.to_dict().items()
                },
            },
            f,
            indent=2,
        )

    with open(
        args.out_dir / "selected_F04_result.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                k: json_value(v)
                for k, v in selected_f04.to_dict().items()
            },
            f,
            indent=2,
        )

    save_f04_plot(
        df,
        cache,
        args.out_dir / "f04_duration_comparison.png",
    )

    print("\n" + "=" * 110)
    print("DURATION SUMMARY")
    print("=" * 110)

    print(
        summary_df[
            [
                "duration_minutes",
                "dev_strict_early_events",
                "all_strict_early_events",
                "dev_false_warnings_per_day",
                "dev_warning_time_pct",
                "all_false_warnings_per_day",
                "all_warning_time_pct",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 110)
    print("EVENT RESULTS BY DURATION")
    print("=" * 110)

    print(
        event_df_all[
            [
                "duration_minutes",
                "event_id",
                "status",
                "strict_new_early_detected",
                "first_strict_early_warning",
                "strict_early_lead_minutes",
                "late_detected",
                "late_warning_relative_to_start_minutes",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 110)
    print("SELECTED USING F01-F03 ONLY")
    print("=" * 110)

    print(
        f"Duration: {selected_duration} min | "
        f"Dev strict early: "
        f"{int(selected['dev_strict_early_events'])}/3 | "
        f"Dev false warnings/day: "
        f"{selected['dev_false_warnings_per_day']:.4f} | "
        f"Dev warning time: "
        f"{selected['dev_warning_time_pct']:.2f}%"
    )

    print("\n" + "=" * 110)
    print("F04 UNDER UNCHANGED SELECTED RULE")
    print("=" * 110)

    print("Status:", selected_f04["status"])

    if int(selected_f04["strict_new_early_detected"]) == 1:
        lead = float(selected_f04["strict_early_lead_minutes"])
        print(
            f"Strict warning lead: {lead:.1f} min "
            f"({lead/60:.2f} h) before F04"
        )
        print(
            "First warning:",
            selected_f04["first_strict_early_warning"],
        )
        print("This materially beats the frozen +150 min LightGBM confirmation.")
    elif int(selected_f04["late_detected"]) == 1:
        rel = float(
            selected_f04["late_warning_relative_to_start_minutes"]
        )
        if rel < 0:
            print(
                f"Warning occurred {abs(rel):.1f} min before F04, "
                "inside the final 2-hour window."
            )
        else:
            print(
                f"Warning occurred {rel:.1f} min after F04 start."
            )
            if rel < 150:
                print(
                    f"Improvement over LightGBM: {150-rel:.1f} min."
                )
    else:
        print("No useful F04 warning under the selected duration.")

    print("\nArtifacts saved to:", args.out_dir.resolve())


if __name__ == "__main__":
    main()
