
"""
AirGuard-LK — Isolation Forest Nuisance-Control Sensitivity
==========================================================

Purpose
-------
Evaluate whether stricter operating points can preserve useful early anomaly
detection while materially reducing nuisance-alert burden.

IMPORTANT:
- The Isolation Forest is NOT retrained.
- EWMA alpha is NOT changed.
- The feature representation is NOT changed.
- Only the frozen February-calibration threshold quantile and persistence rule
  are varied over a small predeclared grid.

Grid
----
Threshold quantile:
    0.995
    0.999
    0.9995

Persistence:
    3-of-4
    4-of-4

Fixed:
    EWMA alpha = 0.2
    episode merge = 30 minutes
    cooldown = 6 hours
    early window = 24 h to 2 h before event

Primary sensitivity question
----------------------------
Can we materially reduce:
    - false-alert episodes/day
    - policy time in alert %

while preserving a meaningful early F04 anomaly signal?

Evaluation note
---------------
The previous experiment counted an episode that was already active before the
start of the 24-hour early window as an "early" hit by clipping it to the
boundary. That can make lead time appear as exactly 1440 minutes.

This sensitivity therefore reports TWO event notions:

1. strict_new_early:
   A NEW alert episode must START inside the 24 h to 2 h pre-event window.

2. overlapping_early:
   Any alert episode overlapping that window.

The strict_new_early metric is the preferred competition-facing metric.

Selection note
--------------
Because F04 has already been examined, this is a POST-HOC sensitivity study,
not a new pristine holdout.

For convenience, the script identifies a "development-selected candidate"
using only:
    - F01/F02/F03 strict-new-early detection
    - normal exposure BEFORE the F04 early-warning window

Then it applies that candidate unchanged to F04 descriptively.

Expected inputs
---------------
artifacts/isolation_forest_early_watch/calibration_scores.csv
artifacts/isolation_forest_early_watch/scored_holdout.csv

Outputs
-------
artifacts/isolation_forest_nuisance_sensitivity/
    sensitivity_all_operating_points.csv
    sensitivity_event_results.csv
    development_selected_candidate.json
    development_selected_F04_result.json
    pareto_summary.csv
    f04_operating_point_comparison.png
    nuisance_tradeoff.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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

THRESHOLD_QUANTILES = [0.995, 0.999, 0.9995]
PERSISTENCE_RULES = [(3, 4), (4, 4)]

BIN_MINUTES = 5
EPISODE_MERGE_MINUTES = 30
COOLDOWN_HOURS = 6
EARLY_START_HOURS = 24
EARLY_END_HOURS = 2

# For development-only candidate selection.
F04_EARLY_WINDOW_START = EVENTS["F04"][0] - pd.Timedelta(hours=EARLY_START_HOURS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("artifacts/isolation_forest_early_watch"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/isolation_forest_nuisance_sensitivity"),
    )
    return p.parse_args()


def persistence_flags(
    timestamps: pd.DatetimeIndex,
    scores: np.ndarray,
    threshold: float,
    required: int,
    window: int,
) -> np.ndarray:
    exceed = np.asarray(scores) >= threshold
    out = np.zeros(len(exceed), dtype=bool)

    history = []
    previous = None

    for i, (t, flag) in enumerate(zip(timestamps, exceed)):
        if (
            previous is None
            or (t - previous).total_seconds() != BIN_MINUTES * 60
        ):
            history = []

        history.append(bool(flag))
        if len(history) > window:
            history.pop(0)

        if len(history) == window and sum(history) >= required:
            out[i] = True

        previous = t

    return out


def raw_alert_episodes(
    timestamps: pd.DatetimeIndex,
    flags: np.ndarray,
) -> list[dict]:
    episodes = []
    start = None
    end = None
    previous = None

    for t, flag in zip(timestamps, flags):
        contiguous = (
            previous is not None
            and (t - previous).total_seconds() == BIN_MINUTES * 60
        )

        if flag:
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

        previous = t

    if start is not None:
        episodes.append({"start": start, "end": end})

    return episodes


def merge_episodes(episodes: list[dict]) -> list[dict]:
    if not episodes:
        return []

    gap = pd.Timedelta(minutes=EPISODE_MERGE_MINUTES)
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

    cooldown = pd.Timedelta(hours=COOLDOWN_HOURS)
    kept = []
    blocked_until = pd.Timestamp.min

    for ep in episodes:
        if ep["start"] < blocked_until:
            continue

        kept.append(ep.copy())
        blocked_until = ep["end"] + cooldown

    return kept


def episodes_to_frame(episodes: list[dict]) -> pd.DataFrame:
    rows = []

    for i, ep in enumerate(episodes, start=1):
        duration = (
            (ep["end"] - ep["start"]).total_seconds() / 60
            + BIN_MINUTES
        )
        rows.append(
            {
                "episode_id": i,
                "start": ep["start"],
                "end": ep["end"],
                "duration_min": duration,
            }
        )

    return pd.DataFrame(
        rows,
        columns=["episode_id", "start", "end", "duration_min"],
    )


def overlap(
    a_start: pd.Timestamp,
    a_end: pd.Timestamp,
    b_start: pd.Timestamp,
    b_end: pd.Timestamp,
) -> bool:
    return a_end >= b_start and a_start < b_end


def event_metrics(
    event_id: str,
    episodes: pd.DataFrame,
) -> dict:
    start, end = EVENTS[event_id]

    early_start = start - pd.Timedelta(hours=EARLY_START_HOURS)
    early_end = start - pd.Timedelta(hours=EARLY_END_HOURS)

    # STRICT: new episode STARTS inside early window.
    strict_hits = episodes[
        (episodes["start"] >= early_start)
        & (episodes["start"] < early_end)
    ].sort_values("start")

    # LEGACY/OVERLAP: any episode overlapping early window.
    overlap_hits = episodes[
        episodes.apply(
            lambda r: overlap(
                r["start"],
                r["end"],
                early_start,
                early_end,
            ),
            axis=1,
        )
    ].sort_values("start")

    late_start = early_end
    late_end = end

    late_hits = episodes[
        episodes.apply(
            lambda r: overlap(
                r["start"],
                r["end"],
                late_start,
                late_end,
            ),
            axis=1,
        )
    ].sort_values("start")

    strict_early = len(strict_hits) > 0
    overlapping_early = len(overlap_hits) > 0
    late = len(late_hits) > 0

    strict_time = strict_hits.iloc[0]["start"] if strict_early else pd.NaT
    strict_lead = (
        (start - strict_time).total_seconds() / 60
        if strict_early
        else np.nan
    )

    overlap_time = overlap_hits.iloc[0]["start"] if overlapping_early else pd.NaT

    # If the overlap episode starts before the early window, expose that fact.
    overlap_already_active = (
        overlapping_early and overlap_time < early_start
    )

    late_time = pd.NaT
    late_relative = np.nan

    if late:
        # Actual episode start can precede late_start; use first time the
        # episode is inside the late window for a comparable event-relative time.
        row = late_hits.iloc[0]
        late_time = max(row["start"], late_start)
        late_relative = (
            late_time - start
        ).total_seconds() / 60

    if strict_early:
        preferred_status = "STRICT_EARLY"
    elif late:
        preferred_status = "LATE"
    elif overlapping_early:
        preferred_status = "ONGOING_AT_EARLY_BOUNDARY"
    else:
        preferred_status = "MISSED"

    return {
        "event_id": event_id,
        "event_start": start,
        "preferred_status": preferred_status,
        "strict_new_early_detected": int(strict_early),
        "strict_first_early_alert": strict_time,
        "strict_early_lead_minutes": strict_lead,
        "overlapping_early_detected": int(overlapping_early),
        "overlap_episode_start": overlap_time,
        "overlap_already_active_at_24h_boundary": int(overlap_already_active),
        "late_detected": int(late),
        "first_late_alert": late_time,
        "late_alert_relative_to_start_minutes": late_relative,
    }


def event_related_interval(
    event_id: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = EVENTS[event_id]
    return (
        start - pd.Timedelta(hours=EARLY_START_HOURS),
        end,
    )


def episode_is_related_to_any_event(row: pd.Series) -> bool:
    for event_id in EVENTS:
        a, b = event_related_interval(event_id)
        if overlap(row["start"], row["end"], a, b):
            return True
    return False


def episode_is_related_to_dev_events(row: pd.Series) -> bool:
    for event_id in ["F01", "F02", "F03"]:
        a, b = event_related_interval(event_id)
        if overlap(row["start"], row["end"], a, b):
            return True
    return False


def point_is_related(
    timestamps: pd.DatetimeIndex,
    event_ids: list[str],
) -> np.ndarray:
    mask = np.zeros(len(timestamps), dtype=bool)

    for event_id in event_ids:
        a, b = event_related_interval(event_id)
        mask |= (timestamps >= a) & (timestamps <= b)

    return mask


def episode_flags(
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


def evaluate_operating_point(
    scored: pd.DataFrame,
    calibration: pd.DataFrame,
    quantile: float,
    required: int,
    window: int,
):
    threshold = float(
        np.quantile(
            calibration["ewma_score"].to_numpy(dtype=float),
            quantile,
        )
    )

    ts = pd.DatetimeIndex(scored["timestamp"])
    score = scored["ewma_score"].to_numpy(dtype=float)

    persistent = persistence_flags(
        timestamps=ts,
        scores=score,
        threshold=threshold,
        required=required,
        window=window,
    )

    episodes = raw_alert_episodes(ts, persistent)
    episodes = merge_episodes(episodes)
    episodes = apply_cooldown(episodes)
    episodes_df = episodes_to_frame(episodes)

    policy = episode_flags(ts, episodes_df)

    event_rows = pd.DataFrame(
        [event_metrics(event_id, episodes_df) for event_id in EVENTS]
    )

    strict_count_all = int(
        event_rows["strict_new_early_detected"].sum()
    )

    # Full descriptive nuisance exposure.
    related_all = point_is_related(
        ts,
        ["F01", "F02", "F03", "F04"],
    )
    normal_all = ~related_all

    false_eps_all = episodes_df[
        ~episodes_df.apply(
            episode_is_related_to_any_event,
            axis=1,
        )
    ]

    normal_days_all = (
        normal_all.sum() * BIN_MINUTES / (60 * 24)
    )
    fa_day_all = (
        len(false_eps_all) / normal_days_all
        if normal_days_all > 0
        else np.nan
    )

    # Development-only nuisance calculation:
    # March -> immediately before F04's 24h early window.
    dev_point_mask = ts < F04_EARLY_WINDOW_START
    dev_ts = ts[dev_point_mask]

    related_dev = point_is_related(
        dev_ts,
        ["F01", "F02", "F03"],
    )
    dev_normal = ~related_dev

    dev_eps = episodes_df[
        episodes_df["start"] < F04_EARLY_WINDOW_START
    ].copy()

    false_eps_dev = dev_eps[
        ~dev_eps.apply(
            episode_is_related_to_dev_events,
            axis=1,
        )
    ]

    dev_normal_days = (
        dev_normal.sum() * BIN_MINUTES / (60 * 24)
    )
    dev_fa_day = (
        len(false_eps_dev) / dev_normal_days
        if dev_normal_days > 0
        else np.nan
    )

    dev_policy = policy[dev_point_mask]
    dev_policy_time_pct = 100.0 * float(dev_policy.mean())

    dev_events = event_rows[
        event_rows["event_id"].isin(["F01", "F02", "F03"])
    ]
    strict_dev_count = int(
        dev_events["strict_new_early_detected"].sum()
    )

    row = {
        "threshold_quantile": quantile,
        "threshold": threshold,
        "persistence_required": required,
        "persistence_window": window,
        "persistence_label": f"{required}-of-{window}",
        "strict_new_early_all": strict_count_all,
        "strict_new_early_dev_F01_F03": strict_dev_count,
        "false_alert_episodes_all": len(false_eps_all),
        "false_alert_episodes_per_day_all": fa_day_all,
        "policy_time_in_alert_pct_all": 100.0 * float(policy.mean()),
        "persistent_time_in_alert_pct_all": 100.0 * float(persistent.mean()),
        "false_alert_episodes_dev": len(false_eps_dev),
        "false_alert_episodes_per_day_dev": dev_fa_day,
        "policy_time_in_alert_pct_dev": dev_policy_time_pct,
        "total_policy_episodes": len(episodes_df),
    }

    return row, event_rows, episodes_df, policy, persistent


def select_development_candidate(table: pd.DataFrame) -> pd.Series:
    """
    Predeclared development-only ranking:
      1) maximize strict-new-early detections among F01-F03
      2) minimize development false alerts/day
      3) minimize development policy time in alert
      4) prefer higher threshold quantile
      5) prefer stricter persistence
    """
    ranked = table.sort_values(
        [
            "strict_new_early_dev_F01_F03",
            "false_alert_episodes_per_day_dev",
            "policy_time_in_alert_pct_dev",
            "threshold_quantile",
            "persistence_required",
        ],
        ascending=[False, True, True, False, False],
    )

    return ranked.iloc[0]


def make_tradeoff_plot(table: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for _, row in table.iterrows():
        label = (
            f"q={row['threshold_quantile']}, "
            f"{row['persistence_label']}"
        )

        ax.scatter(
            row["false_alert_episodes_per_day_all"],
            row["policy_time_in_alert_pct_all"],
            s=80,
        )

        ax.annotate(
            label,
            (
                row["false_alert_episodes_per_day_all"],
                row["policy_time_in_alert_pct_all"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("False alert episodes/day")
    ax.set_ylabel("Policy time in alert (%)")
    ax.set_title("Isolation Forest Nuisance-Control Trade-off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_f04_plot(
    scored: pd.DataFrame,
    calibration: pd.DataFrame,
    table: pd.DataFrame,
    out_path: Path,
) -> None:
    start, end = EVENTS["F04"]
    lo = start - pd.Timedelta(hours=30)
    hi = end + pd.Timedelta(hours=8)

    view = scored[
        (scored["timestamp"] >= lo)
        & (scored["timestamp"] <= hi)
    ].copy()

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        view["timestamp"],
        view["ewma_score"],
        linewidth=1.3,
        label="EWMA anomaly score",
    )

    # Plot the three quantile thresholds once each.
    for q in THRESHOLD_QUANTILES:
        threshold = float(
            np.quantile(
                calibration["ewma_score"].to_numpy(dtype=float),
                q,
            )
        )
        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=1.0,
            label=f"q={q}",
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

    ax.set_title("F04 — Isolation Forest Threshold Sensitivity")
    ax.set_ylabel("EWMA anomaly score")
    ax.set_xlabel("Time")
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    calibration_path = args.input_dir / "calibration_scores.csv"
    scored_path = args.input_dir / "scored_holdout.csv"

    if not calibration_path.exists():
        raise FileNotFoundError(calibration_path)

    if not scored_path.exists():
        raise FileNotFoundError(scored_path)

    calibration = pd.read_csv(
        calibration_path,
        parse_dates=["timestamp"],
    )
    scored = pd.read_csv(
        scored_path,
        parse_dates=["timestamp"],
    )

    required_cal = {"timestamp", "ewma_score"}
    required_scored = {"timestamp", "ewma_score"}

    if not required_cal.issubset(calibration.columns):
        raise ValueError(
            f"Calibration file needs columns {required_cal}"
        )

    if not required_scored.issubset(scored.columns):
        raise ValueError(
            f"Scored holdout file needs columns {required_scored}"
        )

    all_rows = []
    all_event_rows = []

    cache = {}

    for quantile in THRESHOLD_QUANTILES:
        for required, window in PERSISTENCE_RULES:
            (
                row,
                event_rows,
                episodes_df,
                policy,
                persistent,
            ) = evaluate_operating_point(
                scored=scored,
                calibration=calibration,
                quantile=quantile,
                required=required,
                window=window,
            )

            all_rows.append(row)

            event_rows = event_rows.copy()
            event_rows["threshold_quantile"] = quantile
            event_rows["persistence_label"] = f"{required}-of-{window}"
            all_event_rows.append(event_rows)

            key = (quantile, required, window)
            cache[key] = {
                "episodes": episodes_df,
                "policy": policy,
                "persistent": persistent,
            }

    table = pd.DataFrame(all_rows)
    event_table = pd.concat(all_event_rows, ignore_index=True)

    # Identify development-selected candidate without using F04 event outcome.
    best = select_development_candidate(table)
    best_key = (
        float(best["threshold_quantile"]),
        int(best["persistence_required"]),
        int(best["persistence_window"]),
    )

    best_events = event_table[
        (event_table["threshold_quantile"] == best_key[0])
        & (
            event_table["persistence_label"]
            == f"{best_key[1]}-of-{best_key[2]}"
        )
    ].copy()

    f04 = best_events[
        best_events["event_id"] == "F04"
    ].iloc[0]

    # Simple Pareto-style ranking table for discussion.
    pareto = table[
        [
            "threshold_quantile",
            "persistence_label",
            "strict_new_early_dev_F01_F03",
            "strict_new_early_all",
            "false_alert_episodes_per_day_all",
            "policy_time_in_alert_pct_all",
            "false_alert_episodes_per_day_dev",
            "policy_time_in_alert_pct_dev",
        ]
    ].sort_values(
        [
            "strict_new_early_all",
            "false_alert_episodes_per_day_all",
            "policy_time_in_alert_pct_all",
        ],
        ascending=[False, True, True],
    )

    table.to_csv(
        args.out_dir / "sensitivity_all_operating_points.csv",
        index=False,
    )
    event_table.to_csv(
        args.out_dir / "sensitivity_event_results.csv",
        index=False,
    )
    pareto.to_csv(
        args.out_dir / "pareto_summary.csv",
        index=False,
    )

    candidate_payload = {
        "status": "post-hoc nuisance-control sensitivity",
        "model_retrained": False,
        "feature_representation_changed": False,
        "ewma_alpha_changed": False,
        "grid": {
            "threshold_quantiles": THRESHOLD_QUANTILES,
            "persistence_rules": [
                f"{r}-of-{w}" for r, w in PERSISTENCE_RULES
            ],
        },
        "selection_uses_F04_event_outcome": False,
        "selection_rule": [
            "maximize F01-F03 strict-new-early detections",
            "minimize development false-alert episodes/day",
            "minimize development policy time in alert",
            "prefer higher threshold",
            "prefer stricter persistence",
        ],
        "candidate": {
            key: (
                None
                if pd.isna(value)
                else (
                    value.item()
                    if isinstance(value, np.generic)
                    else value
                )
            )
            for key, value in best.to_dict().items()
        },
    }

    f04_payload = {
        "status": "post-hoc descriptive F04 result",
        "candidate_selected_without_F04_event_outcome": True,
        "F04": {
            key: (
                None
                if pd.isna(value)
                else (
                    value.isoformat(sep=" ")
                    if isinstance(value, pd.Timestamp)
                    else (
                        value.item()
                        if isinstance(value, np.generic)
                        else value
                    )
                )
            )
            for key, value in f04.to_dict().items()
        },
    }

    with open(
        args.out_dir / "development_selected_candidate.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(candidate_payload, f, indent=2)

    with open(
        args.out_dir / "development_selected_F04_result.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(f04_payload, f, indent=2)

    make_tradeoff_plot(
        table,
        args.out_dir / "nuisance_tradeoff.png",
    )

    make_f04_plot(
        scored,
        calibration,
        table,
        args.out_dir / "f04_operating_point_comparison.png",
    )

    print("\n" + "=" * 110)
    print("ALL SIX OPERATING POINTS")
    print("=" * 110)

    print(
        table[
            [
                "threshold_quantile",
                "persistence_label",
                "strict_new_early_dev_F01_F03",
                "strict_new_early_all",
                "false_alert_episodes_per_day_all",
                "policy_time_in_alert_pct_all",
                "false_alert_episodes_per_day_dev",
                "policy_time_in_alert_pct_dev",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 110)
    print("STRICT NEW-EARLY EVENT RESULTS")
    print("=" * 110)

    print(
        event_table[
            [
                "threshold_quantile",
                "persistence_label",
                "event_id",
                "preferred_status",
                "strict_new_early_detected",
                "strict_first_early_alert",
                "strict_early_lead_minutes",
                "overlap_already_active_at_24h_boundary",
                "late_detected",
                "late_alert_relative_to_start_minutes",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 110)
    print("DEVELOPMENT-SELECTED CANDIDATE (F01-F03 ONLY)")
    print("=" * 110)

    print(
        f"q={best['threshold_quantile']} | "
        f"{best['persistence_label']} | "
        f"F01-F03 strict early="
        f"{int(best['strict_new_early_dev_F01_F03'])}/3 | "
        f"dev FA/day="
        f"{best['false_alert_episodes_per_day_dev']:.4f} | "
        f"dev policy time="
        f"{best['policy_time_in_alert_pct_dev']:.2f}%"
    )

    print("\n" + "=" * 110)
    print("F04 UNDER THAT UNCHANGED CANDIDATE")
    print("=" * 110)

    print("Status:", f04["preferred_status"])

    if int(f04["strict_new_early_detected"]) == 1:
        print(
            "STRICT NEW EARLY ALERT:",
            f"{f04['strict_early_lead_minutes']:.1f} min before F04",
        )
        print(
            "First early alert:",
            f04["strict_first_early_alert"],
        )
        print("This materially overcomes the +150 min LightGBM limitation.")
    elif int(f04["late_detected"]) == 1:
        rel = f04["late_alert_relative_to_start_minutes"]

        if rel < 0:
            print(
                "Alert occurred",
                f"{abs(rel):.1f} min before F04",
                "but within the final 2-hour late-warning window.",
            )
        else:
            print(
                "Alert occurred",
                f"{rel:.1f} min after F04 start.",
            )
            if rel < 150:
                print(
                    f"Improvement over LightGBM = {150 - rel:.1f} min."
                )
            else:
                print("No improvement over the +150 min LightGBM alert.")
    else:
        print("No useful F04 alert under the selected operating point.")

    print("\nArtifacts saved to:", args.out_dir.resolve())


if __name__ == "__main__":
    main()
