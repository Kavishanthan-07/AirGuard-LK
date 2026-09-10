
"""
AirGuard-LK — Fixed Consensus Early-Watch Experiment
====================================================

Purpose
-------
Test whether agreement between the two already-trained anomaly detectors can
preserve useful early-event timing while reducing nuisance alerts.

NO retraining.
NO threshold tuning.
NO architecture changes.

Consensus rule
--------------
At each aligned 5-minute timestamp:

    consensus_persistent =
        global_IF_persistent
        AND
        state_conditioned_IF_persistent

The same downstream policy is then applied:

    -> merge alert episodes separated by <=30 minutes
    -> 6-hour cooldown
    -> strict early hit = a NEW alert episode starts 24 h to 2 h before event

Scientific status
-----------------
This is a post-hoc consensus analysis. F04 has already been examined in prior
AirGuard-LK experiments, so it is not described as a pristine holdout.

Expected inputs
---------------
artifacts/isolation_forest_early_watch/scored_holdout.csv
artifacts/state_conditioned_isolation_forest/scored_holdout.csv

Optional comparison inputs
--------------------------
artifacts/isolation_forest_nuisance_sensitivity/development_selected_candidate.json
artifacts/state_conditioned_isolation_forest/result.json

Outputs
-------
artifacts/consensus_early_watch/
    result.json
    per_event_results.csv
    alert_episodes.csv
    consensus_scored_holdout.csv
    model_comparison.csv
    f04_consensus_timeline.png
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

BIN_MINUTES = 5
EPISODE_MERGE_MINUTES = 30
COOLDOWN_HOURS = 6
EARLY_START_HOURS = 24
EARLY_END_HOURS = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--global-scored",
        type=Path,
        default=Path("artifacts/isolation_forest_early_watch/scored_holdout.csv"),
    )
    p.add_argument(
        "--state-scored",
        type=Path,
        default=Path("artifacts/state_conditioned_isolation_forest/scored_holdout.csv"),
    )
    p.add_argument(
        "--global-candidate",
        type=Path,
        default=Path(
            "artifacts/isolation_forest_nuisance_sensitivity/"
            "development_selected_candidate.json"
        ),
    )
    p.add_argument(
        "--state-result",
        type=Path,
        default=Path("artifacts/state_conditioned_isolation_forest/result.json"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/consensus_early_watch"),
    )

    return p.parse_args()


def overlap(a_start, a_end, b_start, b_end) -> bool:
    return a_end >= b_start and a_start < b_end


def raw_episodes(
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

    merge_gap = pd.Timedelta(minutes=EPISODE_MERGE_MINUTES)
    merged = [episodes[0].copy()]

    for ep in episodes[1:]:
        current = merged[-1]

        if ep["start"] - current["end"] <= merge_gap:
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
        rows.append(
            {
                "episode_id": i,
                "start": ep["start"],
                "end": ep["end"],
                "duration_min": (
                    (ep["end"] - ep["start"]).total_seconds() / 60
                    + BIN_MINUTES
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=["episode_id", "start", "end", "duration_min"],
    )


def event_metrics(event_id: str, episodes: pd.DataFrame) -> dict:
    start, end = EVENTS[event_id]

    early_start = start - pd.Timedelta(hours=EARLY_START_HOURS)
    early_end = start - pd.Timedelta(hours=EARLY_END_HOURS)

    strict_hits = episodes[
        (episodes["start"] >= early_start)
        & (episodes["start"] < early_end)
    ].sort_values("start")

    overlap_hits = episodes[
        episodes.apply(
            lambda r: overlap(
                r["start"], r["end"], early_start, early_end
            ),
            axis=1,
        )
    ].sort_values("start")

    late_hits = episodes[
        episodes.apply(
            lambda r: overlap(
                r["start"], r["end"], early_end, end
            ),
            axis=1,
        )
    ].sort_values("start")

    strict = len(strict_hits) > 0
    overlap_early = len(overlap_hits) > 0
    late = len(late_hits) > 0

    first_strict = (
        strict_hits.iloc[0]["start"] if strict else pd.NaT
    )
    strict_lead = (
        (start - first_strict).total_seconds() / 60
        if strict
        else np.nan
    )

    overlap_start = (
        overlap_hits.iloc[0]["start"] if overlap_early else pd.NaT
    )
    overlap_active_at_boundary = (
        overlap_early and overlap_start < early_start
    )

    first_late = pd.NaT
    late_relative = np.nan

    if late:
        row = late_hits.iloc[0]
        first_late = max(row["start"], early_end)
        late_relative = (
            first_late - start
        ).total_seconds() / 60

    if strict:
        status = "STRICT_EARLY"
    elif late:
        status = "LATE"
    elif overlap_early:
        status = "ONGOING_AT_EARLY_BOUNDARY"
    else:
        status = "MISSED"

    return {
        "event_id": event_id,
        "event_start": start,
        "status": status,
        "strict_new_early_detected": int(strict),
        "first_strict_early_alert": first_strict,
        "strict_early_lead_minutes": strict_lead,
        "overlapping_early_detected": int(overlap_early),
        "overlap_episode_start": overlap_start,
        "overlap_already_active_at_24h_boundary": int(
            overlap_active_at_boundary
        ),
        "late_detected": int(late),
        "first_late_alert": first_late,
        "late_alert_relative_to_start_minutes": late_relative,
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


def episode_related_to_events(
    row: pd.Series,
    event_ids: list[str],
) -> bool:
    for event_id in event_ids:
        start, end = EVENTS[event_id]
        related_start = start - pd.Timedelta(hours=EARLY_START_HOURS)

        if overlap(
            row["start"], row["end"], related_start, end
        ):
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
            lambda r: episode_related_to_events(r, event_ids),
            axis=1,
        )
    ]

    normal_days = (
        normal.sum() * BIN_MINUTES / (60 * 24)
    )

    fa_day = (
        len(false_eps) / normal_days
        if normal_days > 0
        else np.nan
    )

    return {
        "false_alert_episodes": int(len(false_eps)),
        "normal_exposure_days": float(normal_days),
        "false_alert_episodes_per_day": float(fa_day),
        "policy_time_in_alert_pct": 100.0 * float(scoped_flags.mean()),
    }


def json_value(v):
    if isinstance(v, pd.Timestamp):
        return v.isoformat(sep=" ")
    if isinstance(v, np.generic):
        v = v.item()
    if pd.isna(v):
        return None
    return v


def load_optional_json(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def build_comparison(
    consensus_result: dict,
    global_json: dict,
    state_json: dict,
) -> pd.DataFrame:
    rows = []

    rows.append(
        {
            "model": "consensus_AND",
            "dev_strict_early_events": consensus_result["development_F01_F03"][
                "strict_new_early_events"
            ],
            "dev_false_alerts_per_day": consensus_result[
                "development_F01_F03"
            ]["false_alert_episodes_per_day"],
            "dev_policy_time_pct": consensus_result[
                "development_F01_F03"
            ]["policy_time_in_alert_pct"],
            "F04_status": consensus_result["F04"]["status"],
            "F04_lead_minutes": consensus_result["F04"][
                "strict_early_lead_minutes"
            ],
        }
    )

    candidate = global_json.get("candidate", {})
    if candidate:
        rows.append(
            {
                "model": "global_iforest",
                "dev_strict_early_events": (
                    f"{int(candidate.get('strict_new_early_dev_F01_F03', 0))}/3"
                ),
                "dev_false_alerts_per_day": candidate.get(
                    "false_alert_episodes_per_day_dev"
                ),
                "dev_policy_time_pct": candidate.get(
                    "policy_time_in_alert_pct_dev"
                ),
                "F04_status": "STRICT_EARLY",
                "F04_lead_minutes": 945.0,
            }
        )

    if state_json:
        dev = state_json.get("development_F01_F03", {})
        f04 = state_json.get("F04", {})

        rows.append(
            {
                "model": "state_conditioned_iforest",
                "dev_strict_early_events": dev.get(
                    "strict_new_early_events"
                ),
                "dev_false_alerts_per_day": dev.get(
                    "false_alert_episodes_per_day"
                ),
                "dev_policy_time_pct": dev.get(
                    "policy_time_in_alert_pct"
                ),
                "F04_status": f04.get("status"),
                "F04_lead_minutes": f04.get(
                    "strict_early_lead_minutes"
                ),
            }
        )

    return pd.DataFrame(rows)


def save_f04_plot(
    aligned: pd.DataFrame,
    episodes: pd.DataFrame,
    out_path: Path,
) -> None:
    start, end = EVENTS["F04"]

    lo = start - pd.Timedelta(hours=30)
    hi = end + pd.Timedelta(hours=8)

    view = aligned[
        (aligned["timestamp"] >= lo)
        & (aligned["timestamp"] <= hi)
    ].copy()

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.step(
        view["timestamp"],
        view["global_persistent"],
        where="post",
        linewidth=1.2,
        label="Global IF persistent",
    )

    ax.step(
        view["timestamp"],
        view["state_persistent"],
        where="post",
        linewidth=1.2,
        label="State IF persistent",
    )

    ax.step(
        view["timestamp"],
        view["consensus_persistent"].astype(int) * 1.15,
        where="post",
        linewidth=1.8,
        label="Consensus AND",
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

    for _, ep in episodes.iterrows():
        if ep["end"] < lo or ep["start"] > hi:
            continue
        ax.axvspan(
            max(ep["start"], lo),
            min(ep["end"], hi),
            alpha=0.15,
        )

    ax.set_ylim(-0.05, 1.35)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No", "Persistent"])
    ax.set_title("F04 — Global ∩ State-Conditioned Consensus Early Watch")
    ax.set_xlabel("Time")
    ax.set_ylabel("Persistent anomaly condition")
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.global_scored.exists():
        raise FileNotFoundError(args.global_scored)

    if not args.state_scored.exists():
        raise FileNotFoundError(args.state_scored)

    global_df = pd.read_csv(
        args.global_scored,
        parse_dates=["timestamp"],
    )

    state_df = pd.read_csv(
        args.state_scored,
        parse_dates=["timestamp"],
    )

    global_col = None
    for candidate in [
        "raw_persistent_alert",
        "persistent_alert",
    ]:
        if candidate in global_df.columns:
            global_col = candidate
            break

    if global_col is None:
        raise ValueError(
            "Global scored file needs raw_persistent_alert or persistent_alert."
        )

    if "persistent_alert" not in state_df.columns:
        raise ValueError(
            "State scored file needs persistent_alert."
        )

    g = global_df[
        ["timestamp", global_col]
    ].rename(columns={global_col: "global_persistent"})

    s = state_df[
        ["timestamp", "persistent_alert"]
    ].rename(columns={"persistent_alert": "state_persistent"})

    aligned = (
        g.merge(s, on="timestamp", how="inner")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if len(aligned) == 0:
        raise ValueError("No aligned timestamps between the two scored files.")

    aligned["global_persistent"] = (
        aligned["global_persistent"].astype(int).astype(bool)
    )
    aligned["state_persistent"] = (
        aligned["state_persistent"].astype(int).astype(bool)
    )

    aligned["consensus_persistent"] = (
        aligned["global_persistent"]
        & aligned["state_persistent"]
    )

    timestamps = pd.DatetimeIndex(aligned["timestamp"])
    consensus = aligned["consensus_persistent"].to_numpy(dtype=bool)

    episodes = raw_episodes(timestamps, consensus)
    episodes = merge_episodes(episodes)
    episodes = apply_cooldown(episodes)
    episodes_df = episodes_to_frame(episodes)

    policy = policy_flags(timestamps, episodes_df)
    aligned["consensus_policy_alert"] = policy.astype(int)

    per_event = pd.DataFrame(
        [event_metrics(event_id, episodes_df) for event_id in EVENTS]
    )

    f04_dev_end = (
        EVENTS["F04"][0] - pd.Timedelta(hours=EARLY_START_HOURS)
    )

    dev_nuisance = nuisance_metrics(
        timestamps=timestamps,
        episodes=episodes_df,
        flags=policy,
        event_ids=["F01", "F02", "F03"],
        end_time=f04_dev_end,
    )

    all_nuisance = nuisance_metrics(
        timestamps=timestamps,
        episodes=episodes_df,
        flags=policy,
        event_ids=["F01", "F02", "F03", "F04"],
    )

    dev_strict = int(
        per_event.loc[
            per_event["event_id"].isin(["F01", "F02", "F03"]),
            "strict_new_early_detected",
        ].sum()
    )

    all_strict = int(
        per_event["strict_new_early_detected"].sum()
    )

    f04_row = per_event.loc[
        per_event["event_id"] == "F04"
    ].iloc[0]

    result = {
        "analysis": "AirGuard-LK fixed AND-consensus early-watch experiment",
        "status": "post-hoc consensus analysis",
        "model_retrained": False,
        "thresholds_changed": False,
        "consensus_rule": (
            "global Isolation Forest persistent alert AND "
            "state-conditioned Isolation Forest persistent alert"
        ),
        "downstream_policy": {
            "episode_merge_minutes": EPISODE_MERGE_MINUTES,
            "cooldown_hours": COOLDOWN_HOURS,
            "strict_early_window": "24 h to 2 h before event start",
        },
        "aligned_windows": int(len(aligned)),
        "development_F01_F03": {
            "strict_new_early_events": f"{dev_strict}/3",
            **dev_nuisance,
        },
        "all_F01_F04": {
            "strict_new_early_events": f"{all_strict}/4",
            **all_nuisance,
        },
        "event_results": [
            {
                k: json_value(v)
                for k, v in row.items()
            }
            for row in per_event.to_dict(orient="records")
        ],
        "F04": {
            k: json_value(v)
            for k, v in f04_row.to_dict().items()
        },
    }

    global_json = load_optional_json(args.global_candidate)
    state_json = load_optional_json(args.state_result)

    comparison = build_comparison(
        consensus_result=result,
        global_json=global_json,
        state_json=state_json,
    )

    with open(
        args.out_dir / "result.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2)

    per_event.to_csv(
        args.out_dir / "per_event_results.csv",
        index=False,
    )

    episodes_df.to_csv(
        args.out_dir / "alert_episodes.csv",
        index=False,
    )

    aligned.to_csv(
        args.out_dir / "consensus_scored_holdout.csv",
        index=False,
    )

    comparison.to_csv(
        args.out_dir / "model_comparison.csv",
        index=False,
    )

    save_f04_plot(
        aligned,
        episodes_df,
        args.out_dir / "f04_consensus_timeline.png",
    )

    print("\n" + "=" * 110)
    print("CONSENSUS EVENT RESULTS")
    print("=" * 110)

    print(
        per_event[
            [
                "event_id",
                "status",
                "strict_new_early_detected",
                "first_strict_early_alert",
                "strict_early_lead_minutes",
                "late_detected",
                "late_alert_relative_to_start_minutes",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 110)
    print("DEVELOPMENT F01-F03")
    print("=" * 110)

    print("Strict early:", f"{dev_strict}/3")
    print(
        "False alerts/day:",
        f"{dev_nuisance['false_alert_episodes_per_day']:.4f}",
    )
    print(
        "Policy time in alert:",
        f"{dev_nuisance['policy_time_in_alert_pct']:.2f}%",
    )

    print("\n" + "=" * 110)
    print("F04")
    print("=" * 110)

    print("Status:", f04_row["status"])

    if int(f04_row["strict_new_early_detected"]) == 1:
        lead = float(f04_row["strict_early_lead_minutes"])
        print(
            f"Strict new early lead: {lead:.1f} min ({lead/60:.2f} h)"
        )
        print(
            "First strict early alert:",
            f04_row["first_strict_early_alert"],
        )
        print("This materially beats the frozen +150 min LightGBM timing.")
    elif int(f04_row["late_detected"]) == 1:
        rel = float(
            f04_row["late_alert_relative_to_start_minutes"]
        )

        if rel < 0:
            print(
                f"Alert occurred {abs(rel):.1f} min before F04, "
                "inside the final 2-hour window."
            )
        else:
            print(f"Alert occurred {rel:.1f} min after F04.")
            if rel < 150:
                print(
                    f"Improvement over LightGBM: {150-rel:.1f} min."
                )
    else:
        print("No useful F04 consensus alert.")

    print("\n" + "=" * 110)
    print("MODEL COMPARISON")
    print("=" * 110)

    print(comparison.to_string(index=False))

    print("\nArtifacts saved to:", args.out_dir.resolve())


if __name__ == "__main__":
    main()
