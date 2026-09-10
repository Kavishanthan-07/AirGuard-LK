
"""
AirGuard-LK — Normal-Only Isolation Forest Early-Watch Challenger
=================================================================

Goal
----
Run one fixed anomaly-detection experiment aimed specifically at the weakness
revealed by the frozen LightGBM F04 result (+150 min persistent detection).

The question is different from supervised fault classification:

    "Does the compressor depart from February normal behaviour before a
     documented later leak event?"

This experiment is inspired by the published MetroGuard protocol for the same
MetroPT-3 dataset, while remaining explicitly a post-hoc challenger inside
AirGuard-LK because the AirGuard team has already examined F04.

Protocol frozen in this file
----------------------------
Raw MetroPT-3
    -> causal right-closed 5-minute bins
    -> analogue: mean/std/min/max/last
    -> digital: active ratio/last/transitions
    -> reject bins with <80% expected coverage
    -> 12 consecutive bins = 60-minute window
    -> flatten 12 x 59 = 708 features
    -> fit 300-tree Isolation Forest on 1-21 Feb only
    -> calibrate on 22-29 Feb only
    -> score March-September only
    -> calibration median/MAD score normalization
    -> causal EWMA alpha=0.2
    -> fixed calibration q=0.995 threshold
    -> alert requires 3 exceedances in the latest 4 five-minute scores
    -> merge alert episodes separated by <=30 minutes
    -> suppress new episodes during a 6-hour cooldown
    -> early window = 24 h to 2 h before event
    -> final 2 h before event / active event = late detection

No failure labels are used to fit the Isolation Forest or choose its threshold.

Outputs
-------
artifacts/isolation_forest_early_watch/
    result.json
    per_event_results.csv
    scored_holdout.csv
    calibration_scores.csv
    alert_episodes.csv
    isolation_forest.joblib
    config.json
    f04_early_watch_timeline.png
    all_events_timeline.png

Reference context
-----------------
MetroGuard public documentation reports:
- causal 5-minute bins
- 12-bin / 60-minute windows
- fit on 1-21 Feb
- calibration on 22-29 Feb
- March-September holdout
- 300-tree Isolation Forest comparator
- calibration q=0.995
- causal EWMA alpha=0.2
- 3-of-4 persistence
- 30-minute merge
- 6-hour cooldown
- early window 24 h to 2 h before incident

This script independently implements that documented design rather than
importing MetroGuard code.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    auc,
    precision_recall_curve,
    roc_auc_score,
)


SEED = 42

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


@dataclass(frozen=True)
class Protocol:
    bin_minutes: int = 5
    expected_samples_per_bin: int = 30
    minimum_coverage_fraction: float = 0.80
    window_bins: int = 12
    isolation_trees: int = 300
    isolation_max_samples: str = "auto"
    isolation_max_features: float = 1.0
    calibration_quantile: float = 0.995
    ewma_alpha: float = 0.20
    persistence_required: int = 3
    persistence_window: int = 4
    episode_merge_minutes: int = 30
    cooldown_hours: int = 6
    early_window_start_hours: int = 24
    early_window_end_hours: int = 2
    boundary_purge_minutes: int = 60
    random_seed: int = 42


P = Protocol()

FIT_RAW_START = pd.Timestamp("2020-02-01 00:00:00")
FIT_RAW_END = pd.Timestamp("2020-02-22 00:00:00")

CAL_RAW_START = pd.Timestamp("2020-02-22 00:00:00")
CAL_RAW_END = pd.Timestamp("2020-03-01 00:00:00")

HOLDOUT_RAW_START = pd.Timestamp("2020-03-01 00:00:00")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/raw/metropt3/MetroPT3(AirCompressor).csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/isolation_forest_early_watch"),
    )
    return parser.parse_args()


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)


def load_raw(path: Path) -> pd.DataFrame:
    print(f"Loading {path} ...")

    usecols = ["timestamp", *ANALOG, *DIGITAL]

    # The public MetroPT-3 CSV may contain an exported index. Reading only
    # required columns avoids carrying that column through memory.
    df = pd.read_csv(
        path,
        usecols=lambda c: c in usecols,
        low_memory=False,
    )

    missing = [c for c in usecols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = (
        df.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="first")
        .reset_index(drop=True)
    )

    if df[ANALOG + DIGITAL].isna().any().any():
        raise ValueError(
            "Unexpected missing sensor values. This experiment does not "
            "backfill or interpolate missing raw measurements."
        )

    print(f"Rows: {len(df):,}")
    print("Range:", df["timestamp"].min(), "->", df["timestamp"].max())

    return df


def build_causal_five_minute_bins(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Causal right-closed bins.

    Example:
        observations in (14:25, 14:30] are available at 14:30.

    Exact boundary timestamps remain assigned to that boundary.
    """
    df = raw.copy()

    # Right-closed endpoint.
    df["bin_end"] = df["timestamp"].dt.ceil(f"{P.bin_minutes}min")

    # Digital transitions occurring within the same bin only.
    transition_cols = []
    grouped_bin = df.groupby("bin_end", sort=False)

    for col in DIGITAL:
        name = f"{col}__transition"
        change = grouped_bin[col].diff()
        df[name] = change.fillna(0).ne(0).astype(np.int8)
        transition_cols.append(name)

    g = df.groupby("bin_end", sort=True)

    analog = g[ANALOG].agg(["mean", "std", "min", "max", "last"])
    analog.columns = [
        f"{sensor}_{stat}_5m"
        for sensor, stat in analog.columns
    ]

    digital_mean = g[DIGITAL].mean()
    digital_mean.columns = [
        f"{sensor}_active_ratio_5m" for sensor in DIGITAL
    ]

    digital_last = g[DIGITAL].last()
    digital_last.columns = [
        f"{sensor}_last_5m" for sensor in DIGITAL
    ]

    digital_trans = g[transition_cols].sum()
    digital_trans.columns = [
        c.replace("__transition", "_transitions_5m")
        for c in digital_trans.columns
    ]

    quality = pd.DataFrame(index=analog.index)
    quality["sample_count"] = g.size()
    quality["coverage_fraction"] = (
        quality["sample_count"] / P.expected_samples_per_bin
    )
    quality["bin_valid"] = (
        quality["coverage_fraction"] >= P.minimum_coverage_fraction
    )

    bins = pd.concat(
        [analog, digital_mean, digital_last, digital_trans, quality],
        axis=1,
    ).reset_index()

    feature_cols = [
        c for c in bins.columns
        if c not in {
            "bin_end",
            "sample_count",
            "coverage_fraction",
            "bin_valid",
        }
    ]

    # Any NaN statistic invalidates the bin.
    bins["bin_valid"] = (
        bins["bin_valid"]
        & bins[feature_cols].notna().all(axis=1)
    )

    print("\n5-minute bins")
    print("Total:", len(bins))
    print("Valid:", int(bins["bin_valid"].sum()))
    print("Rejected:", int((~bins["bin_valid"]).sum()))
    print("Per-bin feature count:", len(feature_cols))

    if len(feature_cols) != 59:
        raise AssertionError(
            f"Expected 59 per-bin features; got {len(feature_cols)}"
        )

    return bins


def build_consecutive_windows(
    bins: pd.DataFrame,
) -> tuple[np.ndarray, pd.DatetimeIndex, list[str]]:
    """
    Build 12-bin windows only when all bins are valid and endpoints are exactly
    5 minutes apart.

    Returned X is float32 with shape:
        n_windows x (12 * 59) = n_windows x 708
    """
    feature_cols = [
        c for c in bins.columns
        if c not in {
            "bin_end",
            "sample_count",
            "coverage_fraction",
            "bin_valid",
        }
    ]

    valid_bins = (
        bins.loc[bins["bin_valid"]]
        .sort_values("bin_end")
        .reset_index(drop=True)
    )

    delta = valid_bins["bin_end"].diff().dt.total_seconds()
    new_segment = (
        valid_bins.index.to_series().eq(0)
        | delta.ne(P.bin_minutes * 60)
    )

    valid_bins["segment_id"] = new_segment.cumsum().astype(int)
    valid_bins["segment_position"] = (
        valid_bins.groupby("segment_id").cumcount() + 1
    )

    candidate_end_indices = valid_bins.index[
        valid_bins["segment_position"] >= P.window_bins
    ].to_numpy()

    values = valid_bins[feature_cols].to_numpy(dtype=np.float32)

    n_features = len(feature_cols)
    window_width = P.window_bins * n_features

    X = np.empty(
        (len(candidate_end_indices), window_width),
        dtype=np.float32,
    )
    endpoints = []

    for row, end_idx in enumerate(candidate_end_indices):
        start_idx = end_idx - P.window_bins + 1
        X[row] = values[start_idx : end_idx + 1].reshape(-1)
        endpoints.append(valid_bins.at[end_idx, "bin_end"])

    flattened_names = []
    for lag in range(P.window_bins - 1, -1, -1):
        suffix = "current" if lag == 0 else f"lag_{lag * P.bin_minutes}m"
        flattened_names.extend(
            [f"{c}__{suffix}" for c in feature_cols]
        )

    endpoints = pd.DatetimeIndex(endpoints, name="timestamp")

    print("\n60-minute windows")
    print("Windows:", f"{len(X):,}")
    print("Flattened feature count:", X.shape[1])
    print("Expected:", 12 * 59)

    return X, endpoints, flattened_names


def split_windows(
    X: np.ndarray,
    endpoints: pd.DatetimeIndex,
):
    """
    60-minute purge means every retained window is fully contained within its
    corresponding fit/calibration/holdout raw period.
    """
    purge = pd.Timedelta(minutes=P.boundary_purge_minutes)

    fit_endpoint_start = FIT_RAW_START + purge
    fit_endpoint_end = FIT_RAW_END - purge

    cal_endpoint_start = CAL_RAW_START + purge
    cal_endpoint_end = CAL_RAW_END - purge

    hold_endpoint_start = HOLDOUT_RAW_START + purge

    fit_mask = (
        (endpoints >= fit_endpoint_start)
        & (endpoints <= fit_endpoint_end)
    )
    cal_mask = (
        (endpoints >= cal_endpoint_start)
        & (endpoints <= cal_endpoint_end)
    )
    hold_mask = endpoints >= hold_endpoint_start

    X_fit = X[fit_mask]
    X_cal = X[cal_mask]
    X_hold = X[hold_mask]

    t_fit = endpoints[fit_mask]
    t_cal = endpoints[cal_mask]
    t_hold = endpoints[hold_mask]

    print("\nSplit")
    print(
        f"Fit         : {len(X_fit):,} windows | "
        f"{t_fit.min()} -> {t_fit.max()}"
    )
    print(
        f"Calibration : {len(X_cal):,} windows | "
        f"{t_cal.min()} -> {t_cal.max()}"
    )
    print(
        f"Holdout     : {len(X_hold):,} windows | "
        f"{t_hold.min()} -> {t_hold.max()}"
    )

    return X_fit, t_fit, X_cal, t_cal, X_hold, t_hold


def robust_normalize(
    reference_scores: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    median = float(np.median(reference_scores))
    mad = float(np.median(np.abs(reference_scores - median)))

    if mad <= 1e-12:
        raise ValueError("Calibration MAD is effectively zero.")

    # 1.4826 makes MAD comparable to standard deviation for Gaussian data.
    scale = 1.4826 * mad

    z = (scores - median) / scale
    return z, median, mad


def segmented_ewma(
    timestamps: pd.DatetimeIndex,
    values: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Causal EWMA. State resets whenever 5-minute continuity breaks.
    """
    out = np.empty(len(values), dtype=float)
    prev_t = None
    prev_s = None

    for i, (t, value) in enumerate(zip(timestamps, values)):
        if (
            prev_t is None
            or (t - prev_t).total_seconds() != P.bin_minutes * 60
        ):
            smoothed = float(value)
        else:
            smoothed = alpha * float(value) + (1.0 - alpha) * prev_s

        out[i] = smoothed
        prev_t = t
        prev_s = smoothed

    return out


def three_of_four_persistence(
    timestamps: pd.DatetimeIndex,
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Alert state is True when >=3 of the most recent 4 consecutive 5-minute
    scores exceed threshold.
    """
    exceed = scores >= threshold
    alert = np.zeros(len(scores), dtype=bool)

    # Keep a short consecutive history. Reset at any missing 5-minute endpoint.
    history: list[bool] = []
    previous = None

    for i, (t, flag) in enumerate(zip(timestamps, exceed)):
        if (
            previous is None
            or (t - previous).total_seconds() != P.bin_minutes * 60
        ):
            history = []

        history.append(bool(flag))
        if len(history) > P.persistence_window:
            history.pop(0)

        if (
            len(history) == P.persistence_window
            and sum(history) >= P.persistence_required
        ):
            alert[i] = True

        previous = t

    return alert


def raw_alert_episodes(
    timestamps: pd.DatetimeIndex,
    alert_flags: np.ndarray,
) -> list[dict]:
    """
    Convert pointwise persistent-alert state into contiguous episodes.
    """
    episodes: list[dict] = []

    start = None
    end = None
    previous = None

    for t, flag in zip(timestamps, alert_flags):
        contiguous = (
            previous is not None
            and (t - previous).total_seconds() == P.bin_minutes * 60
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


def merge_episodes(
    episodes: list[dict],
) -> list[dict]:
    if not episodes:
        return []

    merged = [episodes[0].copy()]
    merge_gap = pd.Timedelta(minutes=P.episode_merge_minutes)

    for ep in episodes[1:]:
        current = merged[-1]

        if ep["start"] - current["end"] <= merge_gap:
            current["end"] = max(current["end"], ep["end"])
        else:
            merged.append(ep.copy())

    return merged


def apply_cooldown(
    episodes: list[dict],
) -> list[dict]:
    """
    Keep the first episode, then suppress new episode starts during the
    six-hour cooldown following the end of the previously kept episode.
    """
    if not episodes:
        return []

    kept = []
    cooldown = pd.Timedelta(hours=P.cooldown_hours)
    blocked_until = pd.Timestamp.min

    for ep in episodes:
        if ep["start"] < blocked_until:
            continue

        kept.append(ep.copy())
        blocked_until = ep["end"] + cooldown

    return kept


def episodes_to_frame(episodes: list[dict]) -> pd.DataFrame:
    if not episodes:
        return pd.DataFrame(columns=["episode_id", "start", "end", "duration_min"])

    rows = []
    for i, ep in enumerate(episodes, start=1):
        duration = (
            (ep["end"] - ep["start"]).total_seconds() / 60
            + P.bin_minutes
        )
        rows.append({
            "episode_id": i,
            "start": ep["start"],
            "end": ep["end"],
            "duration_min": duration,
        })

    return pd.DataFrame(rows)


def overlap(
    ep_start: pd.Timestamp,
    ep_end: pd.Timestamp,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> bool:
    return ep_end >= window_start and ep_start < window_end


def classify_event(
    event_id: str,
    episodes: pd.DataFrame,
) -> dict:
    start, end = EVENTS[event_id]

    early_start = start - pd.Timedelta(hours=P.early_window_start_hours)
    early_end = start - pd.Timedelta(hours=P.early_window_end_hours)

    late_start = early_end
    late_end = end

    early_hits = episodes[
        episodes.apply(
            lambda r: overlap(
                r["start"], r["end"], early_start, early_end
            ),
            axis=1,
        )
    ].copy()

    late_hits = episodes[
        episodes.apply(
            lambda r: overlap(
                r["start"], r["end"], late_start, late_end
            ),
            axis=1,
        )
    ].copy()

    early_detected = len(early_hits) > 0
    late_detected = len(late_hits) > 0

    first_early = None
    early_lead = None

    if early_detected:
        hit = early_hits.sort_values("start").iloc[0]
        first_early = max(hit["start"], early_start)
        early_lead = (start - first_early).total_seconds() / 60

    first_late = None
    relative_to_start = None

    if late_detected:
        hit = late_hits.sort_values("start").iloc[0]
        first_late = max(hit["start"], late_start)
        relative_to_start = (
            first_late - start
        ).total_seconds() / 60

    if early_detected:
        status = "EARLY"
    elif late_detected:
        status = "LATE"
    else:
        status = "MISSED"

    return {
        "event_id": event_id,
        "event_start": start,
        "event_end": end,
        "status": status,
        "early_detected": int(early_detected),
        "first_early_alert": first_early,
        "early_lead_minutes": early_lead,
        "late_detected": int(late_detected),
        "first_late_alert": first_late,
        # Negative = minutes before event; positive = minutes after start.
        "late_alert_relative_to_start_minutes": relative_to_start,
    }


def incident_related_mask(
    timestamps: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Incident-related exposure spans 24 h before each documented event through
    its recorded end. Everything else is normal evaluation exposure for the
    false-alarm calculation.
    """
    mask = np.zeros(len(timestamps), dtype=bool)

    for start, end in EVENTS.values():
        related_start = start - pd.Timedelta(
            hours=P.early_window_start_hours
        )
        mask |= (
            (timestamps >= related_start)
            & (timestamps <= end)
        )

    return mask


def episode_is_event_related(row: pd.Series) -> bool:
    for start, end in EVENTS.values():
        related_start = start - pd.Timedelta(
            hours=P.early_window_start_hours
        )
        if overlap(row["start"], row["end"], related_start, end):
            return True
    return False


def failure_labels(timestamps: pd.DatetimeIndex) -> np.ndarray:
    y = np.zeros(len(timestamps), dtype=int)

    for start, end in EVENTS.values():
        y |= (
            (timestamps >= start)
            & (timestamps <= end)
        ).astype(int)

    return y


def ranking_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
) -> dict:
    precision, recall, _ = precision_recall_curve(y_true, score)

    result = {
        "pr_auc": float(auc(recall, precision)),
        "average_precision": float(
            average_precision_score(y_true, score)
        ),
        "prevalence": float(np.mean(y_true)),
    }

    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, score))
    else:
        result["roc_auc"] = None

    return result


def build_policy_alert_flags(
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


def save_f04_plot(
    scored: pd.DataFrame,
    threshold: float,
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
        linewidth=1.4,
        label="EWMA anomaly score",
    )

    ax.axhline(
        threshold,
        linestyle="--",
        linewidth=1.2,
        label="Frozen q=0.995 calibration threshold",
    )

    ax.axvspan(
        start - pd.Timedelta(hours=24),
        start - pd.Timedelta(hours=2),
        alpha=0.12,
        label="Early window (24h to 2h before)",
    )

    ax.axvspan(
        start - pd.Timedelta(hours=2),
        start,
        alpha=0.10,
        label="Late pre-fault window",
    )

    ax.axvspan(
        start,
        end,
        alpha=0.10,
        label="Documented F04",
    )

    policy = view["policy_alert"].to_numpy(dtype=bool)
    if policy.any():
        ax.scatter(
            view.loc[policy, "timestamp"],
            view.loc[policy, "ewma_score"],
            s=16,
            label="Policy alert",
        )

    ax.set_title("F04 — Normal-Only Isolation Forest Early-Watch")
    ax.set_ylabel("Robust-normalized EWMA anomaly score")
    ax.set_xlabel("Time")
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_all_events_plot(
    scored: pd.DataFrame,
    threshold: float,
    out_path: Path,
) -> None:
    lo = min(x[0] for x in EVENTS.values()) - pd.Timedelta(hours=30)
    hi = max(x[1] for x in EVENTS.values()) + pd.Timedelta(hours=12)

    view = scored[
        (scored["timestamp"] >= lo)
        & (scored["timestamp"] <= hi)
    ].copy()

    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(
        view["timestamp"],
        view["ewma_score"],
        linewidth=0.8,
        label="EWMA anomaly score",
    )

    ax.axhline(
        threshold,
        linestyle="--",
        label="Frozen threshold",
    )

    for event_id, (start, end) in EVENTS.items():
        ax.axvspan(start, end, alpha=0.08)
        ax.text(
            start,
            ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else threshold,
            event_id,
            fontsize=8,
            rotation=90,
            va="top",
        )

    ax.set_title("March–September Holdout — Isolation Forest Anomaly Score")
    ax.set_ylabel("Robust-normalized EWMA score")
    ax.set_xlabel("Time")
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def jsonify(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat(sep=" ")
    if pd.isna(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    seed_everything()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "analysis": "AirGuard-LK Isolation Forest early-watch challenger",
                "status": "post-hoc model-family challenger",
                "failure_labels_used_for_fit": False,
                "failure_labels_used_for_threshold": False,
                "protocol": asdict(P),
                "fit_raw_period": [str(FIT_RAW_START), str(FIT_RAW_END)],
                "calibration_raw_period": [str(CAL_RAW_START), str(CAL_RAW_END)],
                "holdout_raw_start": str(HOLDOUT_RAW_START),
                "analog_signals": ANALOG,
                "digital_signals": DIGITAL,
            },
            f,
            indent=2,
        )

    raw = load_raw(args.raw)
    bins = build_causal_five_minute_bins(raw)

    # Free the 1.5M-row frame before allocating flattened windows.
    del raw

    X, endpoints, feature_names = build_consecutive_windows(bins)
    del bins

    (
        X_fit,
        t_fit,
        X_cal,
        t_cal,
        X_hold,
        t_hold,
    ) = split_windows(X, endpoints)

    # Main matrix no longer needed after split.
    del X

    print("\nTraining normal-only 300-tree Isolation Forest ...")

    model = IsolationForest(
        n_estimators=P.isolation_trees,
        max_samples=P.isolation_max_samples,
        max_features=P.isolation_max_features,
        contamination="auto",
        bootstrap=False,
        random_state=P.random_seed,
        n_jobs=-1,
    )

    model.fit(X_fit)

    # Higher means more anomalous.
    fit_raw_score = -model.score_samples(X_fit)
    cal_raw_score = -model.score_samples(X_cal)
    hold_raw_score = -model.score_samples(X_hold)

    # Calibration median/MAD defines the score scale.
    cal_z, cal_median, cal_mad = robust_normalize(
        cal_raw_score,
        cal_raw_score,
    )

    hold_z, _, _ = robust_normalize(
        cal_raw_score,
        hold_raw_score,
    )

    cal_ewma = segmented_ewma(
        t_cal,
        cal_z,
        P.ewma_alpha,
    )

    hold_ewma = segmented_ewma(
        t_hold,
        hold_z,
        P.ewma_alpha,
    )

    # Frozen entirely from February calibration.
    threshold = float(
        np.quantile(cal_ewma, P.calibration_quantile)
    )

    raw_persistent = three_of_four_persistence(
        t_hold,
        hold_ewma,
        threshold,
    )

    episodes = raw_alert_episodes(t_hold, raw_persistent)
    episodes = merge_episodes(episodes)
    episodes = apply_cooldown(episodes)
    episode_df = episodes_to_frame(episodes)

    policy_flags = build_policy_alert_flags(t_hold, episode_df)

    per_event = pd.DataFrame(
        [
            classify_event(event_id, episode_df)
            for event_id in EVENTS
        ]
    )

    early_event_count = int(per_event["early_detected"].sum())

    related = incident_related_mask(t_hold)
    normal_scored_points = ~related

    false_eps = episode_df[
        ~episode_df.apply(
            episode_is_event_related,
            axis=1,
        )
    ].copy()

    # Exposure based on actually scored 5-minute endpoints.
    normal_exposure_days = (
        normal_scored_points.sum()
        * P.bin_minutes
        / (60 * 24)
    )

    false_alarm_rate = (
        len(false_eps) / normal_exposure_days
        if normal_exposure_days > 0
        else None
    )

    failure_y = failure_labels(t_hold)
    rank = ranking_metrics(failure_y, hold_ewma)

    scored = pd.DataFrame({
        "timestamp": t_hold,
        "raw_anomaly_score": hold_raw_score,
        "robust_z_score": hold_z,
        "ewma_score": hold_ewma,
        "threshold": threshold,
        "raw_persistent_alert": raw_persistent.astype(int),
        "policy_alert": policy_flags.astype(int),
        "failure_active": failure_y,
        "incident_related_24h_to_end": related.astype(int),
    })

    calibration = pd.DataFrame({
        "timestamp": t_cal,
        "raw_anomaly_score": cal_raw_score,
        "robust_z_score": cal_z,
        "ewma_score": cal_ewma,
        "threshold": threshold,
    })

    persistent_time_pct = 100.0 * float(raw_persistent.mean())
    policy_time_pct = 100.0 * float(policy_flags.mean())

    result = {
        "analysis": "AirGuard-LK normal-only Isolation Forest early-watch challenger",
        "status": "post-hoc model-family challenger",
        "goal": "Test whether normal-behaviour anomaly detection can overcome the frozen LightGBM +150 minute F04 delay.",
        "failure_labels_used_for_model_fit": False,
        "failure_labels_used_for_threshold_calibration": False,
        "fit_period": {
            "raw_start": str(FIT_RAW_START),
            "raw_end_exclusive": str(FIT_RAW_END),
            "window_end_start": str(t_fit.min()),
            "window_end_end": str(t_fit.max()),
            "window_count": len(X_fit),
        },
        "calibration_period": {
            "raw_start": str(CAL_RAW_START),
            "raw_end_exclusive": str(CAL_RAW_END),
            "window_end_start": str(t_cal.min()),
            "window_end_end": str(t_cal.max()),
            "window_count": len(X_cal),
        },
        "holdout_period": {
            "window_end_start": str(t_hold.min()),
            "window_end_end": str(t_hold.max()),
            "window_count": len(X_hold),
        },
        "feature_design": {
            "per_bin_features": 59,
            "window_bins": P.window_bins,
            "window_minutes": P.window_bins * P.bin_minutes,
            "flattened_features": len(feature_names),
        },
        "model": {
            "type": "IsolationForest",
            "n_estimators": P.isolation_trees,
            "max_samples": P.isolation_max_samples,
            "max_features": P.isolation_max_features,
            "random_state": P.random_seed,
        },
        "score_calibration": {
            "calibration_raw_score_median": cal_median,
            "calibration_raw_score_mad": cal_mad,
            "ewma_alpha": P.ewma_alpha,
            "threshold_quantile": P.calibration_quantile,
            "frozen_threshold": threshold,
        },
        "alert_policy": {
            "persistence": f"{P.persistence_required}-of-{P.persistence_window}",
            "episode_merge_minutes": P.episode_merge_minutes,
            "cooldown_hours": P.cooldown_hours,
            "early_window": (
                f"{P.early_window_start_hours}h to "
                f"{P.early_window_end_hours}h before event start"
            ),
        },
        "event_results": [
            {
                key: jsonify(value)
                for key, value in row.items()
            }
            for row in per_event.to_dict(orient="records")
        ],
        "summary": {
            "early_event_recall": f"{early_event_count}/4",
            "false_alarm_episodes": len(false_eps),
            "normal_exposure_days": normal_exposure_days,
            "false_alarm_episodes_per_day": false_alarm_rate,
            "persistent_time_in_alert_pct": persistent_time_pct,
            "policy_time_in_alert_pct": policy_time_pct,
            "failure_window_pr_auc": rank["pr_auc"],
            "failure_window_average_precision": rank["average_precision"],
            "failure_window_roc_auc": rank["roc_auc"],
            "failure_prevalence": rank["prevalence"],
        },
    }

    # Save before plotting so core results survive any rendering problem.
    with open(args.out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    per_event.to_csv(
        args.out_dir / "per_event_results.csv",
        index=False,
    )

    scored.to_csv(
        args.out_dir / "scored_holdout.csv",
        index=False,
    )

    calibration.to_csv(
        args.out_dir / "calibration_scores.csv",
        index=False,
    )

    episode_df.to_csv(
        args.out_dir / "alert_episodes.csv",
        index=False,
    )

    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "calibration_score_median": cal_median,
            "calibration_score_mad": cal_mad,
            "threshold": threshold,
            "protocol": asdict(P),
        },
        args.out_dir / "isolation_forest.joblib",
    )

    save_f04_plot(
        scored,
        threshold,
        args.out_dir / "f04_early_watch_timeline.png",
    )

    save_all_events_plot(
        scored,
        threshold,
        args.out_dir / "all_events_timeline.png",
    )

    print("\n" + "=" * 80)
    print("EVENT RESULTS")
    print("=" * 80)
    print(per_event.to_string(index=False))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Early event recall       :", f"{early_event_count}/4")
    print("False alert episodes     :", len(false_eps))
    print(
        "False alerts/day         :",
        "n/a" if false_alarm_rate is None else f"{false_alarm_rate:.4f}",
    )
    print(
        "Persistent time alert %  :",
        f"{persistent_time_pct:.2f}",
    )
    print(
        "Policy time alert %      :",
        f"{policy_time_pct:.2f}",
    )
    print(
        "Failure-window PR-AUC    :",
        f"{rank['pr_auc']:.4f}",
    )
    print(
        "Failure-window AP        :",
        f"{rank['average_precision']:.4f}",
    )

    f04 = per_event.loc[per_event["event_id"] == "F04"].iloc[0]
    print("\n" + "=" * 80)
    print("F04 — DID WE BEAT THE +150 MIN LIGHTGBM LIMITATION?")
    print("=" * 80)

    if f04["status"] == "EARLY":
        print(
            "YES — persistent anomaly alert appeared",
            f"{f04['early_lead_minutes']:.1f} minutes BEFORE F04.",
        )
    elif f04["status"] == "LATE":
        rel = f04["late_alert_relative_to_start_minutes"]

        if rel is not None and rel < 0:
            print(
                "PARTIAL — alert appeared",
                f"{abs(rel):.1f} minutes BEFORE F04,",
                "but inside the final 2-hour late-warning window.",
            )
        elif rel is not None:
            print(
                "LATE — alert appeared",
                f"{rel:.1f} minutes AFTER F04 start.",
            )
            if rel < 150:
                print(
                    f"This is {150 - rel:.1f} minutes earlier than "
                    "the frozen LightGBM persistent alert."
                )
            else:
                print(
                    "This did not improve on the frozen LightGBM +150 min alert."
                )
    else:
        print("MISSED — no policy alert in the 24h-pre-event through fault interval.")

    print("\nArtifacts saved to:", args.out_dir.resolve())


if __name__ == "__main__":
    main()
