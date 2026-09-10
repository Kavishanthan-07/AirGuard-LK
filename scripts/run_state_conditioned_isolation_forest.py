
"""
AirGuard-LK — State-Conditioned Normal-Only Isolation Forest
============================================================

Purpose
-------
Test one fixed hypothesis raised by the global Isolation Forest result:

    A single normal model may treat legitimate compressor operating-state
    changes as anomalies, causing excessive alert occupancy.

This challenger therefore learns separate normal-behaviour Isolation Forests
for empirically observed operating regimes defined only by the CURRENT values
of:

    COMP, MPG, DV_eletric

No physical ON/OFF meaning is assigned to these combinations.

Fixed protocol
--------------
Raw MetroPT-3
    -> causal right-closed 5-minute bins
    -> analogue: mean/std/min/max/last
    -> digital: active ratio/last/transitions
    -> reject bins with <80% expected sample coverage
    -> 12 consecutive bins = 60-minute causal window
    -> 708 flattened features
    -> regime = (COMP_last, MPG_last, DV_eletric_last) of current bin
    -> fit regime-specific Isolation Forests on 1-21 Feb NORMAL data only
    -> calibrate on 22-29 Feb only
    -> March-September evaluation
    -> regime-specific calibration median/MAD normalization
    -> pooled February calibration q=0.9995 threshold
    -> causal EWMA alpha=0.2, RESET on regime change or time gap
    -> 3-of-4 persistence, RESET on regime change or time gap
    -> merge episodes separated by <=30 min
    -> 6-hour cooldown
    -> strict early = a NEW alert episode starts 24 h to 2 h before event

Why q=0.9995 and 3-of-4?
------------------------
Those were selected in the prior nuisance-control sensitivity using F01-F03.
This experiment changes only the normal-behaviour modelling strategy.

Rare regimes
------------
A regime receives its own model only if it has at least:
    250 fit windows AND 50 calibration windows.

All other / unseen combinations are mapped to OTHER and scored by an OTHER
normal model when sufficient fit data exist. If OTHER is too small, a global
normal fallback model is used.

Important scientific status
---------------------------
This is a POST-HOC model-family challenger. F04 has already been examined.
F04 is therefore reported descriptively, not as a pristine untouched holdout.

Outputs
-------
artifacts/state_conditioned_isolation_forest/
    result.json
    regime_summary.csv
    per_event_results.csv
    alert_episodes.csv
    scored_holdout.csv
    calibration_scores.csv
    nuisance_comparison.csv
    state_conditioned_iforest.joblib
    f04_timeline.png
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


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

REGIME_DIGITALS = ["COMP", "MPG", "DV_eletric"]

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
    regime_min_fit_windows: int = 250
    regime_min_cal_windows: int = 50
    calibration_quantile: float = 0.9995
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
    p = argparse.ArgumentParser()
    p.add_argument(
        "--raw",
        type=Path,
        default=Path("data/raw/metropt3/MetroPT3(AirCompressor).csv"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/state_conditioned_isolation_forest"),
    )
    p.add_argument(
        "--global-result",
        type=Path,
        default=Path("artifacts/isolation_forest_nuisance_sensitivity/"
                     "development_selected_candidate.json"),
    )
    return p.parse_args()


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)


def load_raw(path: Path) -> pd.DataFrame:
    print(f"Loading {path} ...")
    required = ["timestamp", *ANALOG, *DIGITAL]

    df = pd.read_csv(
        path,
        usecols=lambda c: c in required,
        low_memory=False,
    )

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = (
        df.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="first")
        .reset_index(drop=True)
    )

    if df[ANALOG + DIGITAL].isna().any().any():
        raise ValueError(
            "Unexpected missing values. No interpolation/backfill is permitted."
        )

    print(f"Rows: {len(df):,}")
    print("Range:", df["timestamp"].min(), "->", df["timestamp"].max())
    return df


def build_bins(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["bin_end"] = df["timestamp"].dt.ceil(f"{P.bin_minutes}min")

    grouped = df.groupby("bin_end", sort=False)

    transition_cols = []
    for col in DIGITAL:
        name = f"{col}__transition"
        df[name] = (
            grouped[col].diff().fillna(0).ne(0).astype(np.int8)
        )
        transition_cols.append(name)

    g = df.groupby("bin_end", sort=True)

    analog = g[ANALOG].agg(["mean", "std", "min", "max", "last"])
    analog.columns = [
        f"{sensor}_{stat}_5m"
        for sensor, stat in analog.columns
    ]

    digital_mean = g[DIGITAL].mean()
    digital_mean.columns = [
        f"{c}_active_ratio_5m" for c in DIGITAL
    ]

    digital_last = g[DIGITAL].last()
    digital_last.columns = [
        f"{c}_last_5m" for c in DIGITAL
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

    bins = pd.concat(
        [analog, digital_mean, digital_last, digital_trans, quality],
        axis=1,
    ).reset_index()

    feature_cols = [
        c for c in bins.columns
        if c not in {"bin_end", "sample_count", "coverage_fraction"}
    ]

    bins["bin_valid"] = (
        (bins["coverage_fraction"] >= P.minimum_coverage_fraction)
        & bins[feature_cols].notna().all(axis=1)
    )

    if len(feature_cols) != 59:
        raise AssertionError(
            f"Expected 59 per-bin model features, found {len(feature_cols)}."
        )

    print("\n5-minute bins")
    print("Total:", f"{len(bins):,}")
    print("Valid:", f"{int(bins['bin_valid'].sum()):,}")
    print("Rejected:", f"{int((~bins['bin_valid']).sum()):,}")
    print("Features/bin:", len(feature_cols))

    return bins


def raw_regime_label(row: pd.Series) -> str:
    vals = []
    for c in REGIME_DIGITALS:
        value = int(round(float(row[f"{c}_last_5m"])))
        vals.append(value)
    return f"C{vals[0]}_M{vals[1]}_D{vals[2]}"


def build_windows(
    bins: pd.DataFrame,
) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray, list[str]]:
    exclude = {
        "bin_end",
        "sample_count",
        "coverage_fraction",
        "bin_valid",
    }
    feature_cols = [c for c in bins.columns if c not in exclude]

    valid = (
        bins.loc[bins["bin_valid"]]
        .sort_values("bin_end")
        .reset_index(drop=True)
    )

    delta = valid["bin_end"].diff().dt.total_seconds()
    new_segment = (
        valid.index.to_series().eq(0)
        | delta.ne(P.bin_minutes * 60)
    )

    valid["segment_id"] = new_segment.cumsum().astype(int)
    valid["segment_position"] = (
        valid.groupby("segment_id").cumcount() + 1
    )

    ends = valid.index[
        valid["segment_position"] >= P.window_bins
    ].to_numpy()

    values = valid[feature_cols].to_numpy(dtype=np.float32)

    X = np.empty(
        (len(ends), P.window_bins * len(feature_cols)),
        dtype=np.float32,
    )
    timestamps = []
    regimes = []

    for i, end in enumerate(ends):
        start = end - P.window_bins + 1
        X[i] = values[start : end + 1].reshape(-1)
        timestamps.append(valid.at[end, "bin_end"])
        regimes.append(raw_regime_label(valid.loc[end]))

    flattened_names = []
    for lag in range(P.window_bins - 1, -1, -1):
        suffix = "current" if lag == 0 else f"lag_{lag * P.bin_minutes}m"
        flattened_names.extend(
            [f"{c}__{suffix}" for c in feature_cols]
        )

    timestamps = pd.DatetimeIndex(timestamps, name="timestamp")
    regimes = np.asarray(regimes, dtype=object)

    print("\n60-minute windows")
    print("Count:", f"{len(X):,}")
    print("Features:", X.shape[1])
    print("Raw regime combinations:", sorted(set(regimes)))

    return X, timestamps, regimes, flattened_names


def split_windows(X, timestamps, regimes):
    purge = pd.Timedelta(minutes=P.boundary_purge_minutes)

    fit_mask = (
        (timestamps >= FIT_RAW_START + purge)
        & (timestamps <= FIT_RAW_END - purge)
    )

    cal_mask = (
        (timestamps >= CAL_RAW_START + purge)
        & (timestamps <= CAL_RAW_END - purge)
    )

    hold_mask = timestamps >= HOLDOUT_RAW_START + purge

    return (
        X[fit_mask],
        timestamps[fit_mask],
        regimes[fit_mask],
        X[cal_mask],
        timestamps[cal_mask],
        regimes[cal_mask],
        X[hold_mask],
        timestamps[hold_mask],
        regimes[hold_mask],
    )


def create_regime_mapping(
    fit_regimes: np.ndarray,
    cal_regimes: np.ndarray,
) -> tuple[dict[str, str], pd.DataFrame]:
    fit_counts = pd.Series(fit_regimes).value_counts()
    cal_counts = pd.Series(cal_regimes).value_counts()

    all_regimes = sorted(set(fit_counts.index) | set(cal_counts.index))
    rows = []
    mapping = {}

    for regime in all_regimes:
        n_fit = int(fit_counts.get(regime, 0))
        n_cal = int(cal_counts.get(regime, 0))

        dedicated = (
            n_fit >= P.regime_min_fit_windows
            and n_cal >= P.regime_min_cal_windows
        )

        mapped = regime if dedicated else "OTHER"
        mapping[regime] = mapped

        rows.append({
            "raw_regime": regime,
            "fit_windows": n_fit,
            "calibration_windows": n_cal,
            "dedicated_model": int(dedicated),
            "mapped_regime": mapped,
        })

    return mapping, pd.DataFrame(rows)


def map_regimes(
    regimes: np.ndarray,
    mapping: dict[str, str],
) -> np.ndarray:
    return np.asarray(
        [mapping.get(str(r), "OTHER") for r in regimes],
        dtype=object,
    )


def fit_models(
    X_fit: np.ndarray,
    mapped_fit: np.ndarray,
) -> tuple[dict[str, IsolationForest], IsolationForest]:
    print("\nTraining global fallback model ...")

    global_model = IsolationForest(
        n_estimators=P.isolation_trees,
        max_samples="auto",
        max_features=1.0,
        contamination="auto",
        bootstrap=False,
        random_state=P.random_seed,
        n_jobs=-1,
    )
    global_model.fit(X_fit)

    models = {}
    unique = sorted(set(mapped_fit))

    for i, regime in enumerate(unique):
        mask = mapped_fit == regime
        n = int(mask.sum())

        if regime == "OTHER" and n < P.regime_min_fit_windows:
            print(
                f"Regime OTHER: {n} fit windows -> using global fallback."
            )
            continue

        print(f"Training regime {regime}: {n:,} windows")

        model = IsolationForest(
            n_estimators=P.isolation_trees,
            max_samples="auto",
            max_features=1.0,
            contamination="auto",
            bootstrap=False,
            random_state=P.random_seed + i + 1,
            n_jobs=-1,
        )
        model.fit(X_fit[mask])
        models[regime] = model

    return models, global_model


def score_by_regime(
    X: np.ndarray,
    mapped_regimes: np.ndarray,
    models: dict[str, IsolationForest],
    global_model: IsolationForest,
) -> np.ndarray:
    scores = np.empty(len(X), dtype=float)

    for regime in sorted(set(mapped_regimes)):
        mask = mapped_regimes == regime
        model = models.get(regime, global_model)
        scores[mask] = -model.score_samples(X[mask])

    return scores


def calibration_stats(
    scores: np.ndarray,
    mapped_regimes: np.ndarray,
) -> dict[str, dict[str, float]]:
    stats = {}

    global_median = float(np.median(scores))
    global_mad = float(np.median(np.abs(scores - global_median)))

    if global_mad <= 1e-12:
        raise ValueError("Global calibration MAD is effectively zero.")

    stats["__GLOBAL__"] = {
        "median": global_median,
        "mad": global_mad,
    }

    for regime in sorted(set(mapped_regimes)):
        s = scores[mapped_regimes == regime]

        if len(s) < 10:
            continue

        median = float(np.median(s))
        mad = float(np.median(np.abs(s - median)))

        if mad <= 1e-12:
            continue

        stats[regime] = {
            "median": median,
            "mad": mad,
        }

    return stats


def normalize_by_regime(
    scores: np.ndarray,
    mapped_regimes: np.ndarray,
    stats: dict[str, dict[str, float]],
) -> np.ndarray:
    out = np.empty(len(scores), dtype=float)

    fallback = stats["__GLOBAL__"]

    for i, (score, regime) in enumerate(zip(scores, mapped_regimes)):
        s = stats.get(str(regime), fallback)
        scale = 1.4826 * s["mad"]
        out[i] = (float(score) - s["median"]) / scale

    return out


def state_reset_ewma(
    timestamps: pd.DatetimeIndex,
    regimes: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    out = np.empty(len(values), dtype=float)

    previous_t = None
    previous_regime = None
    previous_smoothed = None

    for i, (t, regime, value) in enumerate(
        zip(timestamps, regimes, values)
    ):
        reset = (
            previous_t is None
            or (t - previous_t).total_seconds() != P.bin_minutes * 60
            or regime != previous_regime
        )

        if reset:
            smoothed = float(value)
        else:
            smoothed = (
                P.ewma_alpha * float(value)
                + (1.0 - P.ewma_alpha) * previous_smoothed
            )

        out[i] = smoothed
        previous_t = t
        previous_regime = regime
        previous_smoothed = smoothed

    return out


def state_reset_persistence(
    timestamps: pd.DatetimeIndex,
    regimes: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    exceed = scores >= threshold
    alert = np.zeros(len(scores), dtype=bool)

    history = []
    previous_t = None
    previous_regime = None

    for i, (t, regime, flag) in enumerate(
        zip(timestamps, regimes, exceed)
    ):
        reset = (
            previous_t is None
            or (t - previous_t).total_seconds() != P.bin_minutes * 60
            or regime != previous_regime
        )

        if reset:
            history = []

        history.append(bool(flag))
        if len(history) > P.persistence_window:
            history.pop(0)

        if (
            len(history) == P.persistence_window
            and sum(history) >= P.persistence_required
        ):
            alert[i] = True

        previous_t = t
        previous_regime = regime

    return alert


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
            and (t - previous).total_seconds() == P.bin_minutes * 60
        )

        if flag:
            if start is None or not contiguous:
                if start is not None:
                    episodes.append({"start": start, "end": end})
                start = t
            end = t
        elif start is not None:
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

    gap = pd.Timedelta(minutes=P.episode_merge_minutes)
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

    cooldown = pd.Timedelta(hours=P.cooldown_hours)
    kept = []
    blocked_until = pd.Timestamp.min

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
                + P.bin_minutes
            ),
        })

    return pd.DataFrame(
        rows,
        columns=["episode_id", "start", "end", "duration_min"],
    )


def overlap(a_start, a_end, b_start, b_end) -> bool:
    return a_end >= b_start and a_start < b_end


def event_metrics(event_id: str, episodes: pd.DataFrame) -> dict:
    start, end = EVENTS[event_id]

    early_start = start - pd.Timedelta(
        hours=P.early_window_start_hours
    )
    early_end = start - pd.Timedelta(
        hours=P.early_window_end_hours
    )

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

    first_early = strict.iloc[0]["start"] if strict_detected else pd.NaT
    early_lead = (
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
        "first_strict_early_alert": first_early,
        "strict_early_lead_minutes": early_lead,
        "late_detected": int(late_detected),
        "first_late_alert": first_late,
        "late_alert_relative_to_start_minutes": late_relative,
    }


def point_related_mask(
    timestamps: pd.DatetimeIndex,
    event_ids: list[str],
) -> np.ndarray:
    mask = np.zeros(len(timestamps), dtype=bool)

    for event_id in event_ids:
        start, end = EVENTS[event_id]
        related_start = start - pd.Timedelta(
            hours=P.early_window_start_hours
        )
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
        related_start = start - pd.Timedelta(
            hours=P.early_window_start_hours
        )
        if overlap(row["start"], row["end"], related_start, end):
            return True
    return False


def episode_policy_flags(
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
    policy_flags: np.ndarray,
    end_time: pd.Timestamp | None = None,
    event_ids: list[str] | None = None,
) -> dict:
    if event_ids is None:
        event_ids = list(EVENTS.keys())

    if end_time is None:
        point_scope = np.ones(len(timestamps), dtype=bool)
        scoped_eps = episodes.copy()
    else:
        point_scope = timestamps < end_time
        scoped_eps = episodes[episodes["start"] < end_time].copy()

    ts_scope = timestamps[point_scope]
    policy_scope = policy_flags[point_scope]

    related = point_related_mask(ts_scope, event_ids)
    normal = ~related

    false_eps = scoped_eps[
        ~scoped_eps.apply(
            lambda r: episode_related_to_events(r, event_ids),
            axis=1,
        )
    ]

    normal_days = (
        normal.sum() * P.bin_minutes / (60 * 24)
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
        "policy_time_in_alert_pct": 100.0 * float(policy_scope.mean()),
    }


def maybe_load_global_baseline(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        candidate = data.get("candidate", {})
        return {
            "global_q": candidate.get("threshold_quantile"),
            "global_persistence": candidate.get("persistence_label"),
            "global_dev_false_alerts_per_day": candidate.get(
                "false_alert_episodes_per_day_dev"
            ),
            "global_dev_policy_time_pct": candidate.get(
                "policy_time_in_alert_pct_dev"
            ),
            "global_strict_early_dev": candidate.get(
                "strict_new_early_dev_F01_F03"
            ),
        }
    except Exception:
        return {}


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
    ]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        view["timestamp"],
        view["ewma_score"],
        linewidth=1.2,
        label="State-conditioned EWMA anomaly score",
    )

    ax.axhline(
        threshold,
        linestyle="--",
        label="Frozen q=0.9995 threshold",
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

    alert = view["policy_alert"].astype(bool)
    if alert.any():
        ax.scatter(
            view.loc[alert, "timestamp"],
            view.loc[alert, "ewma_score"],
            s=16,
            label="Policy alert",
        )

    ax.set_title("F04 — State-Conditioned Normal-Only Isolation Forest")
    ax.set_ylabel("Robust normalized anomaly score")
    ax.set_xlabel("Time")
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def json_value(v):
    if isinstance(v, pd.Timestamp):
        return v.isoformat(sep=" ")
    if isinstance(v, np.generic):
        v = v.item()
    if pd.isna(v):
        return None
    return v


def main() -> None:
    args = parse_args()
    seed_everything()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(args.raw)
    bins = build_bins(raw)
    del raw

    X, timestamps, raw_regimes, feature_names = build_windows(bins)
    del bins

    (
        X_fit,
        t_fit,
        r_fit_raw,
        X_cal,
        t_cal,
        r_cal_raw,
        X_hold,
        t_hold,
        r_hold_raw,
    ) = split_windows(X, timestamps, raw_regimes)

    del X

    mapping, regime_summary = create_regime_mapping(
        r_fit_raw,
        r_cal_raw,
    )

    r_fit = map_regimes(r_fit_raw, mapping)
    r_cal = map_regimes(r_cal_raw, mapping)
    r_hold = map_regimes(r_hold_raw, mapping)

    print("\nRegime mapping")
    print(regime_summary.to_string(index=False))

    models, global_model = fit_models(X_fit, r_fit)

    cal_raw_scores = score_by_regime(
        X_cal,
        r_cal,
        models,
        global_model,
    )

    hold_raw_scores = score_by_regime(
        X_hold,
        r_hold,
        models,
        global_model,
    )

    stats = calibration_stats(cal_raw_scores, r_cal)

    cal_z = normalize_by_regime(
        cal_raw_scores,
        r_cal,
        stats,
    )
    hold_z = normalize_by_regime(
        hold_raw_scores,
        r_hold,
        stats,
    )

    cal_ewma = state_reset_ewma(
        t_cal,
        r_cal,
        cal_z,
    )
    hold_ewma = state_reset_ewma(
        t_hold,
        r_hold,
        hold_z,
    )

    # Fixed from prior F01-F03 nuisance study.
    threshold = float(
        np.quantile(cal_ewma, P.calibration_quantile)
    )

    persistent = state_reset_persistence(
        t_hold,
        r_hold,
        hold_ewma,
        threshold,
    )

    episodes = raw_episodes(t_hold, persistent)
    episodes = merge_episodes(episodes)
    episodes = apply_cooldown(episodes)
    episodes_df = episodes_frame(episodes)

    policy = episode_policy_flags(t_hold, episodes_df)

    per_event = pd.DataFrame(
        [event_metrics(event_id, episodes_df) for event_id in EVENTS]
    )

    # Development nuisance excludes F04 24h early window and later.
    f04_dev_end = EVENTS["F04"][0] - pd.Timedelta(
        hours=P.early_window_start_hours
    )

    dev_nuisance = nuisance_metrics(
        timestamps=t_hold,
        episodes=episodes_df,
        policy_flags=policy,
        end_time=f04_dev_end,
        event_ids=["F01", "F02", "F03"],
    )

    all_nuisance = nuisance_metrics(
        timestamps=t_hold,
        episodes=episodes_df,
        policy_flags=policy,
        event_ids=["F01", "F02", "F03", "F04"],
    )

    scored = pd.DataFrame({
        "timestamp": t_hold,
        "raw_regime": r_hold_raw,
        "mapped_regime": r_hold,
        "raw_anomaly_score": hold_raw_scores,
        "regime_normalized_score": hold_z,
        "ewma_score": hold_ewma,
        "threshold": threshold,
        "persistent_alert": persistent.astype(int),
        "policy_alert": policy.astype(int),
    })

    calibration = pd.DataFrame({
        "timestamp": t_cal,
        "raw_regime": r_cal_raw,
        "mapped_regime": r_cal,
        "raw_anomaly_score": cal_raw_scores,
        "regime_normalized_score": cal_z,
        "ewma_score": cal_ewma,
        "threshold": threshold,
    })

    global_baseline = maybe_load_global_baseline(args.global_result)

    f04 = per_event.loc[
        per_event["event_id"] == "F04"
    ].iloc[0]

    result = {
        "analysis": "AirGuard-LK state-conditioned normal-only Isolation Forest",
        "status": "post-hoc model-family challenger",
        "goal": (
            "Preserve early F04 anomaly detection while reducing nuisance "
            "alerts caused by legitimate operating-state variation."
        ),
        "failure_labels_used_for_fit": False,
        "failure_labels_used_for_threshold": False,
        "regime_definition": REGIME_DIGITALS,
        "regime_semantics_assigned": False,
        "protocol": asdict(P),
        "regime_mapping": mapping,
        "regime_models_trained": sorted(models.keys()),
        "calibration_threshold": threshold,
        "event_results": [
            {k: json_value(v) for k, v in row.items()}
            for row in per_event.to_dict(orient="records")
        ],
        "development_F01_F03": {
            "strict_new_early_events": (
                f"{int(per_event.loc[
                    per_event['event_id'].isin(['F01','F02','F03']),
                    'strict_new_early_detected'
                ].sum())}/3"
            ),
            **dev_nuisance,
        },
        "all_F01_F04": {
            "strict_new_early_events": (
                f"{int(per_event['strict_new_early_detected'].sum())}/4"
            ),
            **all_nuisance,
        },
        "F04": {
            k: json_value(v)
            for k, v in f04.to_dict().items()
        },
        "previous_global_baseline_if_available": global_baseline,
    }

    # Comparison row against the previous global IF candidate if available.
    comparison_rows = [{
        "model": "state_conditioned_iforest",
        "dev_strict_early_events": result["development_F01_F03"][
            "strict_new_early_events"
        ],
        "dev_false_alerts_per_day": dev_nuisance[
            "false_alert_episodes_per_day"
        ],
        "dev_policy_time_pct": dev_nuisance[
            "policy_time_in_alert_pct"
        ],
        "F04_status": f04["status"],
        "F04_lead_minutes": f04["strict_early_lead_minutes"],
    }]

    if global_baseline:
        comparison_rows.append({
            "model": "previous_global_iforest",
            "dev_strict_early_events": (
                f"{int(global_baseline.get('global_strict_early_dev', 0))}/3"
            ),
            "dev_false_alerts_per_day": global_baseline.get(
                "global_dev_false_alerts_per_day"
            ),
            "dev_policy_time_pct": global_baseline.get(
                "global_dev_policy_time_pct"
            ),
            "F04_status": "STRICT_EARLY",
            "F04_lead_minutes": 945.0,
        })

    comparison = pd.DataFrame(comparison_rows)

    with open(
        args.out_dir / "result.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2)

    regime_summary.to_csv(
        args.out_dir / "regime_summary.csv",
        index=False,
    )
    per_event.to_csv(
        args.out_dir / "per_event_results.csv",
        index=False,
    )
    episodes_df.to_csv(
        args.out_dir / "alert_episodes.csv",
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
    comparison.to_csv(
        args.out_dir / "nuisance_comparison.csv",
        index=False,
    )

    joblib.dump(
        {
            "regime_models": models,
            "global_fallback_model": global_model,
            "regime_mapping": mapping,
            "calibration_stats": stats,
            "threshold": threshold,
            "protocol": asdict(P),
            "feature_names": feature_names,
        },
        args.out_dir / "state_conditioned_iforest.joblib",
    )

    save_f04_plot(
        scored,
        threshold,
        args.out_dir / "f04_timeline.png",
    )

    print("\n" + "=" * 100)
    print("EVENT RESULTS")
    print("=" * 100)
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

    print("\n" + "=" * 100)
    print("DEVELOPMENT F01-F03")
    print("=" * 100)
    print(
        "Strict early:",
        result["development_F01_F03"]["strict_new_early_events"],
    )
    print(
        "False alerts/day:",
        f"{dev_nuisance['false_alert_episodes_per_day']:.4f}",
    )
    print(
        "Policy time in alert:",
        f"{dev_nuisance['policy_time_in_alert_pct']:.2f}%",
    )

    print("\n" + "=" * 100)
    print("F04")
    print("=" * 100)
    print("Status:", f04["status"])

    if int(f04["strict_new_early_detected"]) == 1:
        lead = float(f04["strict_early_lead_minutes"])
        print(f"Strict new early lead: {lead:.1f} min ({lead/60:.2f} h)")
        print("First early alert:", f04["first_strict_early_alert"])
        print("This beats the frozen +150 min LightGBM timing.")
    elif int(f04["late_detected"]) == 1:
        rel = float(f04["late_alert_relative_to_start_minutes"])
        if rel < 0:
            print(f"Alert: {abs(rel):.1f} min before F04, inside final 2 h.")
        else:
            print(f"Alert: {rel:.1f} min after F04.")
            if rel < 150:
                print(f"Improvement over LightGBM: {150-rel:.1f} min.")
    else:
        print("No useful F04 alert.")

    print("\n" + "=" * 100)
    print("NUISANCE COMPARISON")
    print("=" * 100)
    print(comparison.to_string(index=False))

    print("\nArtifacts saved to:", args.out_dir.resolve())


if __name__ == "__main__":
    main()
