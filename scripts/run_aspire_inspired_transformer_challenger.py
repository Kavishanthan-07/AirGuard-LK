"""
AirGuard-LK — ASPIRE-inspired raw-sequence Transformer challenger

Purpose
-------
Test one specific hypothesis:

    Did the original 1-minute aggregation remove useful high-frequency
    precursor information?

This is NOT an exact reproduction of ASPIRE. It deliberately borrows the
paper's key representation choices while retaining AirGuard-LK's stricter
operational reporting:

- MetroPT-3 raw ~10 s observations
- data from 2020-04-01 onward
- 12 ASPIRE-selected channels
- 100-observation causal windows (~16.7 min)
- Normal / Warning / Fault labels
- Warning horizon = 2 h
- two Transformer encoder blocks
- d_model=64, 4 heads, FF=128, dropout=0.2
- chronological training before F03
- F03 = development event
- F04 = post-hoc exploratory event
- 3-minute persistent warning metric
- false-warning episodes/day

Important
---------
F04 is NOT treated as a pristine new holdout here. AirGuard-LK has already
examined F04 and relevant literature has now been reviewed. The F04 output
from this challenger is explicitly post-hoc/exploratory.

Reference:
Guha, S. & Datta, A. (2026), ASPIRE: An Agentic Decision System for Early
Equipment Failure Prediction and Intervention in Industrial IIoT,
IEEE Access, 14, 105006–105031.
DOI: 10.1109/ACCESS.2026.3711010
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


SEED = 42

SENSORS = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
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

WARNING_HOURS = 2
WINDOW = 100
MAX_GAP_SECONDS = 30
DEVELOPMENT_NORMAL_HOURS = 48
PERSISTENCE_MINUTES = 3

# Predeclared development threshold grid.
WARNING_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
        default=Path("artifacts/aspire_inspired_transformer"),
    )
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)

    # Compute-conscious training:
    # keep every warning/fault window but sample normal windows.
    p.add_argument("--train-normal-stride", type=int, default=10)
    p.add_argument("--train-event-stride", type=int, default=1)

    # Evaluate every valid raw window endpoint.
    p.add_argument("--eval-stride", type=int, default=1)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    return p.parse_args()


def load_raw(path: Path) -> pd.DataFrame:
    print(f"Loading {path} ...")
    df = pd.read_csv(path, low_memory=False)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    required = ["timestamp", *SENSORS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # ASPIRE excludes data before 1 April 2020.
    df = df[df["timestamp"] >= pd.Timestamp("2020-04-01")].reset_index(drop=True)

    # MetroPT-3 used here has no missing values in these channels, but retain
    # a causal-safe fallback. No future/backward fill is used.
    for c in SENSORS:
        if df[c].isna().any():
            df[c] = df[c].ffill()

    if df[SENSORS].isna().any().any():
        raise ValueError("NaNs remain after causal forward fill.")

    return df


def build_point_labels(timestamps: pd.Series) -> np.ndarray:
    """
    0 = Normal
    1 = Warning (2 h preceding documented failure start)
    2 = Fault

    Fault overrides warning.
    """
    ts = pd.to_datetime(timestamps)
    y = np.zeros(len(ts), dtype=np.int8)

    for start, end in EVENTS.values():
        warning_start = start - pd.Timedelta(hours=WARNING_HOURS)

        warning = (ts >= warning_start) & (ts < start)
        fault = (ts >= start) & (ts <= end)

        y[warning.to_numpy()] = 1
        y[fault.to_numpy()] = 2

    return y


def rolling_window_labels(point_labels: np.ndarray, window: int) -> np.ndarray:
    """
    ASPIRE-inspired any-point window labelling.
    Since Fault=2 > Warning=1 > Normal=0, rolling max implements:
      any fault -> Fault
      else any warning -> Warning
      else Normal
    """
    s = pd.Series(point_labels)
    out = (
        s.rolling(window=window, min_periods=window)
        .max()
        .fillna(-1)
        .astype(np.int8)
        .to_numpy()
    )
    return out


def valid_window_end_mask(timestamps: pd.Series, window: int) -> np.ndarray:
    """
    Exclude raw windows that cross acquisition gaps > MAX_GAP_SECONDS.
    """
    ts = pd.to_datetime(timestamps)

    gap = ts.diff().dt.total_seconds().fillna(0).to_numpy()
    bad_gap = (gap > MAX_GAP_SECONDS).astype(np.int8)

    # A W-sample window contains W-1 internal transitions.
    bad_count = (
        pd.Series(bad_gap)
        .rolling(window=window, min_periods=window)
        .sum()
        .fillna(1)
        .to_numpy()
    )

    valid = bad_count == 0
    valid[: window - 1] = False
    return valid


class WindowDataset(Dataset):
    def __init__(
        self,
        scaled_values: np.ndarray,
        labels: np.ndarray,
        end_indices: np.ndarray,
        window: int,
    ) -> None:
        self.x = scaled_values
        self.y = labels
        self.ends = end_indices.astype(np.int64)
        self.window = window

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, i: int):
        end = int(self.ends[i])
        start = end - self.window + 1
        x = self.x[start : end + 1]
        y = int(self.y[end])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


class TransformerClassifier(nn.Module):
    """
    ASPIRE-inspired baseline:
    input projection -> 2 Transformer encoder blocks -> global mean pool
    -> dense 3-class logits.
    """

    def __init__(
        self,
        n_features: int = 12,
        window: int = 100,
        d_model: int = 64,
        n_heads: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.2,
        n_layers: int = 2,
        n_classes: int = 3,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Linear(n_features, d_model)
        self.positional = nn.Parameter(torch.zeros(1, window, d_model))
        nn.init.normal_(self.positional, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        z = self.input_projection(x)
        z = z + self.positional[:, : z.shape[1]]
        z = self.encoder(z)
        z = self.norm(z)
        z = z.mean(dim=1)
        return self.head(z)


def select_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but this PyTorch build has no CUDA support.")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_indices(
    df: pd.DataFrame,
    window_labels: np.ndarray,
    valid: np.ndarray,
    train_normal_stride: int,
    train_event_stride: int,
    eval_stride: int,
):
    ts = df["timestamp"]

    f3_start, f3_end = EVENTS["F03"]
    f4_start, f4_end = EVENTS["F04"]

    f3_warning_start = f3_start - pd.Timedelta(hours=WARNING_HOURS)
    f4_warning_start = f4_start - pd.Timedelta(hours=WARNING_HOURS)

    # Reserve 48 h of normal context before F03 warning for development
    # false-alarm measurement. Nothing from this interval is used for training.
    train_end = f3_warning_start - pd.Timedelta(hours=DEVELOPMENT_NORMAL_HOURS)

    f3_stream_start = train_end
    f3_stream_end = f3_end

    f4_stream_start = f4_warning_start - pd.Timedelta(hours=DEVELOPMENT_NORMAL_HOURS)
    f4_stream_end = f4_end

    all_idx = np.arange(len(df))
    valid_label = window_labels >= 0

    base_train = valid & valid_label & (ts < train_end).to_numpy()

    # Preserve every warning/fault training example while sub-sampling normal
    # windows to keep this exploratory run feasible on CPU.
    normal_train = all_idx[
        base_train
        & (window_labels == 0)
    ][::train_normal_stride]

    event_train = all_idx[
        base_train
        & (window_labels != 0)
    ][::train_event_stride]

    train_idx = np.sort(np.concatenate([normal_train, event_train]))

    f3_idx = all_idx[
        valid
        & valid_label
        & (ts >= f3_stream_start).to_numpy()
        & (ts <= f3_stream_end).to_numpy()
    ][::eval_stride]

    f4_idx = all_idx[
        valid
        & valid_label
        & (ts >= f4_stream_start).to_numpy()
        & (ts <= f4_stream_end).to_numpy()
    ][::eval_stride]

    split_info = {
        "train_end": str(train_end),
        "F03_stream_start": str(f3_stream_start),
        "F03_warning_start": str(f3_warning_start),
        "F03_fault_start": str(f3_start),
        "F03_fault_end": str(f3_end),
        "F04_stream_start": str(f4_stream_start),
        "F04_warning_start": str(f4_warning_start),
        "F04_fault_start": str(f4_start),
        "F04_fault_end": str(f4_end),
    }

    return train_idx, f3_idx, f4_idx, split_info


def class_weights(labels: np.ndarray, end_indices: np.ndarray) -> torch.Tensor:
    y = labels[end_indices]
    counts = np.bincount(y, minlength=3).astype(float)

    # Inverse-frequency weights, normalized around 1.
    weights = counts.sum() / (3.0 * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32), counts.astype(int)


def train_model(
    model,
    train_loader,
    weights,
    device,
    epochs,
    lr,
    weight_decay,
):
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0

        started = time.time()

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(yb)
            total_n += len(yb)

        avg_loss = total_loss / max(total_n, 1)
        elapsed = time.time() - started

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "seconds": elapsed,
        })

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"loss={avg_loss:.6f} | "
            f"{elapsed:.1f}s"
        )

    return pd.DataFrame(history)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    ys = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        p = torch.softmax(logits, dim=1).cpu().numpy()

        probs.append(p)
        ys.append(yb.numpy())

    return np.concatenate(ys), np.concatenate(probs)


def binary_metrics(y_true, probability, positive_class):
    truth = (y_true == positive_class).astype(int)

    pred = np.argmax(probability, axis=1)
    pred_binary = (pred == positive_class).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        truth,
        pred_binary,
        average="binary",
        zero_division=0,
    )

    result = {
        "precision_argmax": float(precision),
        "recall_argmax": float(recall),
        "f1_argmax": float(f1),
        "average_precision": float(
            average_precision_score(truth, probability[:, positive_class])
        ),
    }

    if len(np.unique(truth)) == 2:
        result["roc_auc"] = float(
            roc_auc_score(truth, probability[:, positive_class])
        )
    else:
        result["roc_auc"] = None

    return result


def persistent_flags(
    timestamps: pd.Series,
    probability: np.ndarray,
    threshold: float,
    persistence_windows: int,
):
    ts = pd.Series(pd.to_datetime(timestamps)).reset_index(drop=True)
    p = np.asarray(probability, dtype=float)

    out = np.zeros(len(p), dtype=bool)
    streak = 0
    prev = None

    for i, (t, score) in enumerate(zip(ts, p)):
        # Reset if evaluation windows are not contiguous raw ~10 s points.
        if prev is not None:
            gap = (t - prev).total_seconds()
            if gap > MAX_GAP_SECONDS:
                streak = 0

        if score >= threshold:
            streak += 1
        else:
            streak = 0

        if streak >= persistence_windows:
            out[i] = True

        prev = t

    return out


def alert_episode_count(timestamps, flags):
    times = pd.Series(pd.to_datetime(timestamps))[flags].reset_index(drop=True)

    if len(times) == 0:
        return 0

    episodes = 1
    for i in range(1, len(times)):
        # Same alert episode if high-risk windows continue with <=30 s gap.
        if (times.iloc[i] - times.iloc[i - 1]).total_seconds() > MAX_GAP_SECONDS:
            episodes += 1

    return episodes


def operational_warning_metrics(
    timestamps,
    labels,
    warning_probability,
    event_start,
    threshold,
    persistence_windows,
):
    ts = pd.Series(pd.to_datetime(timestamps)).reset_index(drop=True)
    labels = np.asarray(labels)
    warning_probability = np.asarray(warning_probability)

    flags = persistent_flags(
        ts,
        warning_probability,
        threshold,
        persistence_windows,
    )

    normal_mask = labels == 0
    warning_mask = labels == 1

    normal_episodes = alert_episode_count(
        ts[normal_mask].reset_index(drop=True),
        flags[normal_mask],
    )

    # Approximate observed normal duration from labelled windows.
    normal_days = normal_mask.sum() * 10.0 / 86400.0
    fa_per_day = (
        normal_episodes / normal_days if normal_days > 0 else None
    )

    warning_alert_times = ts[warning_mask & flags]
    if len(warning_alert_times):
        first = warning_alert_times.min()
        lead_minutes = (
            pd.Timestamp(event_start) - first
        ).total_seconds() / 60.0
    else:
        first = pd.NaT
        lead_minutes = None

    return {
        "threshold": float(threshold),
        "persistent_warning_detected": int(len(warning_alert_times) > 0),
        "first_persistent_warning_time": (
            None if pd.isna(first) else str(first)
        ),
        "warning_lead_minutes": (
            None if lead_minutes is None else float(lead_minutes)
        ),
        "false_warning_episodes": int(normal_episodes),
        "false_warning_episodes_per_day": (
            None if fa_per_day is None else float(fa_per_day)
        ),
    }


def select_warning_threshold(
    timestamps,
    labels,
    warning_probability,
    event_start,
    persistence_windows,
):
    rows = []

    for threshold in WARNING_THRESHOLDS:
        rows.append(
            operational_warning_metrics(
                timestamps=timestamps,
                labels=labels,
                warning_probability=warning_probability,
                event_start=event_start,
                threshold=threshold,
                persistence_windows=persistence_windows,
            )
        )

    table = pd.DataFrame(rows)

    qualified = table[
        table["persistent_warning_detected"].eq(1)
    ].copy()

    if len(qualified):
        # Predeclared rule:
        # 1) warning must be detected,
        # 2) minimize false-warning episodes/day,
        # 3) choose highest threshold on tie.
        qualified["_fa"] = qualified["false_warning_episodes_per_day"].fillna(np.inf)
        best = qualified.sort_values(
            ["_fa", "threshold"],
            ascending=[True, False],
        ).iloc[0]
    else:
        # If none warn, retain 0.50 and report failure transparently.
        best = table.iloc[0]

    return float(best["threshold"]), table


def build_prediction_frame(
    df,
    indices,
    y,
    probs,
):
    out = pd.DataFrame({
        "timestamp": df.loc[indices, "timestamp"].to_numpy(),
        "label": y,
        "p_normal": probs[:, 0],
        "p_warning": probs[:, 1],
        "p_fault": probs[:, 2],
        "predicted_class": np.argmax(probs, axis=1),
    })
    return out.sort_values("timestamp").reset_index(drop=True)


def main():
    args = parse_args()
    seed_everything(SEED)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    print("Device:", device)

    df = load_raw(args.raw)
    print("Filtered rows:", len(df))
    print("Time range:", df["timestamp"].min(), "→", df["timestamp"].max())

    point_labels = build_point_labels(df["timestamp"])
    window_labels = rolling_window_labels(point_labels, WINDOW)
    valid = valid_window_end_mask(df["timestamp"], WINDOW)

    train_idx, f3_idx, f4_idx, split_info = make_indices(
        df=df,
        window_labels=window_labels,
        valid=valid,
        train_normal_stride=args.train_normal_stride,
        train_event_stride=args.train_event_stride,
        eval_stride=args.eval_stride,
    )

    print("\nSplit:")
    print(json.dumps(split_info, indent=2))
    print("Train windows:", len(train_idx))
    print("F03 dev windows:", len(f3_idx))
    print("F04 exploratory windows:", len(f4_idx))

    # Fit scaling on raw samples strictly before the reserved development stream.
    train_end = pd.Timestamp(split_info["train_end"])
    scaler_mask = df["timestamp"] < train_end

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(df.loc[scaler_mask, SENSORS])

    scaled = scaler.transform(df[SENSORS]).astype(np.float32)

    weights, counts = class_weights(window_labels, train_idx)
    print("Training class counts [N,W,F]:", counts.tolist())
    print("Class weights:", weights.tolist())

    train_ds = WindowDataset(scaled, window_labels, train_idx, WINDOW)
    f3_ds = WindowDataset(scaled, window_labels, f3_idx, WINDOW)
    f4_ds = WindowDataset(scaled, window_labels, f4_idx, WINDOW)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    eval_batch = max(args.batch_size, 512)

    f3_loader = DataLoader(
        f3_ds,
        batch_size=eval_batch,
        shuffle=False,
        num_workers=0,
    )
    f4_loader = DataLoader(
        f4_ds,
        batch_size=eval_batch,
        shuffle=False,
        num_workers=0,
    )

    model = TransformerClassifier().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", n_params)

    history = train_model(
        model=model,
        train_loader=train_loader,
        weights=weights,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # F03 development evaluation.
    y3, p3 = predict(model, f3_loader, device)
    pred3 = build_prediction_frame(df, f3_idx, y3, p3)

    f3_warning = binary_metrics(y3, p3, positive_class=1)
    f3_fault = binary_metrics(y3, p3, positive_class=2)

    persistence_windows = int(round(PERSISTENCE_MINUTES * 60 / 10))

    selected_threshold, threshold_table = select_warning_threshold(
        timestamps=pred3["timestamp"],
        labels=y3,
        warning_probability=p3[:, 1],
        event_start=EVENTS["F03"][0],
        persistence_windows=persistence_windows,
    )

    f3_operational = operational_warning_metrics(
        timestamps=pred3["timestamp"],
        labels=y3,
        warning_probability=p3[:, 1],
        event_start=EVENTS["F03"][0],
        threshold=selected_threshold,
        persistence_windows=persistence_windows,
    )

    # Freeze threshold after F03 development, then apply unchanged to F04.
    y4, p4 = predict(model, f4_loader, device)
    pred4 = build_prediction_frame(df, f4_idx, y4, p4)

    f4_warning = binary_metrics(y4, p4, positive_class=1)
    f4_fault = binary_metrics(y4, p4, positive_class=2)

    f4_operational = operational_warning_metrics(
        timestamps=pred4["timestamp"],
        labels=y4,
        warning_probability=p4[:, 1],
        event_start=EVENTS["F04"][0],
        threshold=selected_threshold,
        persistence_windows=persistence_windows,
    )

    result = {
        "analysis": "ASPIRE-inspired raw-sequence Transformer challenger",
        "status": "post-hoc external-method-inspired experiment",
        "exact_ASPIRE_replication": False,
        "purpose": (
            "Test whether raw high-frequency sequence representation recovers "
            "pre-fault information lost by 1-minute aggregation."
        ),
        "reference_DOI": "10.1109/ACCESS.2026.3711010",
        "seed": SEED,
        "sensors": SENSORS,
        "window_samples": WINDOW,
        "approx_window_minutes": WINDOW * 10 / 60,
        "warning_horizon_hours": WARNING_HOURS,
        "training_normal_stride": args.train_normal_stride,
        "training_event_stride": args.train_event_stride,
        "eval_stride": args.eval_stride,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": str(device),
        "trainable_parameters": n_params,
        "split": split_info,
        "training_class_counts": counts.tolist(),
        "selected_warning_threshold_from_F03": selected_threshold,
        "persistence_minutes": PERSISTENCE_MINUTES,
        "F03_development": {
            "warning": f3_warning,
            "fault": f3_fault,
            "operational_warning": f3_operational,
        },
        "F04_post_hoc_exploratory": {
            "warning": f4_warning,
            "fault": f4_fault,
            "operational_warning": f4_operational,
        },
    }

    print("\n" + "=" * 78)
    print("F03 DEVELOPMENT")
    print("=" * 78)
    print(json.dumps(result["F03_development"], indent=2))

    print("\n" + "=" * 78)
    print("F04 POST-HOC EXPLORATORY")
    print("=" * 78)
    print(json.dumps(result["F04_post_hoc_exploratory"], indent=2))

    # Save artifacts.
    history.to_csv(args.out_dir / "training_history.csv", index=False)
    threshold_table.to_csv(args.out_dir / "F03_warning_threshold_sweep.csv", index=False)
    pred3.to_parquet(args.out_dir / "F03_predictions.parquet", index=False)
    pred4.to_parquet(args.out_dir / "F04_predictions.parquet", index=False)

    with open(args.out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "sensors": SENSORS,
            "window": WINDOW,
            "selected_warning_threshold": selected_threshold,
            "result": result,
        },
        args.out_dir / "transformer_model.pt",
    )

    # Save scaler parameters without requiring pickle.
    scaler_table = pd.DataFrame({
        "sensor": SENSORS,
        "data_min": scaler.data_min_,
        "data_max": scaler.data_max_,
        "scale": scaler.scale_,
        "min_offset": scaler.min_,
    })
    scaler_table.to_csv(args.out_dir / "train_only_minmax_scaler.csv", index=False)

    print("\nSaved to:", args.out_dir.resolve())


if __name__ == "__main__":
    main()
