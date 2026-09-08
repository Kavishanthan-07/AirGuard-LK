from pathlib import Path
import gc
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from sklearn.metrics import (
    average_precision_score,
    auc,
    precision_recall_curve,
    roc_auc_score,
)

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MASTER_PATH = (
    ROOT
    / "data"
    / "processed"
    / "metropt3_model_master.parquet"
)

SPLIT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "splits"
    / "fault_state_split_membership.parquet"
)

EVENT_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "metropt3_failure_events.csv"
)

CONFIG_PATH = (
    ROOT
    / "configs"
    / "gru_30m_challenger.yaml"
)

METRIC_DIR = ROOT / "artifacts" / "metrics"
TABLE_DIR = ROOT / "artifacts" / "tables"
MODEL_DIR = ROOT / "artifacts" / "models"

for directory in [
    METRIC_DIR,
    TABLE_DIR,
    MODEL_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# CONFIG
# ============================================================

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8",
) as f:

    config = yaml.safe_load(f)


SEED = int(
    config["random_seed"]
)

SEQ_LEN = int(
    config["sequence"][
        "length_minutes"
    ]
)

CHANNELS = list(
    config["sequence"][
        "channels"
    ]
)

EPOCHS = int(
    config["training"][
        "epochs"
    ]
)

BATCH_SIZE = int(
    config["training"][
        "batch_size"
    ]
)

LEARNING_RATE = float(
    config["training"][
        "learning_rate"
    ]
)

WEIGHT_DECAY = float(
    config["training"][
        "weight_decay"
    ]
)

PERSISTENCE = int(
    config["alerting"][
        "persistence_minutes"
    ]
)

QUANTILES = [
    float(x)
    for x in config["alerting"][
        "threshold_quantiles"
    ]
]


assert len(CHANNELS) == 18


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

threads = min(
    8,
    os.cpu_count() or 1,
)

torch.set_num_threads(
    threads
)

DEVICE = torch.device(
    "cpu"
)


print("=" * 100)
print("AIRGUARD-LK — GRU-30M DEVELOPMENT CHALLENGER")
print("=" * 100)

print(
    f"\nPyTorch: {torch.__version__}"
)

print(
    f"Device: {DEVICE}"
)

print(
    f"CPU threads: {threads}"
)

print(
    f"Sequence: {SEQ_LEN} min x "
    f"{len(CHANNELS)} channels"
)

print(
    "\nF04 is NOT used for "
    "CNN model selection."
)


# ============================================================
# LOAD DATA
# ============================================================

master = pd.read_parquet(
    MASTER_PATH
)

master["timestamp"] = pd.to_datetime(
    master["timestamp"]
)

master = (
    master
    .sort_values("timestamp")
    .reset_index(drop=True)
)


membership = pd.read_parquet(
    SPLIT_PATH
)

membership["timestamp"] = pd.to_datetime(
    membership["timestamp"]
)


events = pd.read_csv(
    EVENT_PATH,
    parse_dates=[
        "start_time",
        "end_time",
    ],
)


event_start_lookup = (
    events
    .set_index("event_id")
    ["start_time"]
    .to_dict()
)


FOLDS = [
    {
        "split": "fault_F02",
        "validation_event": "F02",
    },
    {
        "split": "fault_F03",
        "validation_event": "F03",
    },
]


# ============================================================
# CNN
# ============================================================

class TinyGRU(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=32,
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_size,
                16,
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                16,
                1,
            ),
        )


    def forward(
        self,
        x,
    ):

        # x:
        # batch x time x channels

        _, hidden = self.gru(
            x
        )

        final_hidden = (
            hidden[-1]
        )

        logits = (
            self.classifier(
                final_hidden
            )
            .squeeze(1)
        )

        return logits

# ============================================================
# HELPERS
# ============================================================

def ranking_metrics(
    y_true,
    scores,
):

    ap = average_precision_score(
        y_true,
        scores,
    )

    precision, recall, _ = (
        precision_recall_curve(
            y_true,
            scores,
        )
    )

    pr_auc = auc(
        recall,
        precision,
    )

    roc_auc = roc_auc_score(
        y_true,
        scores,
    )

    prevalence = float(
        np.mean(y_true)
    )

    return {
        "pr_auc":
            float(pr_auc),

        "average_precision":
            float(ap),

        "roc_auc":
            float(roc_auc),

        "positive_prevalence":
            prevalence,

        "ap_lift_over_prevalence":
            float(
                ap / prevalence
            )
            if prevalence > 0
            else np.nan,
    }


def persistent_alert_flags(
    timestamps,
    scores,
    threshold,
):

    timestamps = (
        pd.Series(
            pd.to_datetime(
                timestamps
            )
        )
        .reset_index(drop=True)
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    alerts = np.zeros(
        len(scores),
        dtype=bool,
    )

    streak = 0
    previous_timestamp = None


    for i in range(
        len(scores)
    ):

        timestamp = (
            timestamps.iloc[i]
        )


        if (
            previous_timestamp
            is None
        ):

            consecutive = True

        else:

            consecutive = (
                (
                    timestamp
                    -
                    previous_timestamp
                ).total_seconds()
                == 60
            )


        if not consecutive:

            streak = 0


        if (
            scores[i]
            >= threshold
        ):

            streak += 1

        else:

            streak = 0


        if (
            streak
            >= PERSISTENCE
        ):

            alerts[i] = True


        previous_timestamp = (
            timestamp
        )


    return alerts


def count_alert_episodes(
    timestamps,
    alerts,
):

    alert_times = (
        pd.Series(
            pd.to_datetime(
                timestamps
            )
        )
        .loc[
            np.asarray(
                alerts,
                dtype=bool,
            )
        ]
        .sort_values()
        .reset_index(drop=True)
    )


    if len(
        alert_times
    ) == 0:

        return 0


    episodes = 1


    for i in range(
        1,
        len(alert_times)
    ):

        gap = (
            alert_times.iloc[i]
            -
            alert_times.iloc[
                i - 1
            ]
        ).total_seconds()


        if gap > 60:

            episodes += 1


    return episodes


def operational_metrics(
    metadata,
    scores,
    threshold,
    event_start,
):

    temp = metadata[
        [
            "timestamp",
            "role",
        ]
    ].copy()

    temp["score"] = scores

    temp = (
        temp
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )


    temp["alert"] = (
        persistent_alert_flags(
            temp["timestamp"],
            temp["score"],
            threshold,
        )
    )


    negative = temp[
        temp["role"]
        == "evaluation_negative"
    ]


    false_alarm_episodes = (
        count_alert_episodes(
            negative[
                "timestamp"
            ],
            negative[
                "alert"
            ],
        )
    )


    negative_days = (
        len(negative)
        / 1440.0
    )


    false_alarm_rate = (
        false_alarm_episodes
        / negative_days
        if negative_days > 0
        else np.nan
    )


    failure = temp[
        temp["role"]
        == "evaluation_failure"
    ]


    fault_alerts = failure[
        failure["alert"]
    ]


    detected = (
        len(fault_alerts)
        > 0
    )


    if detected:

        first_alert = (
            fault_alerts[
                "timestamp"
            ].min()
        )

        delay = (
            first_alert
            -
            event_start
        ).total_seconds() / 60.0

    else:

        delay = np.nan


    return {
        "event_detected":
            int(detected),

        "detection_delay_minutes":
            float(delay)
            if np.isfinite(delay)
            else np.nan,

        "false_alarm_episodes":
            int(
                false_alarm_episodes
            ),

        "negative_days":
            float(
                negative_days
            ),

        "false_alarm_episodes_per_day":
            float(
                false_alarm_rate
            ),
    }


def make_sequences(
    master_df,
    role_df,
    allowed_roles,
):

    role_subset = (
        role_df[
            role_df["role"]
            .isin(
                allowed_roles
            )
        ]
        .copy()
    )


    role_map = (
        role_subset
        .set_index(
            "timestamp"
        )["role"]
        .to_dict()
    )


    timestamp_to_index = (
        pd.Series(
            master_df.index,
            index=master_df[
                "timestamp"
            ],
        )
        .to_dict()
    )


    X_list = []
    y_list = []
    metadata_rows = []


    channel_values = (
        master_df[
            CHANNELS
        ]
        .to_numpy(
            dtype=np.float32
        )
    )


    segment_ids = (
        master_df[
            "quality__segment_id"
        ]
        .to_numpy()
    )


    low_sample = (
        master_df[
            "quality__low_sample_minute"
        ]
        .to_numpy(
            dtype=np.int8
        )
    )


    raw_gap = (
        master_df[
            "quality__raw_gap_over_30s"
        ]
        .to_numpy(
            dtype=np.int8
        )
    )


    timestamps = (
        master_df[
            "timestamp"
        ]
        .to_numpy()
    )


    for timestamp, role in (
        role_map.items()
    ):

        if (
            timestamp
            not in
            timestamp_to_index
        ):

            continue


        end_idx = (
            timestamp_to_index[
                timestamp
            ]
        )


        start_idx = (
            end_idx
            -
            SEQ_LEN
            +
            1
        )


        if start_idx < 0:

            continue


        # Must stay inside one
        # acquisition segment.
        if (
            segment_ids[
                start_idx
            ]
            !=
            segment_ids[
                end_idx
            ]
        ):

            continue


        # No problematic minute
        # inside the sequence.
        if (
            low_sample[
                start_idx:
                end_idx + 1
            ].any()
        ):

            continue


        if (
            raw_gap[
                start_idx:
                end_idx + 1
            ].any()
        ):

            continue


        seq_timestamps = (
            timestamps[
                start_idx:
                end_idx + 1
            ]
        )


        if len(
            seq_timestamps
        ) != SEQ_LEN:

            continue


        # Consecutive wall-clock
        # minutes only.
        deltas = (
            np.diff(
                seq_timestamps
            )
            /
            np.timedelta64(
                1,
                "s",
            )
        )


        if not np.all(
            deltas == 60
        ):

            continue


        sequence = (
            channel_values[
                start_idx:
                end_idx + 1
            ]
        )


        if not np.isfinite(
            sequence
        ).all():

            continue


        X_list.append(
            sequence
        )


        y_list.append(
            int(
                role
                in [
                    "train_failure",
                    "evaluation_failure",
                ]
            )
        )


        metadata_rows.append({
            "timestamp":
                timestamp,

            "role":
                role,
        })


    X = np.asarray(
        X_list,
        dtype=np.float32,
    )

    y = np.asarray(
        y_list,
        dtype=np.float32,
    )

    metadata = pd.DataFrame(
        metadata_rows
    )


    return (
        X,
        y,
        metadata,
    )


def fit_scaler(
    X_train,
):

    # X:
    # samples x time x channels

    flattened = (
        X_train
        .reshape(
            -1,
            X_train.shape[-1],
        )
    )


    mean = (
        flattened
        .mean(
            axis=0
        )
        .astype(
            np.float32
        )
    )


    std = (
        flattened
        .std(
            axis=0
        )
        .astype(
            np.float32
        )
    )


    std[
        std < 1e-6
    ] = 1.0


    return (
        mean,
        std,
    )


def normalize(
    X,
    mean,
    std,
):

    return (
        (
            X
            -
            mean[
                None,
                None,
                :
            ]
        )
        /
        std[
            None,
            None,
            :
        ]
    ).astype(
        np.float32
    )


def predict_scores(
    model,
    X,
):

    model.eval()

    dataset = TensorDataset(
        torch.from_numpy(
            X
        )
    )


    loader = DataLoader(
        dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=0,
    )


    scores = []


    with torch.no_grad():

        for (
            batch_x,
        ) in loader:

            batch_x = (
                batch_x
                .to(
                    DEVICE
                )
            )


            logits = model(
                batch_x
            )


            probs = torch.sigmoid(
                logits
            )


            scores.append(
                probs.cpu()
                .numpy()
            )


    return np.concatenate(
        scores
    )


# ============================================================
# RESULTS
# ============================================================

ranking_rows = []
operational_rows = []
training_rows = []
prediction_parts = []


# ============================================================
# RUN FOLDS
# ============================================================

for fold in FOLDS:

    split_name = (
        fold["split"]
    )

    validation_event = (
        fold[
            "validation_event"
        ]
    )


    print(
        "\n"
        +
        "=" * 100
    )

    print(
        f"{split_name} "
        f"→ {validation_event}"
    )

    print(
        "=" * 100
    )


    fold_roles = (
        membership[
            membership["split"]
            == split_name
        ][
            [
                "timestamp",
                "role",
            ]
        ]
        .copy()
    )


    # --------------------------------------------------------
    # BUILD TRAIN SEQUENCES
    # --------------------------------------------------------

    X_train, y_train, train_meta = (
        make_sequences(
            master,
            fold_roles,
            [
                "train_negative",
                "train_failure",
            ],
        )
    )


    # --------------------------------------------------------
    # BUILD EVALUATION SEQUENCES
    # --------------------------------------------------------

    X_eval, y_eval, eval_meta = (
        make_sequences(
            master,
            fold_roles,
            [
                "evaluation_negative",
                "evaluation_pre_event_context",
                "evaluation_failure",
            ],
        )
    )


    print(
        f"Train sequences: "
        f"{len(X_train):,}"
    )

    print(
        f"Train fault sequences: "
        f"{int(y_train.sum()):,}"
    )

    print(
        f"Evaluation sequences: "
        f"{len(X_eval):,}"
    )

    print(
        "Evaluation fault sequences: "
        f"{int(y_eval.sum()):,}"
    )


    if (
        y_train.sum() == 0
        or
        y_eval.sum() == 0
    ):

        raise RuntimeError(
            "Missing fault sequences."
        )


    # ========================================================
    # NORMALIZATION
    # ========================================================

    channel_mean, channel_std = (
        fit_scaler(
            X_train
        )
    )


    X_train = normalize(
        X_train,
        channel_mean,
        channel_std,
    )


    X_eval = normalize(
        X_eval,
        channel_mean,
        channel_std,
    )


    # PyTorch Conv1D:
    # N x C x L

    train_tensor = (
    	torch.from_numpy(
            X_train
   	)
    )


    y_tensor = (
        torch.from_numpy(
            y_train
        )
    )


    dataset = TensorDataset(
        train_tensor,
        y_tensor,
    )


    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        SEED
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )


    # ========================================================
    # MODEL
    # ========================================================

    torch.manual_seed(
        SEED
    )


    model = TinyGRU(
    input_size=len(CHANNELS),
    hidden_size=32,
    ).to(
    	DEVICE
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )


    n_pos = float(
        y_train.sum()
    )

    n_neg = float(
        len(y_train)
        -
        n_pos
    )


    pos_weight_value = (
        n_neg
        /
        max(
            n_pos,
            1.0,
        )
    )


    pos_weight = torch.tensor(
        [
            pos_weight_value
        ],
        dtype=torch.float32,
        device=DEVICE,
    )


    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=
                pos_weight
        )
    )


    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=
                WEIGHT_DECAY,
        )
    )


    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )

    print(
        f"Positive weight: "
        f"{pos_weight_value:.2f}"
    )


    # ========================================================
    # TRAIN
    #
    # IMPORTANT:
    # No F02/F03 validation event is
    # used for early stopping.
    # Epoch count is fixed beforehand.
    # ========================================================

    epoch_losses = []


    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        total_loss = 0.0
        total_rows = 0


        for (
            batch_x,
            batch_y,
        ) in loader:

            batch_x = (
                batch_x
                .to(
                    DEVICE
                )
            )

            batch_y = (
                batch_y
                .to(
                    DEVICE
                )
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            logits = model(
                batch_x
            )


            loss = criterion(
                logits,
                batch_y,
            )


            loss.backward()


            optimizer.step()


            batch_size_actual = (
                batch_x.size(0)
            )


            total_loss += (
                loss.item()
                *
                batch_size_actual
            )

            total_rows += (
                batch_size_actual
            )


        epoch_loss = (
            total_loss
            /
            total_rows
        )


        epoch_losses.append(
            epoch_loss
        )


        print(
            f"Epoch "
            f"{epoch:02d}/"
            f"{EPOCHS} "
            f"| loss "
            f"{epoch_loss:.6f}"
        )


    # ========================================================
    # SCORES
    # ========================================================

    train_scores = predict_scores(
        model,
        X_train,
    )


    eval_scores = predict_scores(
        model,
        X_eval,
    )


    # ========================================================
    # TRAIN NORMAL SCORES
    # ========================================================

    train_normal_mask = (
        train_meta["role"]
        .eq(
            "train_negative"
        )
        .to_numpy()
    )


    train_normal_scores = (
        train_scores[
            train_normal_mask
        ]
    )


    # ========================================================
    # BINARY EVALUATION
    # ========================================================

    binary_mask = (
        eval_meta["role"]
        .isin([
            "evaluation_negative",
            "evaluation_failure",
        ])
        .to_numpy()
    )


    binary_y = (
        eval_meta.loc[
            binary_mask,
            "role",
        ]
        .eq(
            "evaluation_failure"
        )
        .astype(int)
        .to_numpy()
    )


    binary_scores = (
        eval_scores[
            binary_mask
        ]
    )


    ranking = ranking_metrics(
        binary_y,
        binary_scores,
    )


    ranking_rows.append({
        "split":
            split_name,

        "validation_event":
            validation_event,

        "model":
            "GRU-30m",

        "sequence_minutes":
            SEQ_LEN,

        "channels":
            len(CHANNELS),

        "parameters":
            int(
                parameter_count
            ),

        **ranking,
    })


    # ========================================================
    # OPERATIONAL THRESHOLDS
    # ========================================================

    event_start = pd.Timestamp(
        event_start_lookup[
            validation_event
        ]
    )


    for quantile in QUANTILES:

        threshold = float(
            np.quantile(
                train_normal_scores,
                quantile,
            )
        )


        operation = (
            operational_metrics(
                eval_meta,
                eval_scores,
                threshold,
                event_start,
            )
        )


        operational_rows.append({
            "split":
                split_name,

            "validation_event":
                validation_event,

            "model":
                "GRU-30m",

            "threshold_quantile":
                quantile,

            "threshold_value":
                threshold,

            **operation,
        })


    # ========================================================
    # TRAIN LOG
    # ========================================================

    for epoch, loss in enumerate(
        epoch_losses,
        start=1,
    ):

        training_rows.append({
            "split":
                split_name,

            "epoch":
                epoch,

            "loss":
                loss,
        })


    # ========================================================
    # PREDICTIONS
    # ========================================================

    pred = eval_meta.copy()

    pred["split"] = (
        split_name
    )

    pred[
        "validation_event"
    ] = validation_event

    pred["score"] = (
        eval_scores
    )


    prediction_parts.append(
        pred
    )


    # ========================================================
    # SAVE DEVELOPMENT MODEL
    # ========================================================

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "channels":
                CHANNELS,

            "sequence_length":
                SEQ_LEN,

            "channel_mean":
                channel_mean,

            "channel_std":
                channel_std,

            "parameter_count":
                parameter_count,
        },

        MODEL_DIR
        /
        f"gru30m_{split_name}.pt",
    )


    # ========================================================
    # CLEANUP BETWEEN FOLDS
    # ========================================================

    del (
        X_train,
        X_eval,
        train_tensor,
        dataset,
        loader,
        model,
    )

    gc.collect()


# ============================================================
# SAVE RAW RESULTS
# ============================================================

ranking_df = pd.DataFrame(
    ranking_rows
)

operational_df = pd.DataFrame(
    operational_rows
)

training_df = pd.DataFrame(
    training_rows
)

predictions_df = pd.concat(
    prediction_parts,
    ignore_index=True,
)


ranking_df.to_csv(
    METRIC_DIR
    / "gru_30m_development_ranking.csv",
    index=False,
)


operational_df.to_csv(
    METRIC_DIR
    / "gru_30m_development_operational.csv",
    index=False,
)


training_df.to_csv(
    METRIC_DIR
    / "gru_30m_training_log.csv",
    index=False,
)


predictions_df.to_parquet(
    METRIC_DIR
    / "gru_30m_development_predictions.parquet",
    index=False,
)


# ============================================================
# MACRO RANKING
# ============================================================

ranking_scoreboard = (
    ranking_df
    .groupby(
        "model",
        as_index=False,
    )
    .agg(
        mean_pr_auc=(
            "pr_auc",
            "mean",
        ),

        minimum_event_pr_auc=(
            "pr_auc",
            "min",
        ),

        maximum_event_pr_auc=(
            "pr_auc",
            "max",
        ),

        mean_average_precision=(
            "average_precision",
            "mean",
        ),

        mean_roc_auc=(
            "roc_auc",
            "mean",
        ),

        mean_ap_lift=(
            "ap_lift_over_prevalence",
            "mean",
        ),
    )
)


ranking_scoreboard.to_csv(
    TABLE_DIR
    / "gru_30m_ranking_scoreboard.csv",
    index=False,
)


# ============================================================
# OPERATIONAL SCOREBOARD
# ============================================================

operational_scoreboard = (
    operational_df
    .groupby(
        [
            "model",
            "threshold_quantile",
        ],
        as_index=False,
    )
    .agg(
        events_detected=(
            "event_detected",
            "sum",
        ),

        median_detection_delay_minutes=(
            "detection_delay_minutes",
            "median",
        ),

        mean_false_alarms_per_day=(
            "false_alarm_episodes_per_day",
            "mean",
        ),

        worst_false_alarms_per_day=(
            "false_alarm_episodes_per_day",
            "max",
        ),
    )
)


operational_scoreboard[
    "events_total"
] = 2


operational_scoreboard.to_csv(
    TABLE_DIR
    / "gru_30m_operational_scoreboard.csv",
    index=False,
)


# ============================================================
# QUALIFIED
# ============================================================

qualified = (
    operational_scoreboard[
        (
            operational_scoreboard[
                "events_detected"
            ] == 2
        )
        &
        (
            operational_scoreboard[
                "median_detection_delay_minutes"
            ] <= 5
        )
    ]
    .copy()
)


qualified.to_csv(
    TABLE_DIR
    / "gru_30m_qualified.csv",
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "model":
        "GRU-30m",

    "device":
        str(DEVICE),

    "torch_version":
        torch.__version__,

    "development_events": [
        "F02",
        "F03",
    ],

    "F04_used_for_selection":
        False,

    "sequence_length_minutes":
        SEQ_LEN,

    "channel_count":
        len(CHANNELS),

    "epochs":
        EPOCHS,

    "qualified_configuration_count":
        int(
            len(qualified)
        ),
}


with open(
    METRIC_DIR
    / "gru_30m_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# DISPLAY
# ============================================================

pd.set_option(
    "display.max_columns",
    None,
)

pd.set_option(
    "display.width",
    240,
)


print("\n" + "=" * 100)
print("TINY CNN RANKING")
print("=" * 100)

print(
    ranking_df
    .round(5)
    .to_string(index=False)
)


print("\n" + "=" * 100)
print("TINY CNN OPERATIONAL SCOREBOARD")
print("=" * 100)

print(
    operational_scoreboard
    .round(5)
    .to_string(index=False)
)


print("\nIMPORTANT:")
print(
    "F04 was not used for "
    "training, early stopping, "
    "threshold selection, or "
    "challenger selection."
)

print(
    "\nGRU-30M EXPERIMENT COMPLETE"
)
