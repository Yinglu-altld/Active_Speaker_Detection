import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WINDOWS_NPZ = PROJECT_ROOT / "data" / "windows" / "windows.npz"
DEFAULT_WINDOWS_META = PROJECT_ROOT / "data" / "windows" / "windows_meta.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "models" / "cnn_vvad"


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])  # (T, P, 2)
        y = torch.tensor(int(self.y[idx]), dtype=torch.float32)  # {0,1}
        return x, y


class TemporalCNN(nn.Module):
    def __init__(self, num_points: int, num_channels: int = 2):
        super().__init__()
        in_ch = num_points * num_channels
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, P, 2) -> (B, C, T) where C=P*2
        b, t, p, c = x.shape
        x = x.reshape(b, t, p * c).transpose(1, 2)
        feats = self.net(x).squeeze(-1)
        return self.head(feats).squeeze(-1)  # logits


@dataclass(frozen=True)
class TrainConfig:
    npz: str
    meta_csv: str
    val_video: str
    min_speech_ratio: float
    max_speech_ratio: float
    filter_extremes: bool
    neg_max: float
    pos_min: float
    use_delta: bool
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    seed: int
    device: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 5: Train a small temporal CNN on Step 4 landmark windows.\n"
            "Split is done by video_id to avoid leakage."
        )
    )
    parser.add_argument("--npz", default=str(DEFAULT_WINDOWS_NPZ))
    parser.add_argument("--meta-csv", default=str(DEFAULT_WINDOWS_META))
    parser.add_argument("--val-video", required=True)
    parser.add_argument("--min-speech-ratio", type=float, default=0.0)
    parser.add_argument("--max-speech-ratio", type=float, default=1.0)
    parser.add_argument(
        "--filter-extremes",
        action="store_true",
        help="Keep only clear negatives (<= neg-max) or clear positives (>= pos-min).",
    )
    parser.add_argument("--neg-max", type=float, default=0.1)
    parser.add_argument("--pos-min", type=float, default=0.6)
    parser.add_argument(
        "--no-delta",
        action="store_true",
        help="Disable delta features (frame-to-frame landmark differences).",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    losses = []
    y_true = []
    y_prob = []
    bce = nn.BCEWithLogitsLoss()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = bce(logits, y)
        losses.append(float(loss.item()))
        y_true.append(y.detach().cpu().numpy())
        y_prob.append(torch.sigmoid(logits).detach().cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_prob = np.concatenate(y_prob, axis=0)
    y_pred = (y_prob >= 0.5).astype(np.int32)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = (2 * prec * rec) / max(1e-9, (prec + rec))

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pos_rate": float(y_true.mean()) if y_true.size else float("nan"),
        "pred_pos_rate": float(y_pred.mean()) if y_pred.size else float("nan"),
    }


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    thresholds = np.linspace(0.05, 0.95, 19)
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0, "acc": 0.0}
    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.int32)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        acc = (tp + tn) / max(1, (tp + tn + fp + fn))
        prec = tp / max(1, (tp + fp))
        rec = tp / max(1, (tp + fn))
        f1 = (2 * prec * rec) / max(1e-9, (prec + rec))
        if f1 > best["f1"]:
            best = {
                "threshold": float(t),
                "f1": float(f1),
                "precision": float(prec),
                "recall": float(rec),
                "acc": float(acc),
            }
    return best


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        npz=args.npz,
        meta_csv=args.meta_csv,
        val_video=args.val_video,
        min_speech_ratio=args.min_speech_ratio,
        max_speech_ratio=args.max_speech_ratio,
        filter_extremes=args.filter_extremes,
        neg_max=args.neg_max,
        pos_min=args.pos_min,
        use_delta=not args.no_delta,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    data = np.load(cfg.npz)
    X = data["X"]  # (N, T, P, 2)
    y = data["y"]  # (N,)
    meta_all = pd.read_csv(cfg.meta_csv)

    if len(meta_all) != len(y):
        raise ValueError(
            f"Meta rows ({len(meta_all)}) must match y length ({len(y)})."
        )

    mask = np.ones(len(meta_all), dtype=bool)
    if "speech_ratio" in meta_all.columns:
        if cfg.filter_extremes:
            mask &= (meta_all["speech_ratio"] <= cfg.neg_max) | (
                meta_all["speech_ratio"] >= cfg.pos_min
            )
        else:
            mask &= (meta_all["speech_ratio"] >= cfg.min_speech_ratio) & (
                meta_all["speech_ratio"] <= cfg.max_speech_ratio
            )
    meta = meta_all[mask].reset_index(drop=True)
    X = X[mask]
    y = y[mask]

    val_mask = meta["video_id"].astype(str) == cfg.val_video
    if not bool(val_mask.any()):
        raise ValueError(
            f"--val-video {cfg.val_video!r} not found in windows_meta.csv"
        )

    train_idx = np.where(~val_mask.to_numpy())[0]
    val_idx = np.where(val_mask.to_numpy())[0]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    if X_train.size == 0 or X_val.size == 0:
        raise ValueError("Train/val split is empty after speech_ratio filtering.")

    if cfg.use_delta:
        def append_delta(arr: np.ndarray) -> np.ndarray:
            delta = np.zeros_like(arr)
            delta[:, 1:, :, :] = arr[:, 1:, :, :] - arr[:, :-1, :, :]
            return np.concatenate([arr, delta], axis=3)

        X_train = append_delta(X_train)
        X_val = append_delta(X_val)
        num_channels = 4
    else:
        num_channels = 2

    num_points = int(X.shape[2])
    model = TemporalCNN(num_points=num_points, num_channels=num_channels).to(device)

    # Basic class imbalance handling.
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([neg / max(1.0, pos)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    train_loader = DataLoader(
        WindowDataset(X_train, y_train),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        WindowDataset(X_val, y_val),
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    best_f1 = -1.0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))

        val_metrics = eval_epoch(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), out_dir / "best.pt")

        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"val_f1={val_metrics['f1']:.3f} val_acc={val_metrics['acc']:.3f} "
            f"val_pos_rate={val_metrics['pos_rate']:.3f}"
        )

    # Threshold tuning on validation set (post-training)
    model.eval()
    y_true = []
    y_prob = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            y_true.append(yb.numpy())
            y_prob.append(torch.sigmoid(logits).cpu().numpy())
    y_true = np.concatenate(y_true, axis=0)
    y_prob = np.concatenate(y_prob, axis=0)
    best_thresh = tune_threshold(y_true, y_prob)

    torch.save(model.state_dict(), out_dir / "last.pt")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    (out_dir / "threshold.json").write_text(json.dumps(best_thresh, indent=2))

    print(f"train_windows={len(train_idx)}, val_windows={len(val_idx)}")
    print(f"out_dir={out_dir}")
    print(
        f"best_threshold={best_thresh['threshold']:.2f} "
        f"best_f1={best_thresh['f1']:.3f}"
    )


if __name__ == "__main__":
    main()
