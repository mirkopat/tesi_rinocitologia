"""Addestra e valuta una CNN residuale leggera sui crop cellulari."""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from nmcd_common import DEFAULT_OUTPUT_ROOT, ensure_dir, write_json


class CellCropDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, class_to_idx: dict[str, int], transform) -> None:
        self.frame = frame.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.frame.iloc[index]
        image = Image.open(ROOT / record["crop_file"]).convert("RGB")
        label = self.class_to_idx[record["label"]]
        return self.transform(image), torch.tensor(label, dtype=torch.long)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            ConvNormAct(in_channels, out_channels, stride=stride),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(x) + self.shortcut(x))


class ResidualCellCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(3, 32),
            ConvNormAct(32, 32),
        )
        self.features = nn.Sequential(
            ResidualBlock(32, 48, stride=2),
            ResidualBlock(48, 48),
            ResidualBlock(48, 96, stride=2),
            ResidualBlock(96, 96),
            ResidualBlock(96, 160, stride=2),
            ResidualBlock(160, 160),
            ResidualBlock(160, 224, stride=2),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(448, 192),
            nn.SiLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        return self.classifier(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight CNN on cell crops.")
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "metadata" / "cell_crops.csv",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "cnn")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--class-weight-beta", type=float, default=0.999)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load an existing best checkpoint from output-root and only write reports/metrics.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue training from output-root/best_light_cnn.pt.",
    )
    parser.add_argument(
        "--balanced-sampler",
        action="store_true",
        help="Use class-balanced sampling. Off by default because it can overcorrect rare classes.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Debug only: cap rows per split/class before training.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.RandomRotation(25)], p=0.8),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.14, hue=0.035)],
                p=0.75,
            ),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.92, 1.08)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.700, 0.570, 0.815), std=(0.235, 0.260, 0.130)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.700, 0.570, 0.815), std=(0.235, 0.260, 0.130)),
        ]
    )
    return train_transform, eval_transform


def make_loaders(
    data: pd.DataFrame,
    class_to_idx: dict[str, int],
    image_size: int,
    batch_size: int,
    num_workers: int,
    balanced_sampler: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_transform, eval_transform = build_transforms(image_size)
    train = data[data["split"] == "train"].copy()
    valid = data[data["split"] == "valid"].copy()
    test = data[data["split"] == "test"].copy()

    sampler = None
    shuffle = True
    if balanced_sampler:
        train_targets = train["label"].map(class_to_idx).to_numpy()
        class_counts = Counter(train_targets)
        sample_weights = np.array([1.0 / class_counts[target] for target in train_targets], dtype=np.float64)
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    return (
        DataLoader(
            CellCropDataset(train, class_to_idx, train_transform),
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
        ),
        DataLoader(
            CellCropDataset(valid, class_to_idx, eval_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        DataLoader(
            CellCropDataset(test, class_to_idx, eval_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    )


def class_weights(
    data: pd.DataFrame,
    class_to_idx: dict[str, int],
    beta: float,
) -> torch.Tensor:
    train = data[data["split"] == "train"]
    counts = train["label"].map(class_to_idx).value_counts().to_dict()
    weights = []
    for idx in range(len(class_to_idx)):
        count = counts[idx]
        effective_num = 1.0 - np.power(beta, count)
        weights.append((1.0 - beta) / max(effective_num, 1e-8))
    weights_array = np.asarray(weights, dtype=np.float32)
    weights_array = weights_array / weights_array.mean()
    return torch.tensor(weights_array, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    losses = []
    correct = 0
    seen = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        losses.append(float(loss.item()))
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        seen += int(labels.numel())

    return {"loss": float(np.mean(losses)), "accuracy": correct / max(seen, 1)}


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels_out = []
    preds_out = []
    probs_out = []
    for images, labels in loader:
        logits = model(images.to(device))
        probs = torch.softmax(logits, dim=1)
        preds_out.extend(probs.argmax(dim=1).cpu().numpy().tolist())
        probs_out.extend(probs.max(dim=1).values.cpu().numpy().tolist())
        labels_out.extend(labels.numpy().tolist())
    return np.asarray(labels_out), np.asarray(preds_out), np.asarray(probs_out)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    y_true, y_pred, _ = predict(model, loader, device)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def save_split_report(
    model: nn.Module,
    loader: DataLoader,
    split: str,
    labels: list[str],
    output_dir: Path,
    device: torch.device,
) -> None:
    y_true, y_pred, confidence = predict(model, loader, device)
    target_names = labels
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=target_names,
        zero_division=0,
    )
    (output_dir / f"cnn_{split}_classification_report.txt").write_text(report_text, encoding="utf-8")

    report_df = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
            target_names=target_names,
            zero_division=0,
            output_dict=True,
        )
    ).T
    report_df.to_csv(output_dir / f"cnn_{split}_classification_report.csv")

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"cnn - {split}")
    plt.tight_layout()
    plt.savefig(output_dir / f"cnn_{split}_confusion_matrix.png", dpi=160)
    plt.close()

    predictions = pd.DataFrame(
        {
            "true_idx": y_true,
            "predicted_idx": y_pred,
            "true_label": [labels[idx] for idx in y_true],
            "predicted_label": [labels[idx] for idx in y_pred],
            "confidence": confidence,
            "is_correct": y_true == y_pred,
        }
    )
    predictions.to_csv(output_dir / f"cnn_{split}_predictions.csv", index=False)


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)
    if not args.metadata_csv.exists():
        raise FileNotFoundError(
            f"Metadata not found: {args.metadata_csv}. Run 02_export_cell_crops.py first."
        )

    output_dir = ensure_dir(args.output_root)
    data = pd.read_csv(args.metadata_csv)
    if args.limit_per_class is not None:
        data = (
            data.groupby(["split", "label"], group_keys=False)
            .head(args.limit_per_class)
            .reset_index(drop=True)
        )

    labels = sorted(data["label"].unique())
    class_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}
    write_json(output_dir / "class_mapping.json", {"class_to_idx": class_to_idx, "idx_to_class": idx_to_class})

    train_loader, valid_loader, test_loader = make_loaders(
        data=data,
        class_to_idx=class_to_idx,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        balanced_sampler=args.balanced_sampler,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualCellCNN(num_classes=len(labels)).to(device)
    best_path = output_dir / "best_light_cnn.pt"

    if args.eval_only:
        if not best_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {best_path}")
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        metrics = []
        for split, loader in [("valid", valid_loader), ("test", test_loader)]:
            split_metrics = evaluate(model, loader, device)
            metrics.append({"model": "residual_cnn", "split": split, **split_metrics})
            save_split_report(model, loader, split, labels, output_dir, device)
        pd.DataFrame(metrics).to_csv(output_dir / "cnn_metrics.csv", index=False)
        print(f"Loaded checkpoint validation macro-F1: {checkpoint.get('valid_macro_f1', float('nan')):.4f}")
        print(f"Wrote CNN reports to: {output_dir}")
        return

    best_macro_f1 = -1.0
    if args.resume:
        if not best_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {best_path}")
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        best_macro_f1 = float(checkpoint.get("valid_macro_f1", -1.0))
        print(f"Resuming from checkpoint with validation macro-F1: {best_macro_f1:.4f}", flush=True)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights(data, class_to_idx, args.class_weight_beta).to(device),
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    stale_epochs = 0
    history_path = output_dir / "cnn_training_history.csv"
    history = []
    if args.resume and history_path.exists():
        history = pd.read_csv(history_path).to_dict("records")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.grad_clip,
        )
        valid_metrics = evaluate(model, valid_loader, device)
        scheduler.step(valid_metrics["macro_f1"])

        row = {
            "epoch": len(history) + 1,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_macro_f1": valid_metrics["macro_f1"],
            "valid_weighted_f1": valid_metrics["weighted_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(
            f"epoch {epoch:02d} "
            f"train_loss={row['train_loss']:.4f} "
            f"valid_acc={row['valid_accuracy']:.4f} "
            f"valid_macro_f1={row['valid_macro_f1']:.4f}",
            flush=True,
        )

        if valid_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = valid_metrics["macro_f1"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "image_size": args.image_size,
                    "architecture": "ResidualCellCNN",
                    "valid_macro_f1": best_macro_f1,
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    pd.DataFrame(history).to_csv(history_path, index=False)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    metrics = []
    for split, loader in [("valid", valid_loader), ("test", test_loader)]:
        split_metrics = evaluate(model, loader, device)
        metrics.append({"model": "residual_cnn", "split": split, **split_metrics})
        save_split_report(model, loader, split, labels, output_dir, device)

    pd.DataFrame(metrics).to_csv(output_dir / "cnn_metrics.csv", index=False)
    config = vars(args).copy()
    config["metadata_csv"] = str(config["metadata_csv"])
    config["output_root"] = str(config["output_root"])
    config["device"] = str(device)
    config["classes"] = labels
    write_json(output_dir / "cnn_config.json", config)
    print(f"Best validation macro-F1: {best_macro_f1:.4f}")
    print(f"Wrote CNN outputs to: {output_dir}")


if __name__ == "__main__":
    main()
