"""Esegue transfer learning su EfficientNet-B0 o ResNet18 per i crop cellulari."""

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
from torchvision import models, transforms


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer learning on cell crops.")
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "metadata" / "cell_crops.csv",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "transfer_learning")
    parser.add_argument("--backbone", choices=["efficientnet_b0", "resnet18"], default="efficientnet_b0")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--head-epochs", type=int, default=6)
    parser.add_argument("--finetune-epochs", type=int, default=18)
    parser.add_argument("--head-lr", type=float, default=8e-4)
    parser.add_argument("--finetune-lr", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--class-weight-beta", type=float, default=0.997)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
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


def imagenet_normalization() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    mean, std = imagenet_normalization()
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.RandomRotation(25)], p=0.8),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.16, contrast=0.16, saturation=0.12, hue=0.03)],
                p=0.7,
            ),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.92, 1.08)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
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


def class_weights(data: pd.DataFrame, class_to_idx: dict[str, int], beta: float) -> torch.Tensor:
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


def build_model(backbone: str, num_classes: int, pretrained: bool) -> nn.Module:
    if backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(in_features, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )
        return model

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.30),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.20),
        nn.Linear(256, num_classes),
    )
    return model


def classifier_parameter_names(model: nn.Module, backbone: str) -> tuple[str, ...]:
    if backbone == "efficientnet_b0":
        return ("classifier",)
    return ("fc",)


def set_trainable(model: nn.Module, backbone: str, mode: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    classifier_prefixes = classifier_parameter_names(model, backbone)
    for name, parameter in model.named_parameters():
        if name.startswith(classifier_prefixes):
            parameter.requires_grad = True

    if mode == "finetune":
        if backbone == "efficientnet_b0":
            for name, parameter in model.named_parameters():
                if name.startswith("features.5") or name.startswith("features.6") or name.startswith("features.7"):
                    parameter.requires_grad = True
        else:
            for name, parameter in model.named_parameters():
                if name.startswith("layer3") or name.startswith("layer4"):
                    parameter.requires_grad = True


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


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
            nn.utils.clip_grad_norm_(trainable_parameters(model), grad_clip)
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
        probs = torch.softmax(model(images.to(device)), dim=1)
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
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        zero_division=0,
    )
    (output_dir / f"transfer_{split}_classification_report.txt").write_text(report_text, encoding="utf-8")

    report_df = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
            target_names=labels,
            zero_division=0,
            output_dict=True,
        )
    ).T
    report_df.to_csv(output_dir / f"transfer_{split}_classification_report.csv")

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"transfer learning - {split}")
    plt.tight_layout()
    plt.savefig(output_dir / f"transfer_{split}_confusion_matrix.png", dpi=160)
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
    predictions.to_csv(output_dir / f"transfer_{split}_predictions.csv", index=False)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    class_to_idx: dict[str, int],
    idx_to_class: dict[int, str],
    best_macro_f1: float,
    phase: str,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "image_size": args.image_size,
            "backbone": args.backbone,
            "pretrained": args.pretrained,
            "valid_macro_f1": best_macro_f1,
            "phase": phase,
        },
        path,
    )


def run_phase(
    model: nn.Module,
    phase: str,
    epochs: int,
    learning_rate: float,
    best_macro_f1: float,
    args: argparse.Namespace,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    output_dir: Path,
    best_path: Path,
    class_to_idx: dict[str, int],
    idx_to_class: dict[int, str],
    history: list[dict],
) -> float:
    if epochs <= 0:
        return best_macro_f1

    set_trainable(model, args.backbone, phase)
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    stale_epochs = 0
    history_path = output_dir / "transfer_training_history.csv"

    for epoch in range(1, epochs + 1):
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
            "phase": phase,
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
            f"{phase} epoch {epoch:02d} "
            f"train_loss={row['train_loss']:.4f} "
            f"valid_acc={row['valid_accuracy']:.4f} "
            f"valid_macro_f1={row['valid_macro_f1']:.4f}",
            flush=True,
        )

        if valid_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = valid_metrics["macro_f1"]
            stale_epochs = 0
            save_checkpoint(
                best_path,
                model,
                args,
                class_to_idx,
                idx_to_class,
                best_macro_f1,
                phase,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping {phase} after {epoch} epochs.", flush=True)
                break

    return best_macro_f1


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
    model = build_model(args.backbone, len(labels), args.pretrained).to(device)
    best_path = output_dir / "best_transfer_model.pt"

    best_macro_f1 = -1.0
    if args.eval_only or args.resume:
        if not best_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {best_path}")
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        best_macro_f1 = float(checkpoint.get("valid_macro_f1", -1.0))
        print(f"Loaded checkpoint validation macro-F1: {best_macro_f1:.4f}", flush=True)

    if args.eval_only:
        metrics = []
        for split, loader in [("valid", valid_loader), ("test", test_loader)]:
            split_metrics = evaluate(model, loader, device)
            metrics.append({"model": f"transfer_{args.backbone}", "split": split, **split_metrics})
            save_split_report(model, loader, split, labels, output_dir, device)
        pd.DataFrame(metrics).to_csv(output_dir / "transfer_metrics.csv", index=False)
        print(f"Wrote transfer learning reports to: {output_dir}")
        return

    criterion = nn.CrossEntropyLoss(
        weight=class_weights(data, class_to_idx, args.class_weight_beta).to(device),
        label_smoothing=args.label_smoothing,
    )

    history_path = output_dir / "transfer_training_history.csv"
    history = []
    if args.resume and history_path.exists():
        history = pd.read_csv(history_path).to_dict("records")

    best_macro_f1 = run_phase(
        model=model,
        phase="head",
        epochs=0 if args.resume else args.head_epochs,
        learning_rate=args.head_lr,
        best_macro_f1=best_macro_f1,
        args=args,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        device=device,
        output_dir=output_dir,
        best_path=best_path,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        history=history,
    )
    best_macro_f1 = run_phase(
        model=model,
        phase="finetune",
        epochs=args.finetune_epochs,
        learning_rate=args.finetune_lr,
        best_macro_f1=best_macro_f1,
        args=args,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        device=device,
        output_dir=output_dir,
        best_path=best_path,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        history=history,
    )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    metrics = []
    for split, loader in [("valid", valid_loader), ("test", test_loader)]:
        split_metrics = evaluate(model, loader, device)
        metrics.append({"model": f"transfer_{args.backbone}", "split": split, **split_metrics})
        save_split_report(model, loader, split, labels, output_dir, device)

    pd.DataFrame(metrics).to_csv(output_dir / "transfer_metrics.csv", index=False)
    config = vars(args).copy()
    config["metadata_csv"] = str(config["metadata_csv"])
    config["output_root"] = str(config["output_root"])
    config["device"] = str(device)
    config["classes"] = labels
    write_json(output_dir / "transfer_config.json", config)
    print(f"Best validation macro-F1: {best_macro_f1:.4f}")
    print(f"Wrote transfer learning outputs to: {output_dir}")


if __name__ == "__main__":
    main()
