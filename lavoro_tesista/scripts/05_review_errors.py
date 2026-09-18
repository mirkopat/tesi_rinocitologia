"""Esporta report e gallerie per analizzare gli errori del modello migliore."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.metrics import precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from nmcd_common import DEFAULT_OUTPUT_ROOT, ensure_dir, slugify_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review errors from the best feature baseline.")
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "features" / "handcrafted_features.csv",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "models" / "best_feature_baseline.joblib",
    )
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "error_analysis")
    parser.add_argument("--samples-per-confusion", type=int, default=12)
    parser.add_argument("--max-confusions", type=int, default=12)
    return parser.parse_args()


def prediction_scores(model, x: pd.DataFrame, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(model, "decision_function"):
        return np.full(len(pred), np.nan), np.full(len(pred), np.nan)

    scores = np.asarray(model.decision_function(x))
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])

    classes = np.asarray(model.named_steps["model"].classes_)
    pred_indices = np.array([np.where(classes == label)[0][0] for label in pred])
    pred_score = scores[np.arange(len(pred)), pred_indices]

    sorted_scores = np.sort(scores, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2]
    return pred_score, margins


def save_contact_sheet(rows: pd.DataFrame, output_path: Path, tile_size: int = 128) -> None:
    if rows.empty:
        return

    cols = 4
    rows_count = int(np.ceil(len(rows) / cols))
    label_height = 34
    sheet = Image.new("RGB", (cols * tile_size, rows_count * (tile_size + label_height)), "white")
    draw = ImageDraw.Draw(sheet)

    for idx, record in enumerate(rows.to_dict("records")):
        crop_path = ROOT / record["crop_file"]
        if not crop_path.exists():
            continue

        image = Image.open(crop_path).convert("RGB")
        image.thumbnail((tile_size, tile_size))

        x = (idx % cols) * tile_size
        y = (idx // cols) * (tile_size + label_height)
        paste_x = x + (tile_size - image.width) // 2
        paste_y = y + (tile_size - image.height) // 2
        sheet.paste(image, (paste_x, paste_y))

        annotation = str(record["annotation_id"])
        draw.text((x + 4, y + tile_size + 2), f"ann {annotation}", fill=(0, 0, 0))
        draw.text((x + 4, y + tile_size + 17), f"margin {record['prediction_margin']:.2f}", fill=(0, 0, 0))

    ensure_dir(output_path.parent)
    sheet.save(output_path)


def copy_error_samples(rows: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)
    for idx, record in enumerate(rows.to_dict("records"), start=1):
        source = ROOT / record["crop_file"]
        if not source.exists():
            continue
        destination = output_dir / f"{idx:02d}_ann{int(record['annotation_id']):08d}_{source.name}"
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    if not args.features_csv.exists():
        raise FileNotFoundError(f"Features not found: {args.features_csv}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    output_dir = ensure_dir(args.output_root / args.split)
    model_bundle = joblib.load(args.model_path)
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]

    data = pd.read_csv(args.features_csv)
    split_data = data[data["split"] == args.split].copy()
    if split_data.empty:
        raise ValueError(f"No rows found for split: {args.split}")

    pred = model.predict(split_data[feature_columns])
    pred_score, pred_margin = prediction_scores(model, split_data[feature_columns], pred)

    predictions = split_data[
        ["split", "annotation_id", "image_id", "source_file", "label", "crop_file"]
    ].copy()
    predictions["predicted_label"] = pred
    predictions["is_correct"] = predictions["label"] == predictions["predicted_label"]
    predictions["prediction_score"] = pred_score
    predictions["prediction_margin"] = pred_margin
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    mistakes = predictions[~predictions["is_correct"]].copy()
    mistakes = mistakes.sort_values(["label", "predicted_label", "prediction_margin"])
    mistakes.to_csv(output_dir / "mistakes.csv", index=False)

    confusion_pairs = (
        mistakes.groupby(["label", "predicted_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    confusion_pairs.to_csv(output_dir / "confusion_pairs.csv", index=False)

    labels = sorted(split_data["label"].unique())
    precision, recall, f1, support = precision_recall_fscore_support(
        predictions["label"],
        predictions["predicted_label"],
        labels=labels,
        zero_division=0,
    )
    per_class = pd.DataFrame(
        {
            "label": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    ).sort_values(["f1", "support"])
    per_class.to_csv(output_dir / "per_class_metrics.csv", index=False)

    gallery_dir = ensure_dir(output_dir / "galleries")
    sample_root = ensure_dir(output_dir / "sample_crops")
    for pair in confusion_pairs.head(args.max_confusions).to_dict("records"):
        true_label = pair["label"]
        predicted_label = pair["predicted_label"]
        pair_rows = mistakes[
            (mistakes["label"] == true_label) & (mistakes["predicted_label"] == predicted_label)
        ].head(args.samples_per_confusion)

        pair_name = f"{slugify_label(true_label)}__as__{slugify_label(predicted_label)}"
        save_contact_sheet(pair_rows, gallery_dir / f"{pair_name}.jpg")
        copy_error_samples(pair_rows, sample_root / pair_name)

    print(f"Rows in {args.split}: {len(predictions)}")
    print(f"Mistakes: {len(mistakes)}")
    print(f"Accuracy: {predictions['is_correct'].mean():.4f}")
    print(f"Wrote error analysis to: {output_dir}")


if __name__ == "__main__":
    main()
