"""Esporta un crop cellulare per ogni annotazione COCO valida."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from nmcd_common import DEFAULT_DATASET_ROOT, DEFAULT_OUTPUT_ROOT, SPLITS
from nmcd_common import category_lookup, clip_coco_bbox, ensure_dir, image_lookup, load_coco
from nmcd_common import slugify_label


METADATA_COLUMNS = [
    "split",
    "annotation_id",
    "image_id",
    "source_file",
    "category_id",
    "label",
    "crop_file",
    "source_width",
    "source_height",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
    "crop_width",
    "crop_height",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one crop per COCO annotation.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--padding-ratio", type=float, default=0.08)
    parser.add_argument("--padding-px", type=int, default=0)
    parser.add_argument("--min-size", type=int, default=8)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def export_split(args: argparse.Namespace, split: str) -> list[dict]:
    coco = load_coco(args.dataset_root, split)
    categories = category_lookup(coco)
    images = image_lookup(coco)
    rows: list[dict] = []

    for ann in tqdm(coco.get("annotations", []), desc=f"Exporting {split}"):
        category_id = int(ann["category_id"])
        label = categories.get(category_id, str(category_id))
        if label == "cells":
            continue

        image_info = images[int(ann["image_id"])]
        source_file = args.dataset_root / split / image_info["file_name"]
        if not source_file.exists():
            raise FileNotFoundError(f"Image not found: {source_file}")

        with Image.open(source_file) as src:
            image = ImageOps.exif_transpose(src).convert("RGB")
            source_width, source_height = image.size
            x1, y1, x2, y2 = clip_coco_bbox(
                ann["bbox"],
                source_width,
                source_height,
                padding_ratio=args.padding_ratio,
                padding_px=args.padding_px,
            )
            crop = image.crop((x1, y1, x2, y2)).copy()

        crop_width, crop_height = crop.size
        if crop_width < args.min_size or crop_height < args.min_size:
            continue

        label_dir = ensure_dir(args.output_root / "crops" / split / slugify_label(label))
        crop_name = f"{split}_img{int(ann['image_id']):06d}_ann{int(ann['id']):08d}.jpg"
        crop_path = label_dir / crop_name
        crop.save(crop_path, quality=args.jpeg_quality)

        bbox_x, bbox_y, bbox_w, bbox_h = [float(value) for value in ann["bbox"]]
        rows.append(
            {
                "split": split,
                "annotation_id": int(ann["id"]),
                "image_id": int(ann["image_id"]),
                "source_file": str(source_file.relative_to(ROOT)),
                "category_id": category_id,
                "label": label,
                "crop_file": str(crop_path.relative_to(ROOT)),
                "source_width": source_width,
                "source_height": source_height,
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
                "crop_x1": x1,
                "crop_y1": y1,
                "crop_x2": x2,
                "crop_y2": y2,
                "crop_width": crop_width,
                "crop_height": crop_height,
            }
        )

    return rows


def main() -> None:
    args = parse_args()
    metadata_dir = ensure_dir(args.output_root / "metadata")
    all_rows: list[dict] = []

    for split in SPLITS:
        all_rows.extend(export_split(args, split))

    metadata_path = metadata_dir / "cell_crops.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Exported {len(all_rows)} cell crops")
    print(f"Wrote metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
