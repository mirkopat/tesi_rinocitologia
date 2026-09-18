"""Analizza il dataset NMCD in formato COCO e salva statistiche riassuntive."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from nmcd_common import DEFAULT_DATASET_ROOT, DEFAULT_OUTPUT_ROOT, SPLITS
from nmcd_common import category_lookup, ensure_dir, load_coco, write_json


def summarize_split(dataset_root: Path, split: str) -> dict:
    coco = load_coco(dataset_root, split)
    categories = category_lookup(coco)
    counts = Counter(int(ann["category_id"]) for ann in coco.get("annotations", []))
    widths = [float(ann["bbox"][2]) for ann in coco.get("annotations", [])]
    heights = [float(ann["bbox"][3]) for ann in coco.get("annotations", [])]

    return {
        "split": split,
        "images": len(coco.get("images", [])),
        "annotations": len(coco.get("annotations", [])),
        "categories": categories,
        "category_counts": {categories.get(k, str(k)): v for k, v in counts.items()},
        "bbox_width": describe(widths),
        "bbox_height": describe(heights),
    }


def describe(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def write_category_csv(path: Path, summaries: list[dict]) -> None:
    ensure_dir(path.parent)
    fieldnames = ["split", "category", "annotations"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            for category, count in sorted(summary["category_counts"].items()):
                writer.writerow(
                    {
                        "split": summary["split"],
                        "category": category,
                        "annotations": count,
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the NMCD COCO dataset.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [summarize_split(args.dataset_root, split) for split in SPLITS]
    report_dir = ensure_dir(args.output_root / "reports")

    write_json(report_dir / "dataset_summary.json", summaries)
    write_category_csv(report_dir / "category_counts.csv", summaries)

    for summary in summaries:
        print(
            f"{summary['split']}: "
            f"{summary['images']} images, "
            f"{summary['annotations']} annotations"
        )
        for category, count in sorted(
            summary["category_counts"].items(), key=lambda item: item[1], reverse=True
        ):
            print(f"  {category}: {count}")
        print()

    print(f"Wrote reports to: {report_dir}")


if __name__ == "__main__":
    main()
