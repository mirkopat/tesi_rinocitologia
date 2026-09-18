"""Utility condivise per la pipeline sperimentale sul dataset NMCD."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "Dataset Rinocitologia" / "NMCD.coco"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
SPLITS = ("train", "valid", "test")


def load_coco(dataset_root: Path, split: str) -> dict[str, Any]:
    ann_path = dataset_root / split / "_annotations.coco.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")
    return json.loads(ann_path.read_text(encoding="utf-8"))


def category_lookup(coco: dict[str, Any]) -> dict[int, str]:
    return {int(cat["id"]): str(cat["name"]) for cat in coco.get("categories", [])}


def image_lookup(coco: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(img["id"]): img for img in coco.get("images", [])}


def slugify_label(label: str) -> str:
    text = label.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def clip_coco_bbox(
    bbox: list[float] | tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    padding_ratio: float = 0.0,
    padding_px: int = 0,
) -> tuple[int, int, int, int]:
    x, y, width, height = [float(v) for v in bbox]
    pad_x = padding_px + width * padding_ratio
    pad_y = padding_px + height * padding_ratio

    x1 = max(0, math.floor(x - pad_x))
    y1 = max(0, math.floor(y - pad_y))
    x2 = min(image_width, math.ceil(x + width + pad_x))
    y2 = min(image_height, math.ceil(y + height + pad_y))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid clipped bbox: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
