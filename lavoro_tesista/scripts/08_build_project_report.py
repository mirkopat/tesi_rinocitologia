"""Genera un report PDF semplice a partire da un documento Markdown."""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "docs" / "aggiornamento_progetto.md"
DEFAULT_OUTPUT = ROOT / "docs" / "aggiornamento_progetto.pdf"

PAGE_WIDTH = 8.27
PAGE_HEIGHT = 11.69
LEFT = 0.72
RIGHT = 0.72
TOP = 0.72
BOTTOM = 0.58
IMAGE_RE = re.compile(r"^!\[(?P<caption>.*?)\]\((?P<path>.*?)\)$")


class SimplePdfRenderer:
    def __init__(self, output_path: Path, base_dir: Path) -> None:
        self.output_path = output_path
        self.base_dir = base_dir
        self.pdf = PdfPages(output_path)
        self.page_number = 0
        self.fig = None
        self.ax = None
        self.y = 1.0
        self.new_page()

    def close(self) -> None:
        self.finish_page()
        self.pdf.close()

    def new_page(self) -> None:
        self.fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.axis("off")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.page_number += 1
        self.y = 1.0 - TOP / PAGE_HEIGHT

    def finish_page(self) -> None:
        if self.fig is None or self.ax is None:
            return
        self.ax.text(
            0.5,
            BOTTOM / PAGE_HEIGHT * 0.45,
            str(self.page_number),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#555555",
        )
        self.pdf.savefig(self.fig)
        plt.close(self.fig)
        self.fig = None
        self.ax = None

    def available_bottom(self) -> float:
        return BOTTOM / PAGE_HEIGHT

    def line_step(self, font_size: float, factor: float = 1.35) -> float:
        return (font_size * factor / 72.0) / PAGE_HEIGHT

    def ensure_space(self, font_size: float, lines: int = 1, extra: float = 0.0) -> None:
        needed = self.line_step(font_size) * lines + extra
        if self.y - needed < self.available_bottom():
            self.finish_page()
            self.new_page()

    def emit_line(
        self,
        text: str,
        *,
        font_size: float = 10.0,
        weight: str = "normal",
        family: str = "DejaVu Sans",
        indent: float = 0.0,
        color: str = "#111111",
        step_factor: float = 1.35,
    ) -> None:
        self.ensure_space(font_size)
        assert self.ax is not None
        self.ax.text(
            (LEFT + indent) / PAGE_WIDTH,
            self.y,
            text,
            ha="left",
            va="top",
            fontsize=font_size,
            fontweight=weight,
            family=family,
            color=color,
        )
        self.y -= self.line_step(font_size, step_factor)

    def blank(self, amount: float = 0.08) -> None:
        self.y -= amount / PAGE_HEIGHT
        if self.y < self.available_bottom():
            self.finish_page()
            self.new_page()

    def emit_wrapped(
        self,
        text: str,
        *,
        width: int = 96,
        font_size: float = 10.0,
        weight: str = "normal",
        family: str = "DejaVu Sans",
        indent: float = 0.0,
        subsequent_indent: str = "",
    ) -> None:
        lines = textwrap.wrap(
            text,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
            subsequent_indent=subsequent_indent,
        )
        if not lines:
            self.blank()
            return
        self.ensure_space(font_size, lines=len(lines))
        for line in lines:
            self.emit_line(
                line,
                font_size=font_size,
                weight=weight,
                family=family,
                indent=indent,
            )

    def resolve_image_path(self, raw_path: str) -> Path:
        path = Path(raw_path.strip())
        if path.is_absolute():
            return path

        for candidate in (self.base_dir / path, ROOT / path):
            if candidate.exists():
                return candidate
        return self.base_dir / path

    def emit_image(self, raw_path: str, caption: str) -> None:
        image_path = self.resolve_image_path(raw_path)
        if not image_path.exists():
            self.emit_wrapped(f"[Immagine non trovata: {raw_path}]", font_size=9, weight="bold")
            return

        image = plt.imread(image_path)
        image_height_px, image_width_px = image.shape[:2]
        aspect_ratio = image_height_px / max(image_width_px, 1)

        target_width = PAGE_WIDTH - LEFT - RIGHT
        target_height = target_width * aspect_ratio
        max_height = PAGE_HEIGHT - TOP - BOTTOM - 0.55
        if target_height > max_height:
            target_height = max_height
            target_width = target_height / max(aspect_ratio, 0.01)

        caption_height = self.line_step(9.2) if caption else 0
        needed = caption_height + (target_height / PAGE_HEIGHT) + (0.18 / PAGE_HEIGHT)
        if self.y - needed < self.available_bottom():
            self.finish_page()
            self.new_page()

        if caption:
            self.emit_wrapped(caption, width=92, font_size=9.2, weight="bold")
            self.blank(0.04)

        assert self.ax is not None
        image_width_norm = target_width / PAGE_WIDTH
        image_height_norm = target_height / PAGE_HEIGHT
        x0 = 0.5 - image_width_norm / 2
        x1 = 0.5 + image_width_norm / 2
        y1 = self.y
        y0 = y1 - image_height_norm
        self.ax.imshow(image, extent=(x0, x1, y0, y1), aspect="auto")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.y = y0 - 0.14 / PAGE_HEIGHT

    def render_markdown(self, markdown: str) -> None:
        in_code = False
        previous_was_table = False

        for raw_line in markdown.splitlines():
            line = raw_line.rstrip()

            if line.strip().startswith("```"):
                in_code = not in_code
                if not in_code:
                    self.blank(0.06)
                continue

            if in_code:
                self.emit_wrapped(
                    line,
                    width=118,
                    font_size=8.2,
                    family="DejaVu Sans Mono",
                    indent=0.18,
                )
                continue

            stripped = line.strip()
            if not stripped:
                self.blank(0.09 if not previous_was_table else 0.05)
                previous_was_table = False
                continue

            image_match = IMAGE_RE.match(stripped)
            if image_match:
                self.emit_image(image_match.group("path"), image_match.group("caption"))
                previous_was_table = False
                continue

            if stripped.startswith("# "):
                self.blank(0.08)
                self.emit_wrapped(stripped[2:], width=70, font_size=17, weight="bold")
                self.blank(0.14)
                previous_was_table = False
                continue

            if stripped.startswith("## "):
                self.blank(0.13)
                self.emit_wrapped(stripped[3:], width=78, font_size=13.2, weight="bold")
                self.blank(0.05)
                previous_was_table = False
                continue

            if stripped.startswith("### "):
                self.blank(0.09)
                self.emit_wrapped(stripped[4:], width=86, font_size=11.3, weight="bold")
                self.blank(0.03)
                previous_was_table = False
                continue

            if stripped.startswith("|"):
                self.emit_wrapped(
                    stripped,
                    width=125,
                    font_size=7.3,
                    indent=0.02,
                    family="DejaVu Sans Mono",
                )
                previous_was_table = True
                continue

            if stripped.startswith("- "):
                self.emit_wrapped(
                    "- " + stripped[2:],
                    width=92,
                    font_size=10,
                    indent=0.18,
                    subsequent_indent="  ",
                )
                previous_was_table = False
                continue

            self.emit_wrapped(stripped, width=96, font_size=10)
            previous_was_table = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the thesis project update PDF.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source
    output = args.output
    if not source.exists():
        raise FileNotFoundError(f"Report source not found: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    renderer = SimplePdfRenderer(output, source.parent)
    try:
        renderer.render_markdown(source.read_text(encoding="utf-8"))
    finally:
        renderer.close()

    print(f"Wrote PDF report to: {output}")


if __name__ == "__main__":
    main()
