"""PDF export for printable patterns."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Iterable

import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from .counts import compute_counts
from .quantize import QuantizeResult
from .step_planner import Step
from .utils import index_to_symbol


@dataclass(frozen=True)
class PatternMeta:
    title: str
    grid_w: int
    grid_h: int
    palette_name: str


def _make_preview_image(quantized_rgb: np.ndarray, max_size: int = 400) -> Image.Image:
    img = Image.fromarray(quantized_rgb, mode="RGB")
    scale = min(max_size / img.width, max_size / img.height, 1.0)
    if scale < 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), resample=Image.NEAREST)
    return img


def _draw_grid(c: canvas.Canvas, origin_x: float, origin_y: float, cell: float, w: int, h: int) -> None:
    c.setStrokeColorRGB(0.8, 0.8, 0.8)
    for i in range(w + 1):
        x = origin_x + i * cell
        c.line(x, origin_y, x, origin_y + h * cell)
    for j in range(h + 1):
        y = origin_y + j * cell
        c.line(origin_x, y, origin_x + w * cell, y)


def _draw_axes(c: canvas.Canvas, origin_x: float, origin_y: float, cell: float, w: int, h: int) -> None:
    c.setFont("Helvetica", 6)
    c.setFillColorRGB(0, 0, 0)
    # Column numbers at top
    for col in range(w):
        x = origin_x + col * cell + cell * 0.3
        y = origin_y + h * cell + 2
        c.drawString(x, y, str(col + 1))
    # Row numbers at left
    for row in range(h):
        x = origin_x - 10
        y = origin_y + (h - 1 - row) * cell + cell * 0.25
        c.drawString(x, y, str(row + 1))


def _draw_symbols(
    c: canvas.Canvas,
    origin_x: float,
    origin_y: float,
    cell: float,
    indices: np.ndarray,
    mask: np.ndarray,
    symbols: list[str],
) -> None:
    font_size = max(6, min(12, int(cell * 0.6)))
    c.setFont("Helvetica", font_size)
    c.setFillColorRGB(0, 0, 0)
    h, w = indices.shape
    for row in range(h):
        for col in range(w):
            if not mask[row, col]:
                continue
            symbol = symbols[int(indices[row, col])]
            x = origin_x + col * cell + cell * 0.25
            y = origin_y + (h - 1 - row) * cell + cell * 0.2
            c.drawString(x, y, symbol)


def _legend_entries(quantized: QuantizeResult) -> list[tuple[str, str, str, int]]:
    counts = compute_counts(quantized.indices, quantized.palette)
    count_map = {entry.code: entry.count for entry in counts}
    entries: list[tuple[str, str, str, int]] = []
    for idx, color in enumerate(quantized.palette.colors):
        count = count_map.get(color.code, 0)
        if count <= 0:
            continue
        symbol = index_to_symbol(idx)
        entries.append((symbol, color.code, color.name, count))
    return entries


def write_pattern_pdf(out_path: str, meta: PatternMeta, quantized: QuantizeResult, steps: Iterable[Step]) -> None:
    page_w, page_h = letter
    c = canvas.Canvas(out_path, pagesize=letter)

    # Cover page
    c.setFont("Helvetica-Bold", 18)
    c.drawString(0.75 * inch, page_h - 1.0 * inch, meta.title)
    c.setFont("Helvetica", 12)
    c.drawString(0.75 * inch, page_h - 1.4 * inch, f"Grid: {meta.grid_w} x {meta.grid_h}")
    entries = _legend_entries(quantized)
    c.drawString(0.75 * inch, page_h - 1.7 * inch, f"Colors used: {len(entries)}")
    preview = _make_preview_image(quantized.rgb)
    bio = BytesIO()
    preview.save(bio, format="PNG")
    bio.seek(0)
    img_reader = ImageReader(bio)
    img_x = 0.75 * inch
    img_y = page_h - 5.0 * inch
    c.drawImage(img_reader, img_x, img_y, width=3.5 * inch, preserveAspectRatio=True, mask='auto')
    c.showPage()

    # Legend page
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.75 * inch, page_h - 1.0 * inch, "Legend")
    c.setFont("Helvetica", 10)
    start_y = page_h - 1.5 * inch
    line_h = 14
    col_x = [0.75 * inch, 2.0 * inch, 3.2 * inch, 5.2 * inch]
    c.drawString(col_x[0], start_y, "Symbol")
    c.drawString(col_x[1], start_y, "Code")
    c.drawString(col_x[2], start_y, "Name")
    c.drawString(col_x[3], start_y, "Count")
    y = start_y - line_h
    for symbol, code, name, count in entries:
        c.drawString(col_x[0], y, symbol)
        c.drawString(col_x[1], y, code)
        c.drawString(col_x[2], y, name)
        c.drawString(col_x[3], y, str(count))
        y -= line_h
        if y < 1.0 * inch:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = page_h - 1.0 * inch
    c.showPage()

    # Step pages
    margin = 0.5 * inch
    grid_area_w = page_w - 2 * margin
    grid_area_h = page_h - 2.5 * inch
    cell = min(grid_area_w / meta.grid_w, grid_area_h / meta.grid_h)
    origin_x = margin
    origin_y = margin

    symbols = [index_to_symbol(i) for i in range(len(quantized.palette.colors))]
    for idx, step in enumerate(steps, start=1):
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, page_h - 0.75 * inch, f"Step {idx}: {step.name}")
        _draw_grid(c, origin_x, origin_y, cell, meta.grid_w, meta.grid_h)
        _draw_axes(c, origin_x, origin_y, cell, meta.grid_w, meta.grid_h)
        _draw_symbols(c, origin_x, origin_y, cell, quantized.indices, step.mask, symbols)
        c.showPage()

    c.save()
