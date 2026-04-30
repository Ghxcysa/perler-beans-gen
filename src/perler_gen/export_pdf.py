"""PDF export for printable patterns."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Iterable, Literal

import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from .counts import compute_counts
from .quantize import QuantizeResult
from .step_planner import Step
from .utils import index_to_number


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


def _draw_cells(
    c: canvas.Canvas,
    origin_x: float,
    origin_y: float,
    cell: float,
    indices: np.ndarray,
    mask: np.ndarray,
    palette_rgb: np.ndarray,
) -> None:
    h, w = indices.shape
    for row in range(h):
        for col in range(w):
            if not mask[row, col]:
                continue
            r, g, b = palette_rgb[int(indices[row, col])] / 255.0
            c.setFillColorRGB(float(r), float(g), float(b))
            x = origin_x + col * cell
            y = origin_y + (h - 1 - row) * cell
            c.rect(x, y, cell, cell, fill=1, stroke=0)


def _draw_cells_faded(
    c: canvas.Canvas,
    origin_x: float,
    origin_y: float,
    cell: float,
    indices: np.ndarray,
    mask: np.ndarray,
    palette_rgb: np.ndarray,
    alpha: float = 0.25,
) -> None:
    """Draw already-placed cells at reduced opacity as a placement reference."""
    c.saveState()
    c.setFillAlpha(alpha)
    h, w = indices.shape
    for row in range(h):
        for col in range(w):
            if not mask[row, col]:
                continue
            r, g, b = palette_rgb[int(indices[row, col])] / 255.0
            c.setFillColorRGB(float(r), float(g), float(b))
            x = origin_x + col * cell
            y = origin_y + (h - 1 - row) * cell
            c.rect(x, y, cell, cell, fill=1, stroke=0)
    c.restoreState()


def _draw_grid(
    c: canvas.Canvas,
    origin_x: float,
    origin_y: float,
    cell: float,
    w: int,
    h: int,
    interval: int = 5,
) -> None:
    # Minor grid lines — one per cell, very faint.
    c.setStrokeColorRGB(0.85, 0.85, 0.85)
    c.setLineWidth(0.15)
    for i in range(w + 1):
        x = origin_x + i * cell
        c.line(x, origin_y, x, origin_y + h * cell)
    for j in range(h + 1):
        y = origin_y + j * cell
        c.line(origin_x, y, origin_x + w * cell, y)

    # Major grid lines — every ``interval`` cells, darker and thicker for counting.
    if interval > 0:
        c.setStrokeColorRGB(0.55, 0.55, 0.55)
        c.setLineWidth(0.5)
        for i in range(0, w + 1, interval):
            x = origin_x + i * cell
            c.line(x, origin_y, x, origin_y + h * cell)
        for j in range(0, h + 1, interval):
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


def _legend_entries(quantized: QuantizeResult) -> list[tuple[str, str, str, int, tuple[int, int, int]]]:
    counts = compute_counts(quantized.indices, quantized.palette)
    count_map = {entry.code: entry.count for entry in counts}
    entries: list[tuple[str, str, str, int, tuple[int, int, int]]] = []
    for idx, color in enumerate(quantized.palette.colors):
        count = count_map.get(color.code, 0)
        if count <= 0:
            continue
        symbol = index_to_number(idx)
        entries.append((symbol, color.code, color.name, count, color.rgb))
    return entries


def _draw_legend_column_headers(
    c: canvas.Canvas,
    col_x: list[float],
    start_y: float,
) -> None:
    c.setFont("Helvetica-Bold", 10)
    c.drawString(col_x[0], start_y, "No.")
    c.drawString(col_x[1], start_y, "Code")
    c.drawString(col_x[2], start_y, "Name")
    c.drawString(col_x[3], start_y, "Count")


def _draw_legend_swatch(
    c: canvas.Canvas,
    x: float,
    baseline_y: float,
    rgb: tuple[int, int, int],
    size: float,
) -> None:
    """Draw a filled swatch with a light border so pale colors stay visible."""
    bottom = baseline_y - 0.35 * size
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    c.setFillColorRGB(r, g, b)
    c.setStrokeColorRGB(0.55, 0.55, 0.55)
    c.setLineWidth(0.25)
    c.rect(x, bottom, size, size, fill=1, stroke=1)


def write_pattern_pdf(
    out_path: str,
    meta: PatternMeta,
    quantized: QuantizeResult,
    steps: Iterable[Step],
    grid_interval: int = 5,
    step_orientation: Literal["portrait", "landscape"] = "landscape",
) -> None:
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
    img_y = page_h - 5.5 * inch
    c.drawImage(img_reader, img_x, img_y, width=4.25 * inch, preserveAspectRatio=True, mask="auto")
    c.showPage()

    # Legend page(s)
    margin_x = 0.75 * inch
    swatch_size = 11.0
    x_no = margin_x
    x_swatch = margin_x + 34
    x_code = x_swatch + swatch_size + 10
    x_name = 3.25 * inch
    x_count = 5.35 * inch
    col_x = [x_no, x_code, x_name, x_count]
    start_y = page_h - 1.5 * inch
    line_h = 16

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin_x, page_h - 1.0 * inch, "Legend")
    c.setFont("Helvetica", 10)
    _draw_legend_column_headers(c, col_x, start_y)
    y = start_y - line_h
    for symbol, code, name, count, rgb in entries:
        c.setFont("Helvetica", 10)
        c.drawString(col_x[0], y, symbol)
        _draw_legend_swatch(c, x_swatch, y, rgb, swatch_size)
        c.drawString(col_x[1], y, code)
        c.drawString(col_x[2], y, name)
        c.drawString(col_x[3], y, str(count))
        y -= line_h
        if y < 1.0 * inch:
            c.showPage()
            c.setFont("Helvetica-Bold", 16)
            c.drawString(margin_x, page_h - 1.0 * inch, "Legend (cont.)")
            c.setFont("Helvetica", 10)
            _draw_legend_column_headers(c, col_x, start_y)
            y = start_y - line_h
    c.showPage()

    # Step pages (optionally landscape for larger cells)
    if step_orientation == "landscape":
        step_pagesize = landscape(letter)
        c.setPageSize(step_pagesize)
        page_w, page_h = step_pagesize
    elif step_orientation == "portrait":
        page_w, page_h = letter
    else:
        raise ValueError(f"step_orientation must be 'portrait' or 'landscape', got {step_orientation!r}")

    margin = 0.5 * inch
    title_reserve = 2.0 * inch
    grid_area_w = page_w - 2 * margin
    grid_area_h = page_h - title_reserve
    cell = min(grid_area_w / meta.grid_w, grid_area_h / meta.grid_h)
    origin_x = margin
    origin_y = margin

    symbols = [index_to_number(i) for i in range(len(quantized.palette.colors))]
    placed_mask = np.zeros((meta.grid_h, meta.grid_w), dtype=bool)
    for idx, step in enumerate(steps, start=1):
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, page_h - 0.75 * inch, f"Step {idx}: {step.name}")
        # Draw previously placed cells faded as a placement reference.
        if placed_mask.any():
            _draw_cells_faded(c, origin_x, origin_y, cell,
                              quantized.indices, placed_mask, quantized.palette.rgb_array)
        # Draw current step cells in full color.
        _draw_cells(c, origin_x, origin_y, cell, quantized.indices, step.mask, quantized.palette.rgb_array)
        _draw_grid(c, origin_x, origin_y, cell, meta.grid_w, meta.grid_h, interval=grid_interval)
        _draw_axes(c, origin_x, origin_y, cell, meta.grid_w, meta.grid_h)
        _draw_symbols(c, origin_x, origin_y, cell, quantized.indices, step.mask, symbols)
        c.showPage()
        placed_mask = placed_mask | step.mask

    c.save()
