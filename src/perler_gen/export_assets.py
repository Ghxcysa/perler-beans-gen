"""Export preview images and bead lists."""
from __future__ import annotations

import csv
from typing import Iterable

import numpy as np
from PIL import Image

from .counts import CountEntry


def write_preview_png(out_path: str, quantized_rgb: np.ndarray, scale: int = 10) -> None:
    """Write a preview PNG of the quantized grid."""
    if quantized_rgb.ndim != 3 or quantized_rgb.shape[2] != 3:
        raise ValueError("quantized_rgb must be HxWx3.")
    img = Image.fromarray(quantized_rgb, mode="RGB")
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), resample=Image.NEAREST)
    img.save(out_path)


def write_bead_list_csv(out_path: str, counts: Iterable[CountEntry]) -> None:
    """Write bead counts as CSV."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["color_code", "color_name", "count"])
        for entry in counts:
            writer.writerow([entry.code, entry.name, entry.count])


def write_svg(out_path: str, quantized_rgb: np.ndarray, cell_size: int = 10) -> None:
    """Write a simple SVG grid preview."""
    h, w, _ = quantized_rgb.shape
    width_px = w * cell_size
    height_px = h * cell_size
    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width_px}' height='{height_px}' viewBox='0 0 {width_px} {height_px}'>"
    ]
    for y in range(h):
        for x in range(w):
            r, g, b = [int(v) for v in quantized_rgb[y, x]]
            color = f"#{r:02x}{g:02x}{b:02x}"
            lines.append(
                f"<rect x='{x * cell_size}' y='{y * cell_size}' width='{cell_size}' height='{cell_size}' fill='{color}' />"
            )
    lines.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
