"""Color quantization to a fixed palette."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from .palette import Palette


@dataclass(frozen=True)
class QuantizeResult:
    indices: np.ndarray  # shape (H, W) with palette indices
    rgb: np.ndarray      # shape (H, W, 3) quantized RGB
    palette: Palette


def _quantize_with_palette(pixels: np.ndarray, palette: Palette) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, rgb) given pixels as (H, W, 3)."""
    h, w, _ = pixels.shape
    flat = pixels.reshape(-1, 3).astype(np.int16)
    pal = palette.rgb_array.astype(np.int16)
    # Compute squared distances: (N, 3) vs (M, 3)
    # Result shape: (M, N)
    diffs = flat[:, None, :] - pal[None, :, :]
    dists = np.sum(diffs * diffs, axis=2)
    idx = np.argmin(dists, axis=1).astype(np.int32)
    quant_rgb = pal[idx].reshape(h, w, 3).astype(np.uint8)
    return idx.reshape(h, w), quant_rgb


def _top_k_palette(indices: np.ndarray, palette: Palette, k: int) -> Palette:
    counts = np.bincount(indices.flatten(), minlength=len(palette.colors))
    # Sort by count desc, then by palette index asc for determinism
    order = np.lexsort((np.arange(len(counts)), -counts))
    top_indices = [i for i in order if counts[i] > 0][:k]
    # If k > number of used colors, still include unused colors to reach k? No.
    if not top_indices:
        top_indices = [0]
    colors = tuple(palette.colors[i] for i in top_indices)
    return Palette(name=f"{palette.name} Top {len(colors)}", colors=colors)


def quantize_to_palette(
    img: Image.Image,
    palette: Palette,
    max_colors: Optional[int] = None,
) -> QuantizeResult:
    """Quantize an image to the given palette using nearest neighbor in RGB."""
    pixels = np.array(img.convert("RGB"), dtype=np.uint8)
    indices, rgb = _quantize_with_palette(pixels, palette)
    if max_colors is not None and max_colors > 0 and max_colors < len(palette.colors):
        reduced_palette = _top_k_palette(indices, palette, max_colors)
        indices, rgb = _quantize_with_palette(pixels, reduced_palette)
        palette = reduced_palette
    return QuantizeResult(indices=indices, rgb=rgb, palette=palette)
