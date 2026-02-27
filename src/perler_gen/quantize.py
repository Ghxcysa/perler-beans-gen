"""Color quantization to a fixed palette."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab
from sklearn.cluster import KMeans

from .palette import Palette


@dataclass(frozen=True)
class QuantizeResult:
    indices: np.ndarray  # shape (H, W) with palette indices
    rgb: np.ndarray      # shape (H, W, 3) quantized RGB
    palette: Palette


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB array [..., 3] to LAB float array."""
    rgb_float = rgb.astype(np.float32) / 255.0
    return rgb2lab(rgb_float)


def _lab_distance_matrix(pixels_lab: np.ndarray, palette_lab: np.ndarray) -> np.ndarray:
    """Compute Delta E 2000 distance matrix of shape (N_pixels, N_palette)."""
    n = pixels_lab.shape[0]
    m = palette_lab.shape[0]
    # Expand dims for broadcasting: (N, 1, 3) vs (1, M, 3)
    p = pixels_lab[:, None, :]   # (N, 1, 3)
    q = palette_lab[None, :, :]  # (1, M, 3)
    return deltaE_ciede2000(p, q)  # (N, M)


def _kmeans_reduce(pixels: np.ndarray, palette: Palette, k: int) -> Palette:
    """Use K-Means to find k representative colors from pixels, then snap each
    centroid to the nearest palette entry via Delta E 2000."""
    flat = pixels.reshape(-1, 3).astype(np.float32)
    # Cap k by the number of distinct colors actually present
    n_unique = len(np.unique(flat.view(np.dtype((np.void, flat.dtype.itemsize * flat.shape[1])))))
    actual_k = max(1, min(k, n_unique))

    km = KMeans(n_clusters=actual_k, n_init=10, random_state=0)
    km.fit(flat)
    centroids_rgb = np.clip(km.cluster_centers_, 0, 255).astype(np.uint8)  # (K, 3)
    centroids_lab = _rgb_to_lab(centroids_rgb.reshape(1, actual_k, 3)).reshape(actual_k, 3)

    palette_rgb = palette.rgb_array  # (M, 3)
    palette_lab = _rgb_to_lab(palette_rgb.reshape(1, len(palette.colors), 3)).reshape(len(palette.colors), 3)

    dists = _lab_distance_matrix(centroids_lab, palette_lab)  # (K, M)
    best_pal_idx = np.argmin(dists, axis=1)  # (K,)

    # Deduplicate while preserving order
    seen: set[int] = set()
    chosen: list[int] = []
    for i in best_pal_idx:
        if int(i) not in seen:
            seen.add(int(i))
            chosen.append(int(i))

    colors = tuple(palette.colors[i] for i in chosen)
    return Palette(name=f"{palette.name} Top {len(colors)}", colors=colors)


def _floyd_steinberg_dither(
    pixels: np.ndarray,
    palette_rgb: np.ndarray,
    palette_lab: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Floyd-Steinberg error diffusion dithering.

    Parameters
    ----------
    pixels:      uint8 array of shape (H, W, 3)
    palette_rgb: uint8 array of shape (M, 3)
    palette_lab: float array of shape (M, 3)

    Returns
    -------
    (indices, rgb): int32 (H, W) and uint8 (H, W, 3)
    """
    h, w, _ = pixels.shape
    m = palette_rgb.shape[0]

    # Work in float to accumulate error
    buf = pixels.astype(np.float32)
    indices = np.zeros((h, w), dtype=np.int32)
    result_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            old_pixel = np.clip(buf[y, x], 0, 255)
            # Convert single pixel to LAB
            old_lab = _rgb_to_lab(old_pixel.reshape(1, 1, 3)).reshape(3)
            # Find nearest palette color
            dists = deltaE_ciede2000(old_lab[None, :], palette_lab)  # (M,)
            idx = int(np.argmin(dists))
            indices[y, x] = idx
            new_pixel = palette_rgb[idx].astype(np.float32)
            result_rgb[y, x] = palette_rgb[idx]

            # Quantization error
            err = old_pixel - new_pixel

            # Diffuse error to neighbors
            if x + 1 < w:
                buf[y, x + 1] += err * (7 / 16)
            if y + 1 < h:
                if x - 1 >= 0:
                    buf[y + 1, x - 1] += err * (3 / 16)
                buf[y + 1, x] += err * (5 / 16)
                if x + 1 < w:
                    buf[y + 1, x + 1] += err * (1 / 16)

    return indices, result_rgb


def _direct_lab_match(
    pixels: np.ndarray,
    palette_rgb: np.ndarray,
    palette_lab: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor matching in LAB space (no dithering)."""
    h, w, _ = pixels.shape
    pixels_lab = _rgb_to_lab(pixels).reshape(-1, 3)
    dists = _lab_distance_matrix(pixels_lab, palette_lab)  # (N, M)
    idx = np.argmin(dists, axis=1).astype(np.int32)
    quant_rgb = palette_rgb[idx].reshape(h, w, 3)
    return idx.reshape(h, w), quant_rgb


def quantize_to_palette(
    img: Image.Image,
    palette: Palette,
    max_colors: Optional[int] = None,
    dither: bool = False,
) -> QuantizeResult:
    """Quantize an image to the given palette using LAB Delta E 2000.

    Steps:
    1. K-Means cluster the image to ``max_colors`` representative colors.
    2. Snap each centroid to the nearest palette entry (Delta E 2000).
    3. Map every pixel to the reduced palette via LAB nearest neighbor,
       optionally with Floyd-Steinberg dithering.
    """
    pixels = np.array(img.convert("RGB"), dtype=np.uint8)

    if max_colors is not None and max_colors > 0 and max_colors < len(palette.colors):
        reduced_palette = _kmeans_reduce(pixels, palette, max_colors)
    else:
        reduced_palette = palette

    palette_rgb = reduced_palette.rgb_array.astype(np.uint8)  # (M, 3)
    palette_lab = _rgb_to_lab(palette_rgb.reshape(1, len(reduced_palette.colors), 3)).reshape(
        len(reduced_palette.colors), 3
    )

    if dither:
        indices, rgb = _floyd_steinberg_dither(pixels, palette_rgb, palette_lab)
    else:
        indices, rgb = _direct_lab_match(pixels, palette_rgb, palette_lab)

    return QuantizeResult(indices=indices, rgb=rgb, palette=reduced_palette)
