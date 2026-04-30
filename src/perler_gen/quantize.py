"""Color quantization to a fixed palette."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab

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


def _edge_weight_map(rgb: np.ndarray) -> np.ndarray:
    """Per-pixel weights >= 1; larger on strong luminance edges (numpy only)."""
    gray = (
        rgb[..., 0].astype(np.float32) * 0.299
        + rgb[..., 1].astype(np.float32) * 0.587
        + rgb[..., 2].astype(np.float32) * 0.114
    )
    h, w = gray.shape
    gx = np.zeros((h, w), dtype=np.float32)
    gy = np.zeros((h, w), dtype=np.float32)
    if w > 2:
        gx[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2]) * 0.5
    if h > 2:
        gy[1:-1, :] = np.abs(gray[2:, :] - gray[:-2, :]) * 0.5
    mag = np.maximum(gx, gy)
    base = float(np.mean(mag) + 1e-3)
    return 1.0 + mag / base


def _greedy_reduce(
    pixels: np.ndarray,
    palette: Palette,
    k: int,
    *,
    edge_weight: bool = False,
) -> Palette:
    """Match all pixels to the full palette, keep the top-k most-used palette colors."""
    flat = pixels.reshape(-1, 3)
    palette_rgb = palette.rgb_array.astype(np.uint8)
    palette_lab = _rgb_to_lab(palette_rgb.reshape(1, len(palette.colors), 3)).reshape(
        len(palette.colors), 3
    )
    pixels_lab = _rgb_to_lab(flat).reshape(-1, 3)
    dists = _lab_distance_matrix(pixels_lab, palette_lab)
    idx = np.argmin(dists, axis=1)
    m = len(palette.colors)
    if edge_weight:
        wmap = _edge_weight_map(pixels).reshape(-1)
        weighted = np.bincount(idx, weights=wmap, minlength=m)
        order = np.argsort(-weighted)
        top_k: list[int] = []
        for j in order:
            if weighted[int(j)] <= 0:
                break
            top_k.append(int(j))
            if len(top_k) >= k:
                break
        if not top_k:
            top_k = [int(np.argmax(weighted))]
        top_k_arr = np.array(top_k, dtype=np.int64)
    else:
        unique, counts = np.unique(idx, return_counts=True)
        take = min(k, len(unique))
        top_k_arr = unique[np.argsort(-counts)[:take]]
    top_k_sorted = np.sort(top_k_arr)
    colors = tuple(palette.colors[int(i)] for i in top_k_sorted)
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


def _majority_vote_filter(
    indices: np.ndarray,
    n_passes: int = 1,
    *,
    palette_lab: Optional[np.ndarray] = None,
    mode: str = "standard",
    conservative_de_max: float = 8.0,
) -> np.ndarray:
    """Replace isolated pixels with their neighborhood majority color.

    For each cell, if its color differs from the majority of its 8 neighbours,
    replace it with that majority color.  Repeat for ``n_passes`` iterations.

    ``mode='conservative'`` requires ``palette_lab`` (shape ``(M, 3)`` for the reduced
    palette). A flip only happens when the neighbor majority is a strong supermajority
    and the current vs majority palette colors are within ``conservative_de_max`` ΔE2000,
    so single-cell high-contrast detail is preserved.
    """
    result = indices.copy()
    h, w = result.shape
    use_conservative = mode == "conservative" and palette_lab is not None
    for _ in range(n_passes):
        new = result.copy()
        for y in range(h):
            for x in range(w):
                neighbors: list[int] = []
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            neighbors.append(int(result[ny, nx]))
                if not neighbors:
                    continue
                counts: dict[int, int] = {}
                for v in neighbors:
                    counts[v] = counts.get(v, 0) + 1
                majority = max(counts, key=lambda k: counts[k])
                cur = int(result[y, x])
                if majority == cur:
                    continue
                if counts[majority] <= len(neighbors) // 2:
                    continue
                if use_conservative:
                    need = (2 * len(neighbors) + 2) // 3
                    if counts[majority] < need:
                        continue
                    cur_lab = palette_lab[cur].reshape(1, 3)
                    maj_lab = palette_lab[majority].reshape(1, 3)
                    de = float(np.asarray(deltaE_ciede2000(cur_lab, maj_lab)).ravel()[0])
                    if de > conservative_de_max:
                        continue
                new[y, x] = majority
        result = new
    return result


def _small_component_filter(
    indices: np.ndarray,
    n_passes: int = 1,
    *,
    max_size: int = 2,
    min_neighbor_ratio: float = 0.6,
) -> np.ndarray:
    """Replace tiny color components when their boundary has a clear dominant color.

    This targets visual speckles after palette matching without flattening connected
    one-cell-wide strokes. Components are found with 4-neighbor connectivity; their
    surrounding replacement candidates are counted with 8-neighbor adjacency.
    """
    if max_size <= 0:
        return indices.copy()

    result = indices.copy()
    h, w = result.shape
    cardinal = ((-1, 0), (1, 0), (0, -1), (0, 1))
    around = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    for _ in range(max(0, n_passes)):
        visited = np.zeros((h, w), dtype=bool)
        new = result.copy()
        for sy in range(h):
            for sx in range(w):
                if visited[sy, sx]:
                    continue
                color = int(result[sy, sx])
                stack = [(sy, sx)]
                visited[sy, sx] = True
                component: list[tuple[int, int]] = []
                while stack:
                    y, x = stack.pop()
                    component.append((y, x))
                    for dy, dx in cardinal:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                            if int(result[ny, nx]) == color:
                                visited[ny, nx] = True
                                stack.append((ny, nx))

                if len(component) > max_size:
                    continue

                boundary: dict[int, int] = {}
                for y, x in component:
                    for dy, dx in around:
                        ny, nx = y + dy, x + dx
                        if not (0 <= ny < h and 0 <= nx < w):
                            continue
                        neighbor = int(result[ny, nx])
                        if neighbor == color:
                            continue
                        boundary[neighbor] = boundary.get(neighbor, 0) + 1
                if not boundary:
                    continue

                replacement = max(boundary, key=lambda k: boundary[k])
                total = sum(boundary.values())
                if boundary[replacement] / total < min_neighbor_ratio:
                    continue

                for y, x in component:
                    new[y, x] = replacement
        result = new
    return result


def quantize_to_palette(
    img: Image.Image,
    palette: Palette,
    max_colors: Optional[int] = None,
    dither: bool = False,
    post_smooth: int = 0,
    *,
    post_smooth_mode: str = "standard",
    reduce_palette_edge_weight: bool = False,
    conservative_de_max: float = 8.0,
    speckle_size: int = 2,
    speckle_neighbor_ratio: float = 0.6,
) -> QuantizeResult:
    """Quantize an image to the given palette using LAB Delta E 2000.

    Steps:
    1. Match all pixels to the full palette (Delta E 2000), count palette usage,
       and keep the top ``max_colors`` colors (greedy frequency reduction).
    2. Map every pixel to the reduced palette via LAB nearest neighbor,
       optionally with Floyd-Steinberg dithering.
    """
    pixels = np.array(img.convert("RGB"), dtype=np.uint8)

    if max_colors is not None and max_colors > 0 and max_colors < len(palette.colors):
        reduced_palette = _greedy_reduce(
            pixels,
            palette,
            max_colors,
            edge_weight=reduce_palette_edge_weight,
        )
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

    if post_smooth > 0 and post_smooth_mode == "speckle":
        indices = _small_component_filter(
            indices,
            n_passes=post_smooth,
            max_size=speckle_size,
            min_neighbor_ratio=speckle_neighbor_ratio,
        )
        rgb = palette_rgb[indices]
    elif post_smooth > 0:
        pal_for_vote = palette_lab if post_smooth_mode == "conservative" else None
        indices = _majority_vote_filter(
            indices,
            n_passes=post_smooth,
            palette_lab=pal_for_vote,
            mode=post_smooth_mode,
            conservative_de_max=conservative_de_max,
        )
        rgb = palette_rgb[indices]

    return QuantizeResult(indices=indices, rgb=rgb, palette=reduced_palette)
