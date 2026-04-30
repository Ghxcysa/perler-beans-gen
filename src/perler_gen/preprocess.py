"""Image loading and preprocessing utilities."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from skimage.color import rgb2lab


def load_image(path: str, alpha_background: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Load an image from disk, normalize orientation/alpha, and convert to RGB."""
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (*alpha_background, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return img.convert("RGB")


def denoise_image(img: Image.Image, passes: int = 0) -> Image.Image:
    """Apply light median denoising before downsampling.

    Median filtering removes salt-and-pepper and compression speckles while keeping hard
    edges sharper than a blur. Multiple passes are intentionally capped by the caller.
    """
    result = img
    for _ in range(max(0, passes)):
        result = result.filter(ImageFilter.MedianFilter(size=3))
    return result


def smooth_image(img: Image.Image, radius: float) -> Image.Image:
    """Apply Gaussian blur to reduce fine noise before downsampling."""
    if radius <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _cell_bounds(axis_len: int, count: int, i: int) -> tuple[int, int]:
    """Half-open integer bounds for cell ``i`` along an axis of length ``axis_len``."""
    start = i * axis_len // count
    end = (i + 1) * axis_len // count
    if end <= start:
        end = start + 1
    return start, end


def resample_to_grid_cell_dominant_lab(img: Image.Image, w: int, h: int) -> Image.Image:
    """For each output cell, pick the dominant color by coarse LAB-bin mode, then mean RGB.

    Each target cell maps to the corresponding rectangle in the source image. Pixels in
    that rectangle are binned in LAB (L/5, a/4, b/4), the densest bin wins, and the mean
    RGB of pixels in that bin becomes the cell color.
    """
    if w <= 0 or h <= 0:
        raise ValueError("Grid dimensions must be positive.")
    src = np.asarray(img.convert("RGB"), dtype=np.uint8)
    sh, sw = src.shape[:2]
    out = np.empty((h, w, 3), dtype=np.uint8)
    rgb_f = src.astype(np.float32) / 255.0
    lab = rgb2lab(rgb_f)

    for y in range(h):
        y0, y1 = _cell_bounds(sh, h, y)
        row_lab = lab[y0:y1, :, :]
        row_rgb = src[y0:y1]
        for x in range(w):
            x0, x1 = _cell_bounds(sw, w, x)
            block_lab = row_lab[:, x0:x1].reshape(-1, 3)
            block_rgb = row_rgb[:, x0:x1].reshape(-1, 3)
            if block_lab.size == 0:
                out[y, x] = src[min(y0, sh - 1), min(x0, sw - 1)]
                continue
            l_bin = np.clip((block_lab[:, 0] // 5.0).astype(np.int32), 0, 100)
            a_bin = np.clip((block_lab[:, 1] // 4.0).astype(np.int32), -32, 32)
            b_bin = np.clip((block_lab[:, 2] // 4.0).astype(np.int32), -32, 32)
            keys = (l_bin.astype(np.int64) * 10000 + (a_bin + 32) * 100 + (b_bin + 32)).astype(
                np.int64
            )
            uniq, inv = np.unique(keys, return_inverse=True)
            counts = np.bincount(inv)
            mode = int(uniq[int(np.argmax(counts))])
            mask = keys == mode
            mean_rgb = block_rgb[mask].mean(axis=0)
            out[y, x] = np.clip(np.round(mean_rgb), 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def resample_to_grid(img: Image.Image, w: int, h: int, mode: str = "lanczos") -> Image.Image:
    """Resample an image to a target grid size.

    Parameters
    ----------
    mode
        ``lanczos``: single PIL Lanczos resize (one sample per cell).
        ``cell-dominant``: LAB-binned dominant color per mapped source rectangle.
    """
    if w <= 0 or h <= 0:
        raise ValueError("Grid dimensions must be positive.")
    if mode == "lanczos":
        return img.resize((w, h), resample=Image.LANCZOS)
    if mode == "cell-dominant":
        return resample_to_grid_cell_dominant_lab(img, w, h)
    raise ValueError(f"Unknown resample mode: {mode!r}")
