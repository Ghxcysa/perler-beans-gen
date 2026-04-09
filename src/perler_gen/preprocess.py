"""Image loading and preprocessing utilities."""
from __future__ import annotations

from PIL import Image, ImageFilter


def load_image(path: str) -> Image.Image:
    """Load an image from disk and convert to RGB."""
    img = Image.open(path)
    return img.convert("RGB")


def smooth_image(img: Image.Image, radius: float) -> Image.Image:
    """Apply Gaussian blur to reduce fine noise before downsampling."""
    if radius <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def resample_to_grid(img: Image.Image, w: int, h: int) -> Image.Image:
    """Resample an image to a target grid size using Lanczos resize."""
    if w <= 0 or h <= 0:
        raise ValueError("Grid dimensions must be positive.")
    return img.resize((w, h), resample=Image.LANCZOS)
