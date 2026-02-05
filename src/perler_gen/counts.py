"""Bead count computation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .palette import Palette


@dataclass(frozen=True)
class CountEntry:
    code: str
    name: str
    rgb: tuple[int, int, int]
    count: int


def compute_counts(indices: np.ndarray, palette: Palette) -> list[CountEntry]:
    """Compute bead counts for each palette color used in indices."""
    if indices.ndim != 2:
        raise ValueError("Indices must be a 2D array.")
    counts = np.bincount(indices.flatten(), minlength=len(palette.colors))
    results: list[CountEntry] = []
    for idx, cnt in enumerate(counts.tolist()):
        if cnt <= 0:
            continue
        color = palette.colors[idx]
        results.append(CountEntry(code=color.code, name=color.name, rgb=color.rgb, count=int(cnt)))
    return results
