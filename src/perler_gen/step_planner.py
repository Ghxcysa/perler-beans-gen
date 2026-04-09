"""Step planning for bead placement."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Step:
    name: str
    mask: np.ndarray  # bool HxW


def _row_steps(w: int, h: int, rows_per_step: int) -> list[Step]:
    if rows_per_step <= 0:
        raise ValueError("rows_per_step must be positive.")
    steps: list[Step] = []
    for start in range(0, h, rows_per_step):
        end = min(h, start + rows_per_step)
        mask = np.zeros((h, w), dtype=bool)
        mask[start:end, :] = True
        steps.append(Step(name=f"Rows {start + 1}-{end}", mask=mask))
    return steps


def _quadrant_steps(w: int, h: int) -> list[Step]:
    w_mid = w // 2
    h_mid = h // 2
    regions = [
        ("Quadrant 1 (Top-Left)", (0, h_mid), (0, w_mid)),
        ("Quadrant 2 (Top-Right)", (0, h_mid), (w_mid, w)),
        ("Quadrant 3 (Bottom-Left)", (h_mid, h), (0, w_mid)),
        ("Quadrant 4 (Bottom-Right)", (h_mid, h), (w_mid, w)),
    ]
    steps: list[Step] = []
    for name, (r0, r1), (c0, c1) in regions:
        if r0 == r1 or c0 == c1:
            continue
        mask = np.zeros((h, w), dtype=bool)
        mask[r0:r1, c0:c1] = True
        steps.append(Step(name=name, mask=mask))
    return steps


def _color_steps(indices: np.ndarray, palette_size: int) -> list[Step]:
    """One step per palette color index, in order.

    Each step's mask covers only the cells with that specific color.
    Colors that are not present in the image are skipped.
    """
    steps: list[Step] = []
    for color_idx in range(palette_size):
        mask = (indices == color_idx)
        if not mask.any():
            continue
        steps.append(Step(name=f"Color {color_idx + 1}", mask=mask))
    return steps


def plan_steps(
    w: int,
    h: int,
    mode: str,
    rows_per_step: int = 1,
    indices: Optional[np.ndarray] = None,
    palette_size: int = 0,
) -> list[Step]:
    """Plan steps in row, quadrant, or color mode.

    Parameters
    ----------
    w, h:          Grid dimensions.
    mode:          'row', 'quadrant', or 'color'.
    rows_per_step: Used only for mode='row'.
    indices:       Palette-index array (H x W); required for mode='color'.
    palette_size:  Number of palette colors; required for mode='color'.
    """
    if w <= 0 or h <= 0:
        raise ValueError("Grid dimensions must be positive.")
    mode = mode.lower().strip()
    if mode == "row":
        return _row_steps(w, h, rows_per_step)
    if mode == "quadrant":
        return _quadrant_steps(w, h)
    if mode == "color":
        if indices is None:
            raise ValueError("indices must be provided for mode='color'.")
        return _color_steps(indices, palette_size)
    raise ValueError("Unknown step mode: expected 'row', 'quadrant', or 'color'.")
