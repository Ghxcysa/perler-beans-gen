"""Palette loading and data structures."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PaletteColor:
    code: str
    name: str
    rgb: tuple[int, int, int]


@dataclass(frozen=True)
class Palette:
    name: str
    colors: tuple[PaletteColor, ...]

    @property
    def rgb_array(self) -> np.ndarray:
        return np.array([c.rgb for c in self.colors], dtype=np.int16)


def _validate_rgb(rgb: Iterable[int]) -> tuple[int, int, int]:
    vals = list(rgb)
    if len(vals) != 3:
        raise ValueError("RGB must have 3 values.")
    for v in vals:
        if not (0 <= int(v) <= 255):
            raise ValueError("RGB values must be in 0..255.")
    return int(vals[0]), int(vals[1]), int(vals[2])


def load_palette(path: str) -> Palette:
    """Load a palette JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    name = data.get("name", "Palette")
    colors_data = data.get("colors")
    if not isinstance(colors_data, list) or not colors_data:
        raise ValueError("Palette JSON must include a non-empty 'colors' list.")
    colors: list[PaletteColor] = []
    for entry in colors_data:
        code = str(entry.get("code", ""))
        color_name = str(entry.get("name", ""))
        rgb = _validate_rgb(entry.get("rgb", []))
        if not code:
            raise ValueError("Palette color missing code.")
        if not color_name:
            raise ValueError(f"Palette color {code} missing name.")
        colors.append(PaletteColor(code=code, name=color_name, rgb=rgb))
    return Palette(name=name, colors=tuple(colors))
