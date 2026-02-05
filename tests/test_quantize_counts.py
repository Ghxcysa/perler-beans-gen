import numpy as np
from PIL import Image

from perler_gen.counts import compute_counts
from perler_gen.palette import Palette, PaletteColor
from perler_gen.quantize import quantize_to_palette


def _make_palette():
    return Palette(
        name="Test",
        colors=(
            PaletteColor(code="A", name="Black", rgb=(0, 0, 0)),
            PaletteColor(code="B", name="White", rgb=(255, 255, 255)),
        ),
    )


def test_counts_sum_equals_grid():
    palette = _make_palette()
    data = np.array(
        [
            [[0, 0, 0], [255, 255, 255]],
            [[0, 0, 0], [255, 255, 255]],
            [[0, 0, 0], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    img = Image.fromarray(data, mode="RGB")
    quantized = quantize_to_palette(img, palette)
    counts = compute_counts(quantized.indices, quantized.palette)
    total = sum(entry.count for entry in counts)
    assert total == 2 * 3


def test_quantize_indices_valid():
    palette = _make_palette()
    data = np.random.randint(0, 255, size=(4, 5, 3), dtype=np.uint8)
    img = Image.fromarray(data, mode="RGB")
    quantized = quantize_to_palette(img, palette)
    assert quantized.indices.min() >= 0
    assert quantized.indices.max() < len(palette.colors)
