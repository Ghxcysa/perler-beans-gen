"""Regression tests for grid resampling: Lanczos vs cell-dominant LAB mode."""
from __future__ import annotations

import numpy as np
from PIL import Image

from perler_gen.palette import Palette, PaletteColor
from perler_gen.preprocess import denoise_image, load_image, resample_to_grid
from perler_gen.quantize import _majority_vote_filter, _small_component_filter, quantize_to_palette


def test_cell_dominant_checkerboard_cells_stay_saturated_vs_lanczos_gray():
    """4×4 checkerboard down to 2×2: dominant cells stay pure B/W; Lanczos often mixes."""
    pat = np.zeros((4, 4, 3), dtype=np.uint8)
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                pat[i, j] = (255, 255, 255)
    img = Image.fromarray(pat, mode="RGB")
    l_arr = np.asarray(resample_to_grid(img, 2, 2, mode="lanczos"), dtype=np.uint8)
    d_arr = np.asarray(resample_to_grid(img, 2, 2, mode="cell-dominant"), dtype=np.uint8)
    pure = np.logical_or(
        np.all(d_arr == 0, axis=-1),
        np.all(d_arr == 255, axis=-1),
    )
    assert pure.all(), "cell-dominant should output only saturated black or white per cell"
    assert np.any(np.logical_and(l_arr > 15, l_arr < 240).any(-1)), (
        "expected Lanczos 2×2 from 4×4 checker to include at least one mid-tone mixed pixel"
    )


def test_cell_dominant_and_lanczos_rgb_differ_on_checkerboard():
    """Same synthetic checkerboard: two resample paths must not be byte-identical RGB."""
    pat = np.zeros((4, 4, 3), dtype=np.uint8)
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                pat[i, j] = (255, 255, 255)
    img = Image.fromarray(pat, mode="RGB")
    l_arr = np.asarray(resample_to_grid(img, 2, 2, mode="lanczos"))
    d_arr = np.asarray(resample_to_grid(img, 2, 2, mode="cell-dominant"))
    assert not np.array_equal(l_arr, d_arr), (
        "Lanczos resize vs LAB-bin dominant should yield different RGB on 4×4 checker → 2×2"
    )


def test_load_image_composites_transparent_pixels_on_background(tmp_path):
    """Transparent PNG input should not turn into black noise after RGB conversion."""
    data = np.array(
        [
            [[255, 0, 0, 255], [0, 255, 0, 0]],
            [[0, 0, 255, 128], [0, 0, 0, 0]],
        ],
        dtype=np.uint8,
    )
    path = tmp_path / "alpha.png"
    Image.fromarray(data, mode="RGBA").save(path)

    arr = np.asarray(load_image(str(path), alpha_background=(255, 255, 255)), dtype=np.uint8)
    assert arr[0, 1].tolist() == [255, 255, 255]
    assert arr[1, 1].tolist() == [255, 255, 255]
    assert arr[0, 0].tolist() == [255, 0, 0]


def test_denoise_image_removes_single_pixel_salt_noise():
    """Median denoise should clean isolated source speckles before grid sampling."""
    rgb = np.full((5, 5, 3), 255, dtype=np.uint8)
    rgb[2, 2] = (0, 0, 0)
    img = Image.fromarray(rgb, mode="RGB")

    out = np.asarray(denoise_image(img, passes=1), dtype=np.uint8)
    assert out[2, 2].tolist() == [255, 255, 255]


def test_small_component_filter_removes_speckle_but_keeps_connected_line():
    """Speckle cleanup removes tiny islands without erasing connected strokes."""
    idx = np.zeros((5, 5), dtype=np.int32)
    idx[1, 1] = 1
    idx[3, 1:4] = 2

    out = _small_component_filter(idx, n_passes=1, max_size=2, min_neighbor_ratio=0.6)
    assert int(out[1, 1]) == 0
    assert out[3, 1:4].tolist() == [2, 2, 2]


def test_quantize_with_palette_edge_weight_runs():
    """Smoke test: edge-weighted greedy reduction path."""
    pal = Palette(
        name="p",
        colors=tuple(
            PaletteColor(code=str(i), name=str(i), rgb=(40 * i, 20 * i, 255 - 35 * i))
            for i in range(5)
        ),
    )
    rgb = np.random.default_rng(0).integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    q = quantize_to_palette(
        img, pal, max_colors=3, post_smooth=0, reduce_palette_edge_weight=True
    )
    assert len(q.palette.colors) == 3
    assert q.indices.shape == (8, 8)


def test_conservative_majority_keeps_high_contrast_singleton():
    """Conservative mode should not erase a single pixel that differs strongly from neighbors."""
    from perler_gen.quantize import _rgb_to_lab

    pal = Palette(
        name="bw",
        colors=(
            PaletteColor(code="a", name="Black", rgb=(0, 0, 0)),
            PaletteColor(code="b", name="White", rgb=(255, 255, 255)),
        ),
    )
    pr = pal.rgb_array.astype(np.uint8)
    plab = _rgb_to_lab(pr.reshape(1, 2, 3)).reshape(2, 3)

    idx = np.zeros((3, 3), dtype=np.int32)
    idx[1, 1] = 1
    out_std = _majority_vote_filter(idx, n_passes=1, palette_lab=plab, mode="standard")
    out_con = _majority_vote_filter(
        idx, n_passes=1, palette_lab=plab, mode="conservative", conservative_de_max=8.0
    )
    assert int(out_std[1, 1]) == 0
    assert int(out_con[1, 1]) == 1
