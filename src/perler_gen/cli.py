"""Command-line interface for Perler-Gen."""
from __future__ import annotations

import argparse
from pathlib import Path

from .counts import compute_counts
from .export_assets import write_bead_list_csv, write_preview_png, write_svg
from .export_pdf import PatternMeta, write_pattern_pdf
from .palette import load_palette
from .preprocess import denoise_image, load_image, resample_to_grid, smooth_image
from .quantize import quantize_to_palette
from .step_planner import plan_steps


def _parse_rgb(value: str) -> tuple[int, int, int]:
    """Parse '#rrggbb' or 'r,g,b' CLI colors."""
    text = value.strip()
    if text.startswith("#") and len(text) == 7:
        try:
            return tuple(int(text[i:i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid RGB color: {value!r}") from exc
    parts = text.split(",")
    if len(parts) == 3:
        try:
            rgb = tuple(int(p) for p in parts)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid RGB color: {value!r}") from exc
        if all(0 <= c <= 255 for c in rgb):
            return rgb  # type: ignore[return-value]
    raise argparse.ArgumentTypeError("Use '#rrggbb' or 'r,g,b'.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perler-Gen: Image to perler bead pattern.")
    parser.add_argument("--input", required=True, help="Input image path (jpg/png).")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument("--grid", nargs=2, type=int, metavar=("W", "H"), default=[48, 48])
    parser.add_argument(
        "--max-colors",
        type=int,
        default=24,
        help="Greedy palette reduction: keep only the top-N most matched full-palette "
             "colors before final match. Lower values simplify shopping lists but can "
             "merge or gray out rare edge/structure colors.",
    )
    parser.add_argument("--palette", default="assets/palettes/hama_midi.json")
    parser.add_argument("--steps", choices=["row", "quadrant", "color"], default="row",
                        help="Step mode: 'row' (N rows per step), 'quadrant' (4 quadrants), "
                             "or 'color' (one step per palette color, matching SVG numbering).")
    parser.add_argument("--rows-per-step", type=int, default=2)
    parser.add_argument("--export-svg", action="store_true")
    parser.add_argument(
        "--dither",
        action="store_true",
        help="Floyd–Steinberg dithering for preview.png and SVG only: breaks up banding "
             "but adds noisy single-cell variation. The PDF pattern always uses solid "
             "nearest-neighbor colors (one bead = one color).",
    )
    parser.add_argument(
        "--pre-smooth",
        type=float,
        default=0.0,
        metavar="RADIUS",
        help="Gaussian blur before downsampling (0 = off). Non-zero reduces JPEG/compression "
             "noise but softens thin lines and texture before they are folded into each cell.",
    )
    parser.add_argument(
        "--denoise",
        type=int,
        default=1,
        metavar="N",
        help="Median denoise passes before downsampling (default: 1). Use 0 to keep every "
             "source pixel detail; use 2 for visibly noisy photos.",
    )
    parser.add_argument(
        "--alpha-background",
        type=_parse_rgb,
        default=(255, 255, 255),
        metavar="RGB",
        help="Background color used when flattening transparent PNGs, e.g. '#ffffff' "
             "or '255,255,255'.",
    )
    parser.add_argument(
        "--resample",
        choices=["lanczos", "cell-dominant"],
        default="cell-dominant",
        help="How to build each grid cell color before quantization. 'lanczos' is a single "
             "high-quality resize (fast). 'cell-dominant' picks a LAB-binned mode color inside "
             "each source cell rectangle, which often preserves sharp boundaries better at the "
             "same grid size (slower).",
    )
    parser.add_argument(
        "--post-smooth",
        type=int,
        default=1,
        metavar="N",
        help="Majority-vote denoise passes on palette indices after match (0 = off). "
             "Each pass can remove isolated pixels; values >0 reduce speckle but can erase "
             "single-cell highlights, thin strokes, and small facial details.",
    )
    parser.add_argument(
        "--post-smooth-mode",
        choices=["standard", "conservative", "speckle"],
        default="speckle",
        help="How post-smooth majority voting is applied (only if --post-smooth > 0). "
             "'speckle' replaces tiny connected color islands only when their boundary has "
             "one clear dominant neighbor. "
             "'standard' replaces a cell when a strict neighbor majority disagrees. "
             "'conservative' only flips when the majority is stronger and the palette colors "
             "are similar in ΔE2000, so high-contrast single-cell features are kept.",
    )
    parser.add_argument(
        "--speckle-size",
        type=int,
        default=2,
        metavar="N",
        help="Largest same-color connected island cleaned by --post-smooth-mode speckle "
             "(default: 2 cells).",
    )
    parser.add_argument(
        "--palette-edge-weight",
        action="store_true",
        help="When --max-colors is below the palette size, weight edge pixels more in the "
             "greedy color-frequency reduction so colors that appear on strong boundaries "
             "are less likely to be dropped.",
    )
    parser.add_argument("--grid-interval", type=int, default=5, metavar="N",
                        help="Draw bold major grid lines every N cells (default: 5).")
    parser.add_argument(
        "--pdf-step-orientation",
        choices=["portrait", "landscape"],
        default="landscape",
        help="PDF step pages orientation: 'landscape' (larger grid, default) or 'portrait'.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    w, h = args.grid

    img = load_image(str(input_path), alpha_background=args.alpha_background)
    img = denoise_image(img, passes=args.denoise)
    img = smooth_image(img, radius=args.pre_smooth)
    img = resample_to_grid(img, w, h, mode=args.resample)

    palette = load_palette(str(args.palette))

    # Clean quantization (no dithering) — used for PDF pattern and bead count CSV.
    # Each cell in the PDF represents one physical bead; dithering is not meaningful here.
    quant_kwargs = dict(
        max_colors=args.max_colors,
        post_smooth=args.post_smooth,
        post_smooth_mode=args.post_smooth_mode,
        reduce_palette_edge_weight=args.palette_edge_weight,
        speckle_size=args.speckle_size,
    )
    quantized_clean = quantize_to_palette(
        img, palette,
        dither=False,
        **quant_kwargs,
    )

    # Display quantization — used for preview.png and SVG where adjacent pixels
    # blend visually and dithering can improve perceived color accuracy.
    if args.dither:
        quantized_display = quantize_to_palette(
            img, palette,
            dither=True,
            **quant_kwargs,
        )
    else:
        quantized_display = quantized_clean

    counts = compute_counts(quantized_clean.indices, quantized_clean.palette)

    steps = plan_steps(
        w, h,
        mode=args.steps,
        rows_per_step=args.rows_per_step,
        indices=quantized_clean.indices,
        palette_size=len(quantized_clean.palette.colors),
    )

    preview_path = outdir / "preview.png"
    bead_list_path = outdir / "bead_list.csv"
    pdf_path = outdir / "pattern.pdf"
    svg_path = outdir / "pattern.svg"

    write_preview_png(str(preview_path), quantized_display.rgb, scale=10)
    write_bead_list_csv(str(bead_list_path), counts)

    title = input_path.stem
    meta = PatternMeta(
        title=title,
        grid_w=w,
        grid_h=h,
        palette_name=quantized_clean.palette.name,
    )
    write_pattern_pdf(
        str(pdf_path),
        meta,
        quantized_clean,
        steps,
        grid_interval=args.grid_interval,
        step_orientation=args.pdf_step_orientation,
    )

    if args.export_svg:
        write_svg(str(svg_path), quantized_display.rgb, quantized_display.indices, cell_size=10)


if __name__ == "__main__":
    main()
