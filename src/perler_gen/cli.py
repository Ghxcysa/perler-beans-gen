"""Command-line interface for Perler-Gen."""
from __future__ import annotations

import argparse
from pathlib import Path

from .counts import compute_counts
from .export_assets import write_bead_list_csv, write_preview_png, write_svg
from .export_pdf import PatternMeta, write_pattern_pdf
from .palette import load_palette
from .preprocess import load_image, resample_to_grid, smooth_image
from .quantize import quantize_to_palette
from .step_planner import plan_steps


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perler-Gen: Image to perler bead pattern.")
    parser.add_argument("--input", required=True, help="Input image path (jpg/png).")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument("--grid", nargs=2, type=int, metavar=("W", "H"), default=[48, 48])
    parser.add_argument("--max-colors", type=int, default=24)
    parser.add_argument("--palette", default="assets/palettes/perler_basic.json")
    parser.add_argument("--steps", choices=["row", "quadrant", "color"], default="row",
                        help="Step mode: 'row' (N rows per step), 'quadrant' (4 quadrants), "
                             "or 'color' (one step per palette color, matching SVG numbering).")
    parser.add_argument("--rows-per-step", type=int, default=2)
    parser.add_argument("--export-svg", action="store_true")
    parser.add_argument("--dither", action="store_true",
                        help="Enable Floyd-Steinberg dithering for preview.png and SVG only. "
                             "The PDF pattern always uses clean nearest-neighbor quantization "
                             "because each bead is one solid color.")
    parser.add_argument("--pre-smooth", type=float, default=1.0, metavar="RADIUS",
                        help="Gaussian blur radius applied before downsampling (0 to disable).")
    parser.add_argument("--post-smooth", type=int, default=None, metavar="N",
                        help="Majority-vote filter passes after quantization (0 to disable, default: 1).")
    parser.add_argument("--grid-interval", type=int, default=5, metavar="N",
                        help="Draw bold major grid lines every N cells (default: 5).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    w, h = args.grid

    post_smooth = args.post_smooth if args.post_smooth is not None else 1

    img = load_image(str(input_path))
    img = smooth_image(img, radius=args.pre_smooth)
    img = resample_to_grid(img, w, h)

    palette = load_palette(str(args.palette))

    # Clean quantization (no dithering) — used for PDF pattern and bead count CSV.
    # Each cell in the PDF represents one physical bead; dithering is not meaningful here.
    quantized_clean = quantize_to_palette(
        img, palette,
        max_colors=args.max_colors,
        dither=False,
        post_smooth=post_smooth,
    )

    # Display quantization — used for preview.png and SVG where adjacent pixels
    # blend visually and dithering can improve perceived color accuracy.
    if args.dither:
        quantized_display = quantize_to_palette(
            img, palette,
            max_colors=args.max_colors,
            dither=True,
            post_smooth=post_smooth,
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
    write_pattern_pdf(str(pdf_path), meta, quantized_clean, steps, grid_interval=args.grid_interval)

    if args.export_svg:
        write_svg(str(svg_path), quantized_display.rgb, quantized_display.indices, cell_size=10)


if __name__ == "__main__":
    main()
