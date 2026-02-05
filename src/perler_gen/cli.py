"""Command-line interface for Perler-Gen."""
from __future__ import annotations

import argparse
from pathlib import Path

from .counts import compute_counts
from .export_assets import write_bead_list_csv, write_preview_png, write_svg
from .export_pdf import PatternMeta, write_pattern_pdf
from .palette import load_palette
from .preprocess import load_image, resample_to_grid
from .quantize import quantize_to_palette
from .step_planner import plan_steps


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perler-Gen: Image to perler bead pattern.")
    parser.add_argument("--input", required=True, help="Input image path (jpg/png).")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument("--grid", nargs=2, type=int, metavar=("W", "H"), default=[48, 48])
    parser.add_argument("--max-colors", type=int, default=24)
    parser.add_argument("--palette", default="assets/palettes/perler_basic.json")
    parser.add_argument("--steps", choices=["row", "quadrant"], default="row")
    parser.add_argument("--rows-per-step", type=int, default=2)
    parser.add_argument("--export-svg", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    w, h = args.grid

    img = load_image(str(input_path))
    img = resample_to_grid(img, w, h)

    palette = load_palette(str(args.palette))
    quantized = quantize_to_palette(img, palette, max_colors=args.max_colors)

    counts = compute_counts(quantized.indices, quantized.palette)

    steps = plan_steps(w, h, mode=args.steps, rows_per_step=args.rows_per_step)

    preview_path = outdir / "preview.png"
    bead_list_path = outdir / "bead_list.csv"
    pdf_path = outdir / "pattern.pdf"
    svg_path = outdir / "pattern.svg"

    write_preview_png(str(preview_path), quantized.rgb, scale=10)
    write_bead_list_csv(str(bead_list_path), counts)

    title = input_path.stem
    meta = PatternMeta(
        title=title,
        grid_w=w,
        grid_h=h,
        palette_name=quantized.palette.name,
    )
    write_pattern_pdf(str(pdf_path), meta, quantized, steps)

    if args.export_svg:
        write_svg(str(svg_path), quantized.rgb, cell_size=10)


if __name__ == "__main__":
    main()
