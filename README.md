# perler-beads-gen
This repository hosts the Perler-Gen project.

This project converts images into perler bead patterns.

# Perler-Gen (MVP)

Perler-Gen converts a single image into a printable Perler bead pattern with step-by-step pages.

## Features
- Load a single JPG/PNG
- Normalize EXIF orientation and transparent PNG backgrounds
- Denoise, resample to a fixed grid (e.g. 48x48), and clean isolated speckles
- Quantize colors to a fixed palette with perceptual LAB/Delta E matching
- Export `pattern.pdf` (cover, legend, step pages), `preview.png`, `bead_list.csv`, optional SVG

## Requirements
- Python 3.10+
- Dependencies: Pillow, numpy, reportlab, scikit-image

Install:
```
pip install -r requirements.txt
```

## Quick Start
Example command (from repo root):
```
python3 -m perler_gen.cli \
  --input /Users/xiaorui/Desktop/3cdc5129eb5fc3e89545cd99b4ecb309.jpg \
  --outdir out --grid 48 48 --max-colors 24 \
  --steps color \
  --export-svg
```

Output files:
- `examples/output/sample1/pattern.pdf`
- `examples/output/sample1/preview.png`
- `examples/output/sample1/bead_list.csv`

## CLI Options
- `--input`: input image path (jpg/png)
- `--outdir`: output directory
- `--grid W H`: grid size
- `--max-colors`: maximum number of colors (default 24)
- `--palette`: palette JSON file
- `--steps`: `row`, `quadrant`, or `color`
- `--rows-per-step`: rows per step (row mode)
- `--export-svg`: export `pattern.svg`
- `--denoise`: median denoise passes before downsampling (default 1)
- `--resample`: `cell-dominant` (default) or `lanczos`
- `--post-smooth`: palette-index cleanup passes after matching (default 1)
- `--post-smooth-mode`: `speckle` (default), `standard`, or `conservative`
- `--speckle-size`: largest tiny color island cleaned in `speckle` mode (default 2)
- `--alpha-background`: background color for transparent PNGs, such as `#ffffff`
- `--dither`: optional preview/SVG dithering; leave off for clean no-noise output

Numeric CLI options are validated before generation starts, so invalid values
such as zero-sized grids or negative denoise passes fail with a clear message.

## Palette Format
Example:
```json
{
  "name": "Perler Basic",
  "colors": [
    {"code": "P01", "name": "White", "rgb": [255,255,255]},
    {"code": "P02", "name": "Black", "rgb": [0,0,0]}
  ]
}
```

## Tests
Run tests with:
```
pytest
```
