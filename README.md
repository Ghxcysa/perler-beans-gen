# perler-beads-gen
This repository hosts the Perler-Gen project.

This project converts images into perler bead patterns.

# Perler-Gen (MVP)

Perler-Gen converts a single image into a printable Perler bead pattern with step-by-step pages.

## Features
- Load a single JPG/PNG
- Resample to a fixed grid (e.g. 48x48)
- Quantize colors to a fixed palette (nearest neighbor in RGB)
- Export `pattern.pdf` (cover, legend, step pages), `preview.png`, `bead_list.csv`, optional SVG

## Requirements
- Python 3.10+
- Dependencies: Pillow, numpy, reportlab

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
  --export-svg --pre-smooth 1.5
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
- `--steps`: `row` or `quadrant`
- `--rows-per-step`: rows per step (row mode)
- `--export-svg`: export `pattern.svg`

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




