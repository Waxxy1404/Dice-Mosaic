"""Command-line interface for batch dice mosaic generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.dice_render import render_mosaic
from src.mosaic import (
    GRID_SIZE,
    MosaicConfig,
    build_mosaic,
    load_image,
    matrix_to_assembly_text,
    matrix_to_csv,
    matrix_to_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a D6 dice mosaic.")
    parser.add_argument("input", type=Path, help="Input image path")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--brightness", type=float, default=0.0)
    parser.add_argument("--hist-eq", action="store_true")
    parser.add_argument("--invert", action="store_true")
    parser.add_argument(
        "--no-optimize-rotation",
        action="store_true",
        help="Disable rotation optimization for faces 2, 3, and 6",
    )
    parser.add_argument("--cell-size", type=int, default=48, help="Preview die size in px")
    args = parser.parse_args()

    image = load_image(args.input)

    config = MosaicConfig(
        brightness=args.brightness,
        use_histogram_equalization=args.hist_eq,
        invert=args.invert,
        optimize_rotation=not args.no_optimize_rotation,
    )

    result = build_mosaic(image, config)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mosaic = render_mosaic(
        result.matrix,
        cell_size=args.cell_size,
        rotations=result.rotations if config.optimize_rotation else None,
    )
    cv2.imwrite(str(out_dir / "mosaic_preview.png"), mosaic)

    (out_dir / "assembly_matrix.csv").write_text(
        matrix_to_csv(
            result.matrix,
            result.rotations if config.optimize_rotation else None,
        ),
        encoding="utf-8",
    )
    (out_dir / "assembly_matrix.json").write_text(
        matrix_to_json(
            result.matrix,
            result.rotations if config.optimize_rotation else None,
        ),
        encoding="utf-8",
    )
    (out_dir / "assembly_guide.txt").write_text(
        matrix_to_assembly_text(
            result.matrix,
            result.rotations if config.optimize_rotation else None,
        ),
        encoding="utf-8",
    )

    print(f"Grid: {GRID_SIZE} x {GRID_SIZE} ({GRID_SIZE * GRID_SIZE} dice)")
    print(f"Saved outputs to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
