"""Classic computer-vision pipeline for D6 dice mosaics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


FACE_COUNT = 6
GRID_SIZE = 40


@dataclass(frozen=True)
class MosaicConfig:
    """Parameters controlling the dice mosaic."""

    dice_cols: int = GRID_SIZE
    dice_rows: int = GRID_SIZE
    brightness: float = 0.0
    use_histogram_equalization: bool = False
    invert: bool = False
    optimize_rotation: bool = True


@dataclass
class MosaicResult:
    """Outputs of the mosaic pipeline."""

    matrix: np.ndarray
    rotations: np.ndarray
    preview_gray: np.ndarray
    adjusted_gray: np.ndarray
    source_shape: tuple[int, int]


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        target_size = 500
        h, w = image.shape[:2]
        scale = target_size / max(h,w)
        new_w, new_h = int(w*scale), int(h*scale)
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
        pad_vert = target_size - new_h
        pad_horz = target_size - new_w
        top = pad_vert // 2
        bottom = pad_vert - top
        left = pad_horz // 2
        right = pad_horz - left
        square_img = cv2.copyMakeBorder(resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=pad_color)
    return image


def load_image_from_bytes(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image bytes.")
    return image


def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def validate_square_image(image: np.ndarray) -> None:
    """Require a 1:1 input image so the dice grid stays square."""
    height, width = image.shape[:2]
    if width != height:
        raise ValueError(
            f"Image must be square (1:1). Got {width}×{height} pixels."
        )


def adjust_brightness(gray: np.ndarray, brightness: float) -> np.ndarray:
    """Shift pixel intensities by a constant brightness offset."""
    adjusted = gray.astype(np.float32) + brightness
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def resize_to_grid(gray: np.ndarray, cols: int, rows: int) -> np.ndarray:
    return cv2.resize(
        gray,
        (cols, rows),
        interpolation=cv2.INTER_AREA,
    )


def quantize_to_dice_faces(gray: np.ndarray, invert: bool = False) -> np.ndarray:
    """
    Map each pixel intensity to a D6 face value in {1..6}.

    Uses uniform histogram bins over 0-255 (classic intensity quantization).
    Face 1 = lightest region, face 6 = darkest (matches pip count vs. tone).
    """
    working = 255 - gray if invert else gray
    bins = np.linspace(0, 256, FACE_COUNT + 1, dtype=np.int32)
    matrix = np.zeros(gray.shape, dtype=np.int32)
    for bin_index in range(FACE_COUNT):
        face = FACE_COUNT - bin_index
        low = bins[bin_index]
        high = bins[bin_index + 1] - 1 if bin_index < FACE_COUNT - 1 else 255
        mask = (working >= low) & (working <= high)
        matrix[mask] = face
    return matrix


def build_mosaic(image_bgr: np.ndarray, config: MosaicConfig) -> MosaicResult:
    validate_square_image(image_bgr)
    gray = to_grayscale(image_bgr)
    adjusted = adjust_brightness(gray, config.brightness)

    if config.use_histogram_equalization:
        adjusted = cv2.equalizeHist(adjusted)

    grid = resize_to_grid(adjusted, config.dice_cols, config.dice_rows)
    matrix = quantize_to_dice_faces(grid, invert=config.invert)

    if config.optimize_rotation:
        from .dice_render import optimize_rotations

        rotations = optimize_rotations(grid, matrix)
    else:
        rotations = np.zeros_like(matrix)

    return MosaicResult(
        matrix=matrix,
        rotations=rotations,
        preview_gray=grid,
        adjusted_gray=adjusted,
        source_shape=(image_bgr.shape[1], image_bgr.shape[0]),
    )


def _format_face_cell(face: int, rotation_quarters: int) -> str:
    if rotation_quarters % 4 == 0:
        return f"{face}"
    return f"{face}@{rotation_quarters * 90}"


def matrix_to_csv(matrix: np.ndarray, rotations: np.ndarray | None = None) -> str:
    if rotations is None:
        lines = ["row,col,face"]
        rows, cols = matrix.shape
        for row in range(rows):
            for col in range(cols):
                lines.append(f"{row},{col},{matrix[row, col]}")
        return "\n".join(lines)

    lines = ["row,col,face,rotation_deg"]
    rows, cols = matrix.shape
    for row in range(rows):
        for col in range(cols):
            lines.append(
                f"{row},{col},{matrix[row, col]},{int(rotations[row, col]) * 90}"
            )
    return "\n".join(lines)


def matrix_to_assembly_text(
    matrix: np.ndarray,
    rotations: np.ndarray | None = None,
) -> str:
    """Human-readable guide for physical assembly (row 0 = top)."""
    rows, cols = matrix.shape
    header = (
        f"D6 Dice Mosaic Assembly Guide\n"
        f"Grid: {cols} columns x {rows} rows ({cols * rows} dice)\n"
        f"Reading: left-to-right, top-to-bottom. Each number is the face showing up.\n"
        f"Face 1 = lightest, Face 6 = darkest.\n"
    )
    if rotations is not None:
        header += (
            "Faces 2, 3, and 6 may include @90/@180/@270 when the die must be "
            "rotated on the table.\n"
        )
    header += f"{'=' * max(40, cols * 2)}\n"
    body_lines = []
    for row in range(rows):
        row_vals = []
        for col in range(cols):
            face = int(matrix[row, col])
            if rotations is None:
                row_vals.append(f"{face}")
            else:
                row_vals.append(
                    _format_face_cell(face, int(rotations[row, col]))
                )
        body_lines.append(f"R{row:03d}: {' '.join(row_vals)}")
    legend = (
        "\nLegend:\n"
        "  1 ●        2 ● ●      3 ● ● ●\n"
        "             (diag)         (diag)\n"
        "  4 ● ●      5 ● ● ●    6 ● ● ●\n"
        "     ● ●        ● ● ●      ● ● ●\n"
    )
    return header + "\n".join(body_lines) + legend


def matrix_to_json(matrix: np.ndarray, rotations: np.ndarray | None = None) -> str:
    payload = {
        "rows": int(matrix.shape[0]),
        "cols": int(matrix.shape[1]),
        "face_min": 1,
        "face_max": FACE_COUNT,
        "reading_order": "row-major, top-left origin",
        "matrix": matrix.astype(int).tolist(),
    }
    if rotations is not None:
        payload["rotations_deg"] = (rotations.astype(int) * 90).tolist()
    return json.dumps(payload, indent=2)


def suggest_grid_size(
    image_width: int,
    image_height: int,
    dice_size_px: int,
) -> tuple[int, int]:
    """How many dice fit given target dice footprint in source pixels."""
    cols = max(1, image_width // dice_size_px)
    rows = max(1, image_height // dice_size_px)
    return cols, rows
