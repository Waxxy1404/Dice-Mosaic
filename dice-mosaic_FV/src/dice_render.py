"""Render D6 dice faces into a mosaic preview image."""

from __future__ import annotations

import cv2
import numpy as np

from .mosaic import FACE_COUNT

ASYMMETRIC_FACES = frozenset({2, 3, 6})

# Standard pip positions on a unit square (x, y) in [0, 1]
_PIP_LAYOUT: dict[int, list[tuple[float, float]]] = {
    1: [(0.5, 0.5)],
    2: [(0.25, 0.25), (0.75, 0.75)],
    3: [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)],
    4: [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)],
    5: [
        (0.25, 0.25),
        (0.75, 0.25),
        (0.5, 0.5),
        (0.25, 0.75),
        (0.75, 0.75),
    ],
    6: [
        (0.25, 0.25),
        (0.25, 0.5),
        (0.25, 0.75),
        (0.75, 0.25),
        (0.75, 0.5),
        (0.75, 0.75),
    ],
}


def rotate_normalized(px: float, py: float, quarter_turns: int) -> tuple[float, float]:
    x, y = px, py
    for _ in range(quarter_turns % 4):
        x, y = 1.0 - y, x
    return x, y


def render_face_template(
    face: int,
    quarter_turns: int,
    size: int,
    *,
    die_gray: float = 245.0,
    pip_gray: float = 25.0,
) -> np.ndarray:
    canvas = np.full((size, size), die_gray, dtype=np.float32)
    face = int(np.clip(face, 1, FACE_COUNT))
    radius = max(1, size // 10)
    margin = size // 8
    inner = size - 2 * margin

    for px, py in _PIP_LAYOUT[face]:
        rx, ry = rotate_normalized(px, py, quarter_turns)
        cx = int(margin + rx * inner)
        cy = int(margin + ry * inner)
        cv2.circle(canvas, (cx, cy), radius, pip_gray, thickness=-1)

    return canvas


def optimize_rotations(
    grid_gray: np.ndarray,
    faces: np.ndarray,
    *,
    template_size: int = 24,
) -> np.ndarray:
    """Pick 0/90/180/270° rotations for faces 2, 3, and 6 per cell."""
    rows, cols = grid_gray.shape
    rotations = np.zeros((rows, cols), dtype=np.int32)

    for row in range(rows):
        for col in range(cols):
            face = int(faces[row, col])
            if face not in ASYMMETRIC_FACES:
                continue

            r0, r1 = max(0, row - 1), min(rows, row + 2)
            c0, c1 = max(0, col - 1), min(cols, col + 2)
            patch = cv2.resize(
                grid_gray[r0:r1, c0:c1].astype(np.float32),
                (template_size, template_size),
                interpolation=cv2.INTER_LINEAR,
            )

            best_r, best_err = 0, float("inf")
            for quarter_turn in range(4):
                template = render_face_template(face, quarter_turn, template_size)
                err = float(np.mean((patch - template) ** 2))
                if err < best_err:
                    best_err, best_r = err, quarter_turn
            rotations[row, col] = best_r

    return rotations


def rotations_summary(rotations: np.ndarray, faces: np.ndarray) -> str:
    mask = np.isin(faces, list(ASYMMETRIC_FACES))
    if not mask.any():
        return "No asymmetric faces (2, 3, or 6) in the mosaic."
    rotated = int(np.count_nonzero(rotations[mask]))
    total = int(mask.sum())
    pct = rotated * 100 // max(1, total)
    return f"Asymmetric faces: {total} cells, {rotated} rotated ({pct}%)"


def draw_die_face(
    canvas: np.ndarray,
    top_left: tuple[int, int],
    cell_size: int,
    face: int,
    quarter_turns: int = 0,
    *,
    die_color: tuple[int, int, int] = (245, 245, 245),
    pip_color: tuple[int, int, int] = (25, 25, 25),
    border_color: tuple[int, int, int] = (80, 80, 80),
) -> None:
    x0, y0 = top_left
    x1, y1 = x0 + cell_size, y0 + cell_size
    cv2.rectangle(canvas, (x0, y0), (x1, y1), die_color, thickness=-1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), border_color, thickness=1)

    face = int(np.clip(face, 1, FACE_COUNT))
    radius = max(2, cell_size // 10)
    margin = cell_size // 8
    inner = cell_size - 2 * margin

    for px, py in _PIP_LAYOUT[face]:
        rx, ry = rotate_normalized(px, py, quarter_turns)
        cx = int(x0 + margin + rx * inner)
        cy = int(y0 + margin + ry * inner)
        cv2.circle(canvas, (cx, cy), radius, pip_color, thickness=-1)


def render_mosaic(
    matrix: np.ndarray,
    cell_size: int = 48,
    *,
    show_grid_labels: bool = False,
    rotations: np.ndarray | None = None,
) -> np.ndarray:
    rows, cols = matrix.shape
    label_w = 36 if show_grid_labels else 0
    label_h = 24 if show_grid_labels else 0
    width = label_w + cols * cell_size
    height = label_h + rows * cell_size
    canvas = np.full((height, width, 3), 200, dtype=np.uint8)

    if show_grid_labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.35
        for row in range(rows):
            y = label_h + row * cell_size + cell_size // 2
            cv2.putText(
                canvas,
                str(row),
                (4, y + 4),
                font,
                scale,
                (40, 40, 40),
                1,
                cv2.LINE_AA,
            )
        for col in range(cols):
            x = label_w + col * cell_size + cell_size // 3
            cv2.putText(
                canvas,
                str(col),
                (x, 16),
                font,
                scale,
                (40, 40, 40),
                1,
                cv2.LINE_AA,
            )

    for row in range(rows):
        for col in range(cols):
            x = label_w + col * cell_size
            y = label_h + row * cell_size
            quarter_turns = (
                int(rotations[row, col]) if rotations is not None else 0
            )
            draw_die_face(
                canvas,
                (x, y),
                cell_size,
                int(matrix[row, col]),
                quarter_turns,
            )

    return canvas


def render_comparison_strip(
    original_bgr: np.ndarray,
    adjusted_gray: np.ndarray,
    matrix: np.ndarray,
    max_height: int = 320,
    rotations: np.ndarray | None = None,
) -> np.ndarray:
    """Side-by-side: source | adjusted gray | dice mosaic."""
    h, w = original_bgr.shape[:2]
    scale = max_height / h
    new_w = int(w * scale)
    new_h = max_height

    orig_small = cv2.resize(original_bgr, (new_w, new_h))
    gray_bgr = cv2.cvtColor(adjusted_gray, cv2.COLOR_GRAY2BGR)
    gray_small = cv2.resize(gray_bgr, (new_w, new_h))

    rows, cols = matrix.shape
    mosaic_h = max_height
    mosaic_w = int(mosaic_h * (cols / rows)) if rows else new_w
    cell = max(8, min(mosaic_w // cols, mosaic_h // rows))
    mosaic = render_mosaic(matrix, cell_size=cell, rotations=rotations)

    mosaic_small = cv2.resize(mosaic, (new_w, new_h))
    return np.hstack([orig_small, gray_small, mosaic_small])
