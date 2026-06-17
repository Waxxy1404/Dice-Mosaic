"""
PoC: mosaico D6 — cinza → grade → binarização (Otsu) → quantização (6 faces)
→ rotação opcional nas faces 2, 3 e 6.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

FACE_COUNT = 6
ASYMMETRIC_FACES = frozenset({2, 3, 6})

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


@dataclass(frozen=True)
class MosaicConfig:
    dice_cols: int
    dice_rows: int
    optimize_rotation: bool = True


@dataclass
class MosaicResult:
    faces: np.ndarray
    rotations: np.ndarray
    grid_gray: np.ndarray
    grid_binary: np.ndarray


def load_image_from_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Não foi possível decodificar a imagem.")
    return image


def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def resize_to_grid(gray: np.ndarray, cols: int, rows: int) -> np.ndarray:
    return cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)


def binarize_otsu(gray: np.ndarray) -> np.ndarray:
    """Binarização automática (Otsu): 0 = preto, 255 = branco."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def quantize_to_faces(gray: np.ndarray) -> np.ndarray:
    """
    Quantização em 6 níveis (faces 1..6).
    Face 1 = pixels mais claros, face 6 = mais escuros.
    """
    bins = np.linspace(0, 256, FACE_COUNT + 1, dtype=np.int32)
    matrix = np.zeros(gray.shape, dtype=np.int32)
    for bin_index in range(FACE_COUNT):
        face = FACE_COUNT - bin_index
        low = bins[bin_index]
        high = bins[bin_index + 1] - 1 if bin_index < FACE_COUNT - 1 else 255
        mask = (gray >= low) & (gray <= high)
        matrix[mask] = face
    return matrix


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


def draw_die_face_bgr(
    canvas: np.ndarray,
    top_left: tuple[int, int],
    cell_size: int,
    face: int,
    quarter_turns: int = 0,
) -> None:
    x0, y0 = top_left
    x1, y1 = x0 + cell_size, y0 + cell_size
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (245, 245, 245), thickness=-1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (80, 80, 80), thickness=1)

    face = int(np.clip(face, 1, FACE_COUNT))
    radius = max(2, cell_size // 10)
    margin = cell_size // 8
    inner = cell_size - 2 * margin

    for px, py in _PIP_LAYOUT[face]:
        rx, ry = rotate_normalized(px, py, quarter_turns)
        cx = int(x0 + margin + rx * inner)
        cy = int(y0 + margin + ry * inner)
        cv2.circle(canvas, (cx, cy), radius, (25, 25, 25), thickness=-1)


def optimize_rotations(
    grid_gray: np.ndarray,
    faces: np.ndarray,
    *,
    template_size: int = 24,
) -> np.ndarray:
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
            for r in range(4):
                tmpl = render_face_template(face, r, template_size)
                err = float(np.mean((patch - tmpl) ** 2))
                if err < best_err:
                    best_err, best_r = err, r
            rotations[row, col] = best_r

    return rotations


def build_mosaic(image_bgr: np.ndarray, config: MosaicConfig) -> MosaicResult:
    gray = to_grayscale(image_bgr)
    grid_gray = resize_to_grid(gray, config.dice_cols, config.dice_rows)
    grid_binary = binarize_otsu(grid_gray)
    faces = quantize_to_faces(grid_gray)

    if config.optimize_rotation:
        rotations = optimize_rotations(grid_gray, faces)
    else:
        rotations = np.zeros_like(faces)

    return MosaicResult(
        faces=faces,
        rotations=rotations,
        grid_gray=grid_gray,
        grid_binary=grid_binary,
    )


def render_mosaic_bgr(
    faces: np.ndarray,
    rotations: np.ndarray,
    cell_size: int = 40,
) -> np.ndarray:
    rows, cols = faces.shape
    canvas = np.full((rows * cell_size, cols * cell_size, 3), 200, dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            draw_die_face_bgr(
                canvas,
                (col * cell_size, row * cell_size),
                cell_size,
                int(faces[row, col]),
                int(rotations[row, col]),
            )
    return canvas


def suggest_grid_size(
    image_width: int, image_height: int, dice_size_px: int
) -> tuple[int, int]:
    cols = max(1, image_width // dice_size_px)
    rows = max(1, image_height // dice_size_px)
    return cols, rows


def rotations_summary(rotations: np.ndarray, faces: np.ndarray) -> str:
    mask = np.isin(faces, list(ASYMMETRIC_FACES))
    if not mask.any():
        return "Nenhuma face 2, 3 ou 6 no mosaico."
    nonzero = int(np.count_nonzero(rotations[mask]))
    total = int(mask.sum())
    return (
        f"Faces 2/3/6: {total} células, {nonzero} com rotação ≠ 0° "
        f"({nonzero * 100 // max(1, total)}%)."
    )
