"""Streamlit UI for D6 dice mosaic generator."""

from __future__ import annotations

import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from streamlit_cropper import st_cropper

from src.dice_render import (
    render_comparison_strip,
    render_mosaic,
    rotations_summary,
)
from src.mosaic import (
    GRID_SIZE,
    TARGET_CROP_SIZE,
    MosaicConfig,
    build_mosaic,
    load_image_from_bytes,
    matrix_to_assembly_text,
    matrix_to_csv,
    matrix_to_json,
)

st.set_page_config(
    page_title="D6 Dice Mosaic",
    page_icon="🎲",
    layout="wide",
)

st.title("Gerador de Mosaico com Dados D6")
st.markdown(
    "Transforme uma imagem em mosaico físico construído a partir de **dados D6**."
)


@st.cache_data(show_spinner=False)
def process_cached(
    cropped_image_bytes: bytes,
    brightness: float,
    use_hist_eq: bool,
    invert: bool,
    optimize_rotation: bool,
):
    image = load_image_from_bytes(cropped_image_bytes)
    config = MosaicConfig(
        brightness=brightness,
        use_histogram_equalization=use_hist_eq,
        invert=invert,
        optimize_rotation=optimize_rotation,
    )
    return build_mosaic(image, config), image


def open_uploaded_image(uploaded_file) -> Image.Image:
    """Open a Streamlit UploadedFile safely as an RGB PIL image."""
    file_bytes = uploaded_file.getvalue()
    if not file_bytes:
        raise ValueError("O arquivo enviado está vazio.")

    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.load()  # força a leitura antes do BytesIO sair de escopo
    except UnidentifiedImageError as exc:
        raise ValueError(
            "Não foi possível identificar o arquivo como imagem. "
            "Envie um arquivo PNG, JPG/JPEG, BMP ou WEBP válido."
        ) from exc

    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def pil_image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def normalize_crop_to_600(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    if image.size == (TARGET_CROP_SIZE, TARGET_CROP_SIZE):
        return image
    return image.resize(
        (TARGET_CROP_SIZE, TARGET_CROP_SIZE),
        Image.Resampling.LANCZOS,
    )


def center_square_crop(image: Image.Image, side: int) -> Image.Image:
    """Fallback crop used before the cropper returns coordinates."""
    width, height = image.size
    side = min(side, width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def crop_from_cropper_box(image: Image.Image, crop_box: dict) -> Image.Image:
    """
    Crop manually from coordinates returned by streamlit-cropper.

    Using return_type='box' avoids PIL.UnidentifiedImageError caused when the
    cropper component briefly returns an invalid/empty image payload during reruns.
    """
    if not isinstance(crop_box, dict):
        return center_square_crop(image, min(TARGET_CROP_SIZE, *image.size))

    width, height = image.size

    left = int(round(float(crop_box.get("left", 0))))
    top = int(round(float(crop_box.get("top", 0))))
    box_width = int(round(float(crop_box.get("width", 0))))
    box_height = int(round(float(crop_box.get("height", 0))))

    # O cropper mantém proporção 1:1, mas este ajuste evita erro caso o navegador
    # retorne valores arredondados ou temporariamente inconsistentes.
    selected_side = min(box_width, box_height)
    if selected_side <= 0:
        return center_square_crop(image, min(TARGET_CROP_SIZE, width, height))

    # Se a imagem original permitir, usa uma janela real de 600 x 600 px centrada
    # na seleção. Caso a imagem seja menor, usa o maior quadrado possível e depois
    # redimensiona para 600 x 600 px.
    crop_side = min(TARGET_CROP_SIZE, width, height)
    center_x = left + selected_side // 2
    center_y = top + selected_side // 2

    crop_left = center_x - crop_side // 2
    crop_top = center_y - crop_side // 2

    crop_left = max(0, min(crop_left, width - crop_side))
    crop_top = max(0, min(crop_top, height - crop_side))

    return image.crop(
        (
            crop_left,
            crop_top,
            crop_left + crop_side,
            crop_top + crop_side,
        )
    )


uploaded = st.file_uploader(
    "Upload da imagem",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
)

if uploaded is None:
    st.info("Faça o upload para começar.")
    st.stop()

try:
    source_pil = open_uploaded_image(uploaded)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

img_w, img_h = source_pil.size

st.subheader("1. Selecione a região quadrada")
st.caption(
    "Arraste a caixa sobre a área de interesse. Quando a imagem original tiver "
    f"pelo menos {TARGET_CROP_SIZE} px em largura e altura, o app usa uma janela "
    f"real de {TARGET_CROP_SIZE} × {TARGET_CROP_SIZE} px centralizada na seleção. "
    "Se a imagem for menor, o maior quadrado possível será redimensionado."
)

crop_col, preview_col = st.columns([2, 1])

with crop_col:
    crop_box = st_cropper(
        source_pil,
        realtime_update=True,
        box_color="#2E86C1",
        aspect_ratio=(1, 1),
        return_type="box",
    )

cropped_pil = crop_from_cropper_box(source_pil, crop_box)
pre_resize_size = cropped_pil.size
cropped_pil = normalize_crop_to_600(cropped_pil)
cropped_bytes = pil_image_to_png_bytes(cropped_pil)

with preview_col:
    st.markdown("**Recorte usado no mosaico**")
    st.image(cropped_pil, use_container_width=True)
    if pre_resize_size == (TARGET_CROP_SIZE, TARGET_CROP_SIZE):
        st.caption(f"Recorte final: {TARGET_CROP_SIZE} × {TARGET_CROP_SIZE} px")
    else:
        st.caption(
            f"Recorte original: {pre_resize_size[0]} × {pre_resize_size[1]} px; "
            f"processado como {TARGET_CROP_SIZE} × {TARGET_CROP_SIZE} px."
        )

with st.sidebar:
    st.header("Mosaico")
    st.metric("Grid dos dados", f"{GRID_SIZE} × {GRID_SIZE}")
    st.caption(f"{GRID_SIZE * GRID_SIZE:,} dados físicos necessários")
    st.caption(f"Imagem original: {img_w} × {img_h} px")
    st.caption(f"Recorte processado: {TARGET_CROP_SIZE} × {TARGET_CROP_SIZE} px")

    st.divider()
    st.header("Ajustes da Imagem")

    brightness = st.slider("Brilho", -80.0, 80.0, 0.0, 1.0)
    use_hist_eq = st.checkbox("Equalização do histograma", value=False)
    invert = st.checkbox(
        "Inverter claro/escuro",
        value=False,
    )

    st.divider()
    st.header("Orientação dos dados")

    optimize_rotation = st.checkbox(
        "Otimizar rotação das faces 2, 3 e 6",
        value=True,
        help=(
            "Faces assimétricas podem rotacionar 0°, 90°, 180° ou 270° para melhor "
            "se adequar ao padrão de cada célula do grid."
        ),
    )

result, source_bgr = process_cached(
    cropped_bytes,
    brightness,
    use_hist_eq,
    invert,
    optimize_rotation,
)

matrix = result.matrix
rotations = result.rotations
if optimize_rotation:
    st.sidebar.caption(rotations_summary(rotations, matrix))

mosaic_img = render_mosaic(
    matrix,
    cell_size=40,
    show_grid_labels=True,
    rotations=rotations if optimize_rotation else None,
)
comparison = render_comparison_strip(
    source_bgr,
    result.adjusted_gray,
    matrix,
    max_height=360,
    rotations=rotations if optimize_rotation else None,
)

st.subheader("2. Resultado do mosaico")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Prévia do mosaico")
    st.image(
        cv2.cvtColor(mosaic_img, cv2.COLOR_BGR2RGB),
        use_container_width=True,
    )

with col2:
    st.subheader("Pipeline de comparação")
    st.caption("Recorte selecionado → ajuste em escala de cinza → mosaico de dados")
    st.image(
        cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB),
        use_container_width=True,
    )

st.subheader("Matriz de montagem")
st.caption(
    "Coloque a face indicada virada para cima. "
    "Linha 0 é a linha mais acima; colunas vão da esquerda para direita. "
    "Faces 2, 3 e 6 podem necessitar de rotação, conforme o guia de montagem."
)

tab_heatmap, tab_rotations, tab_table, tab_files = st.tabs(
    ["Heatmap", "Rotações", "Tabela", "Guia de download"]
)

with tab_heatmap:
    st.dataframe(matrix, use_container_width=True, height=400)

with tab_rotations:
    if optimize_rotation:
        st.dataframe(rotations * 90, use_container_width=True, height=400)
        st.caption("Rotação em graus para cada dado (0° = orientação padrão).")
    else:
        st.info("Ative a otimização de rotação na barra lateral para visualizar essa tabela.")

with tab_table:
    st.text(matrix_to_assembly_text(matrix, rotations if optimize_rotation else None))

with tab_files:
    _, c1, c2, c3 = st.columns(4)

    mosaic_png = cv2.imencode(".png", mosaic_img)[1].tobytes()
    c1.download_button(
        "Prévia do mosaico (PNG)",
        mosaic_png,
        file_name="dice_mosaic_preview.png",
        mime="image/png",
    )

    c2.download_button(
        "Matriz de montagem (CSV)",
        matrix_to_csv(matrix, rotations if optimize_rotation else None),
        file_name="assembly_matrix.csv",
        mime="text/csv",
    )

    c3.download_button(
        "Matriz de montagem (JSON)",
        matrix_to_json(matrix, rotations if optimize_rotation else None),
        file_name="assembly_matrix.json",
        mime="application/json",
    )

    st.download_button(
        "Guia de montagem (TXT)",
        matrix_to_assembly_text(matrix, rotations if optimize_rotation else None),
        file_name="assembly_guide.txt",
        mime="text/plain",
    )

    high_res = render_mosaic(
        matrix,
        cell_size=64,
        rotations=rotations if optimize_rotation else None,
    )
    hi_png = cv2.imencode(".png", high_res)[1].tobytes()
    st.download_button(
        "Mosaico em alta resolução (PNG)",
        hi_png,
        file_name="dice_mosaic_hires.png",
        mime="image/png",
    )

with st.expander("Como funciona:"):
    st.markdown(
        f"""
        1. **Upload** — A imagem pode ser quadrada ou retangular.
        2. **Seleção manual** — O usuário escolhe a região de interesse na imagem.
        3. **Recorte** — O app extrai uma janela quadrada de `{TARGET_CROP_SIZE} × {TARGET_CROP_SIZE}` px quando possível.
        4. **Normalização** — Caso necessário, o recorte é convertido para `{TARGET_CROP_SIZE} × {TARGET_CROP_SIZE}` px.
        5. **Escala de cinza** — `cv2.cvtColor` converte BGR → cinza.
        6. **Brilho** — Altera o tom da imagem com offset constante.
        7. **Equalização opcional** — `cv2.equalizeHist` espalha o histograma.
        8. **Downsample** — `cv2.resize` define um grid fixo de 40 × 40 (`INTER_AREA`).
        9. **Quantização** — Associa a intensidade de cada célula às faces do dado **1** (claro) até **6** (escuro).
        10. **Rotação** — Para as faces **2**, **3** e **6**, testa quatro orientações e escolhe a que melhor se enquadra no padrão de tonalidade de cada célula.
        11. **Exportar** — Gera matriz com a lista de faces e rotações para montagem física do mosaico.
        """
    )
