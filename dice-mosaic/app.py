"""Streamlit UI for D6 dice mosaic generator."""

from __future__ import annotations

import io

import cv2
import numpy as np
import streamlit as st

from src.dice_render import (
    render_comparison_strip,
    render_mosaic,
    rotations_summary,
)
from src.mosaic import (
    GRID_SIZE,
    MosaicConfig,
    build_mosaic,
    load_image_from_bytes,
    matrix_to_assembly_text,
    matrix_to_csv,
    matrix_to_json,
    validate_square_image,
)

st.set_page_config(
    page_title="D6 Dice Mosaic",
    page_icon="🎲",
    layout="wide",
)

st.title("Gerador de Mosaico com Dados D6")
st.markdown(
    "Transforme uma imagem em mosaico físico contruído a partir de **dados D6**"
)


@st.cache_data(show_spinner=False)
def process_cached(
    image_bytes: bytes,
    brightness: float,
    use_hist_eq: bool,
    invert: bool,
    optimize_rotation: bool,
):
    image = load_image_from_bytes(image_bytes)
    config = MosaicConfig(
        brightness=brightness,
        use_histogram_equalization=use_hist_eq,
        invert=invert,
        optimize_rotation=optimize_rotation,
    )
    return build_mosaic(image, config), image


uploaded = st.file_uploader(
    "Upload da imagem (a imagem deve ser quadrada(Altura = Largura))",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
)

if uploaded is None:
    st.info("Faça o upload para começar.")
    st.stop()

file_bytes = uploaded.getvalue()
probe = load_image_from_bytes(file_bytes)
img_h, img_w = probe.shape[:2]

try:
    validate_square_image(probe)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header("Mosaico")
    st.metric("Grid dos dados", f"{GRID_SIZE} × {GRID_SIZE}")
    st.caption(f"{GRID_SIZE * GRID_SIZE:,} dados físicos necessários")
    st.caption(f"Imagem de origem: {img_w} × {img_h} px (1:1)")

    st.divider()
    st.header("Ajustes da Imagem")

    brightness = st.slider("Brilho", -80.0, 80.0, 0.0, 1.0)
    use_hist_eq = st.checkbox("Equalização do histograma", value=False)
    invert = st.checkbox(
        "Inverter (Troca claro/escuro)", value=False
    )

    st.divider()
    st.header("Orientação dos dados")

    optimize_rotation = st.checkbox(
        "Otimizar rotação das faces 2, 3 e 6",
        value=True,
        help=(
            "Faces assimétricas podem rotacionar 0°, 90°, 180°, or 270° para melhor "
            "se adequar ao padrão de cada célula do grid."
        ),
    )

result, source_bgr = process_cached(
    file_bytes,
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

col1, col2 = st.columns(2)

with col1:
    st.subheader("Prévia do mosaico")
    st.image(
        cv2.cvtColor(mosaic_img, cv2.COLOR_BGR2RGB),
        use_container_width=True,
    )

with col2:
    st.subheader("Pipeline de comparação")
    st.caption("Original → Ajuste em escala de cinza → mosaico de dados")
    st.image(
        cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB),
        use_container_width=True,
    )

st.subheader("Matriz de montagem")
st.caption(
    "Coloque a face indicada virada para cima. "
    "Linha 0 é a linha mais acima; Colunas vão da esquerda para direita. "
    "Faces 2, 3 e 6 podem necessitar de rotação (consultar guia de montagem)."
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

with st.expander("Como funiona:"):
    st.markdown(
        """
        1. **Square check** — Imagem de entrada deve ser 1:1.
        2. **Escala de cinza** — `cv2.cvtColor` (BGR → Cinza).
        3. **Brilho** — Altera do tom da imagem com offset constante.
        4. **Equalização opcional** — `cv2.equalizeHist` espalha o histograma.
        5. **Downsample** — `cv2.resize` define um grid fixo de 40x40 (`INTER_AREA`).
        6. **Quantização** — Assimila a intensidade de cada célula as faces do dado **1** (light) … **6** (dark).
        7. **Rotação** — Para as faces **2**, **3** e **6**, testa quatro orientações e escolhe a que melhor se enquadra no padrão de tonalidade de cada célula.
        8. **Exportar** — Matriz com a lista de cada face (e rotação, quando necessário) para montagem fisica do mosaico.
        """
    )
