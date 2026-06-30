# Gerador de Mosaico com dados D6

Projeto para geração de um mosaico físico a partir das 6 faces de um dado D6. A aplicação permite fazer upload de uma imagem, selecionar manualmente uma região quadrada e gerar um mosaico de dados a partir desse recorte.

O recorte selecionado é convertido para **600 × 600 px** antes do processamento.

## Saídas geradas

- Uma **imagem prévia** com faces virtuais dos dados.
- Uma **matriz de montagem** em CSV / JSON / TXT para construção do mosaico físico.
- Uma versão em PNG de alta resolução do mosaico.

## Pipeline de processamento

1. Upload de imagem quadrada ou retangular.
2. Seleção manual de uma região quadrada pelo usuário.
3. Redimensionamento do recorte para **600 × 600 px**.
4. Conversão para escala de cinza (`cv2.cvtColor`).
5. Ajuste de brilho por offset na intensidade dos pixels.
6. Equalização opcional de histograma (`cv2.equalizeHist`).
7. Definição de grid físico de 40 × 40 (`cv2.resize`, `INTER_AREA`).
8. Quantização uniforme em seis níveis de intensidade → faces dos dados 1–6.
9. Otimização de rotação opcional para as faces assimétricas 2, 3 e 6.

Face **1** representa regiões mais claras; face **6** representa regiões mais escuras.

## Setup

Faça o download do diretório **dice-mosaic_FV**.

Abra o prompt de comando na pasta do projeto e execute:

```bash

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

No Linux/macOS:

```bash

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Executar a aplicação

```bash
streamlit run app.py
```

A aplicação abrirá no navegador padrão, normalmente em `http://localhost:8501`.

## Controles

| Controle | Efeito |
|---|---|
| **Upload da imagem** | Aceita PNG, JPG, JPEG, BMP e WEBP |
| **Seleção quadrada** | Define manualmente qual parte da imagem será usada |
| **Recorte 600 × 600** | O recorte selecionado é redimensionado para 600 × 600 px |
| **Grid** | Fixado em 40 × 40 dados, totalizando 1.600 dados |
| **Brilho** | Aumenta ou reduz o brilho da imagem |
| **Equalização do histograma** | Espalha os tons para imagens com pouco contraste |
| **Inverter claro/escuro** | Troca a representação da ordem claro/escuro |
| **Otimização de rotação** | Rotaciona as faces 2, 3 e 6 para melhor ajuste visual |

## Montagem

A matriz usa **linha 0 = topo**, com colunas da esquerda para a direita. Cada célula indica a face do dado, de 1 a 6, que deve ficar virada para cima.

Exemplo de linha no arquivo TXT:

```text
R000: 3 5 2 6 1 4 ...
```

## Layout do Projeto

```text
dice-mosaic-crop/
  app.py              # Interface do Streamlit
  run_cli.py          # Execução por linha de comando
  requirements.txt
  src/
    __init__.py
    mosaic.py         # Pipeline de processamento com OpenCV
    dice_render.py    # Desenho das faces digitais dos dados
```

## Observação sobre o CLI

A interface Streamlit faz o recorte manual. Já o `run_cli.py` espera receber uma imagem quadrada, pois a seleção de região é interativa e depende da interface web.
