# Dice Mosaic — PoC para Google Colab

Prova de conceito mínima: **binarização**, **quantização** em 6 faces e **rotação** nas faces 2, 3 e 6.

## Pipeline

1. Imagem → escala de cinza
2. Redimensionar para grade `cols × rows` (1 pixel = 1 dado)
3. **Binarização** — Otsu na grade (`grid_binary`)
4. **Quantização** — 6 níveis de cinza → faces 1..6 (`faces`)
5. **Rotação** (opcional) — faces 2, 3, 6
6. Renderizar pré-visualização

Sem ajuste de brilho, contraste ou gamma.

## Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `Dice_Mosaic_PoC.ipynb` | Notebook Colab |
| `mosaic_poc.py` | Lógica do pipeline |

## Uso no Colab

1. Envie o notebook e `mosaic_poc.py` (ou clone o repositório).
2. Execute as células e faça upload de uma imagem.

## Parâmetros

- `DICE_PX` — resolução do mosaico (pixels da foto por dado)
- `OPTIMIZE_ROTATION` — ligar/desligar rotação 2/3/6
