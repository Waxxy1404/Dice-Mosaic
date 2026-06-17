# Gerador de Mosaico com dados D6

Construa um mosaico fisico a partir das 6 faces de um dado D6.Faça upload de uma imagem **quadrada (1:1)**, ajuste o brilho, e faça o download:

- Uma **imagem prévia** com faces virtuais dos dados.
- Uma **Matrix de monstagem** (CSV / JSON / TXT) para construção do mosaico físico.

## Pipeline de processamento

1. Validação da imagem quadrada (Altura = Largura)
2. Converção para escala de cinza (`cv2.cvtColor`)
3. Offset de brilho na intesidade de cada pixel
4. Equalização opcional de histograma (`cv2.equalizeHist`)
5. Definição de grid fisico de 40×40 (`cv2.resize`, `INTER_AREA`)
6. Quantização uniforme em seis níveis de intensidade → faces dos dados 1–6
7. Otimização de rotação opcional para as faces assimétricas 2, 3 e 6

Face **1** é a região mais clara; face **6** é a mais escura.

## Setup
Abra prompt de comando na pasta do projeto e execute os seguintes comandos:
```bash
cd dice-mosaic
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Executar a aplicação
Para exucutar a aplicação execute o comando abaixo:
```bash
streamlit run app.py
```

A aplicação irá abrir no navegador padrão (URL padrão: http://localhost:8501).

## Controles

| Controle | Efeito |
|--------|--------|
| **Imagem de Entrada** | Deve ser quadrada (1:1 aspect ratio) |
| **Grid** | Fixado 40×40 dados (1.600 dados no total) |
| **Brilho** | Altera o brilho da imagem (+/-) |
| **Equalização do histograma** | Espalha os tons para imagens com pouco contraste |
| **Inverter** | Troca a representação da ordem de Claro vs Escuro (para dados pretos com marcadores brancos) |
| **Otimização de Rotação** | Rotaciona as faces 2, 3 e 6 (0°/90°/180°/270°) para melhor se adequar a distribuição de intensidade de cada célula |

## Montagem

A matriz usa **row 0 = topo**, colunas da esquerda para a direita. Cada célula é o valor da face (1-6) que deve ser colocada para **cima** no dado.

Exemplo de linha no arquivo TXT:

```
R000: 3 5 2 6 1 4 ...
```

## Layout do Projeto

```
dice-mosaic/
  app.py              # Interface do Streamlit
  requirements.txt
  src/
    mosaic.py         # Pipeline de processamento cm biblioteca CV
    dice_render.py    # Desenho das faces digitais dos dados
```
