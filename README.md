# Gerador de Mosaico com dados D6

O projeto converte uma imagem em um mosaico físico feito com dados D6. Cada dado representa um nível de brilho — face 1 para as regiões mais claras, face 6 para as mais escuras — e, organizados em uma grade de 40×40.\n
A interface roda no Streamlit e oferece controles de brilho, equalização de histograma, inversão de polaridade (para dados pretos com marcadores brancos) e ativação da otimização de rotação. O output é uma prévia digital do mosaico e uma matriz de montagem exportável em CSV, JSON ou TXT — o guia para montagem do mosaico físico.

## Pipeline de processamento

1.Validação da imagem de entrada — só são aceitas imagens com proporção 1:1 (altura = largura)\n
2.Conversão para escala de cinza via cv2.cvtColor, descartando informação de cor e mantendo apenas a luminância
3.Aplicação de offset de brilho sobre a intensidade de cada pixel, controlável pelo usuário
4.Equalização opcional do histograma com cv2.equalizeHist, redistribuindo os tons para aumentar o contraste em imagens flat
5.Redimensionamento para a grade física de 40×40 células via cv2.resize com interpolação INTER_AREA, adequada para reduções de escala
6.Quantização uniforme dos níveis de intensidade em seis faixas, mapeadas diretamente para as faces 1–6 do dado
7.Otimização opcional de rotação para as faces assimétricas 2, 3 e 6 — cada célula é testada nas quatro orientações possíveis (0°, 90°, 180°, 270°) e a que melhor representa a distribuição local de intensidade é mantida

## Setup
Caso não possua Python instalado rode apenas o comando abaixo para instalar:
```bash
Python
```
Abra prompt de comando na pasta do projeto e execute os comandos abaixo, caso ```.ven\Scripts\activate``` apresente algum erro, siga sem executar:
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
