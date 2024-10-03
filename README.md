# Gait Stability COM Analysis Web App

Este aplicativo web foi desenvolvido para analisar a movimentação do centro de massa (COM), a margem de estabilidade (MoS) e o **step width** (largura dos passos) usando medidas 2D a partir de vídeos enviados. Ele utiliza o **Mediapipe** para rastreamento de pose e realiza cálculos baseados nas medidas fornecidas.

## Funcionalidades

- **Cálculo do Centro de Massa (COM)**: Determina a movimentação do centro de massa do corpo a partir dos landmarks fornecidos pelo Mediapipe.
- **Cálculo da Margem de Estabilidade (MoS)**: Baseado na projeção do COM, calcula a margem de estabilidade.
- **Cálculo do Step Width**: Mede a largura dos passos durante a marcha.
- **Suporte a Medidas 2D**: Todos os cálculos são realizados utilizando coordenadas 2D.
- **Visualização e Download**: O usuário pode visualizar e baixar o vídeo processado, além de obter um arquivo CSV com os dados calculados.

## Como Utilizar

### Requisitos

- **Python 3.8+**
- **Bibliotecas Python**:
  - `streamlit`
  - `opencv-python`
  - `mediapipe`
  - `numpy`
  - `pandas`

### Instalação

1. **Clone este repositório:**

   ```bash
   git clone https://github.com/seu-usuario/gait-stability-webapp.git
   cd gait-stability-webapp
