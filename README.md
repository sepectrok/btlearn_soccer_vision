# Pipeline de Análise Tática de Futebol por Visão Computacional

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%20Enabled-orange.svg)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-green.svg)](https://github.com/ultralytics/ultralytics)
[![ByteTrack](https://img.shields.io/badge/Tracker-ByteTrack-red.svg)](https://github.com/ifzhang/ByteTrack)
[![Tests](https://img.shields.io/badge/Tests-32%20Passed-brightgreen.svg)](#testes-unitários)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Pipeline de visão computacional ponta a ponta para análise tática e rastreamento de futebol a partir de transmissões monológicas em vídeo de alta definição (*broadcast*). O sistema detecta atores e bola, rastreia identidades estáveis, classifica uniformes de equipes, calcula a homografia planar para o campo regulamentar FIFA (105x68 m), estima velocidades e distâncias físicas e renderiza um minimapa 2D tipo radar em tempo de execução.

---

## ⚽ Estrutura do Pipeline (5 Etapas)

```
[ Vídeo Broadcast ] ──▶ 1. Detecção (YOLOv8s + BallDetector)
                             │
                             ▼
                        2. Rastreamento Temporal (ByteTrack)
                             │
                             ▼
                        3. Classificação de Equipes (K-Means + Torso)
                             │
                             ▼
                        4. Homografia Planar (Keypoints + RANSAC + EMA)
                             │
                             ▼
                        5. Métricas Físicas/Táticas & Minimapa 2D
                             │
                             ▼
                    [ Vídeo Tático Final + JSON ]
```

1. **Detecção Multiclasse**: YOLOv8s treinado para `player`, `goalkeeper`, `referee` e `ball`. Tratamento especial da bola com filtro de salto máximo, interpolação linear de lacunas e mediana móvel.
2. **Rastreamento Temporal**: ByteTrack nativo mantendo IDs consistentes dos jogadores e isolando a bola de colisões indevidas.
3. **Classificação de Equipes**: Recorte anatômico do tronco (*torso crop*), agrupamento K-Means ($k=2$), votação temporal de maioria por `track_id`, resolução do goleiro por proximidade espacial e árbitro por classe.
4. **Homografia Planar**: Detecção de 32 marcos canônicos do campo via YOLOv8x-pose na resolução nativa de $640\times 640$, estimativa por RANSAC ($L=105.0$m, $W=68.0$m), suavização temporal exponencial (EMA, $\alpha=0.15$) e fallback para até 30 frames.
5. **Minimapa 2D e Métricas**: Velocidade instantânea (km/h) com filtro biológico para ruídos ($> 40.0$ km/h), distância acumulada, posse de bola por proximidade espacial ($\le 2.0$m) e radar 2D no vídeo.

---

## 📊 Estudos de Ablação Quantitativos

O projeto implementou e avaliou formalmente três estudos de ablação:

### 1. Detecção da Bola: Inferência Direta vs. Fatiamento em Alta Resolução (SAHI)
- **Sem Slicing (Direto $1280\times 1280$)**: **30.3 FPS**, Recall Bruto 51.0%, **Recall Pós-Processado: 65.3%**.
- **Com Slicing (SAHI $640\times 640$)**: **4.0 FPS**, Recall Bruto 51.7%, **Recall Pós-Processado: 65.3%**.
- **Conclusão:** O fatiamento derrubou a velocidade em $86.8\%$ sem trazer ganho após o pós-processamento temporal. A inferência direta garante operação em tempo real (>30 FPS).

### 2. Representação de Uniformes: Baseline HSV vs. Embeddings SigLIP
- **Baseline HSV (com filtro de grama)**: **1.495 FPS**, **Silhouette: 0.5436**, **Davies-Bouldin: 0.7639**.
- **SigLIP Embeddings + PCA**: **39.4 FPS**, Silhouette: 0.4497, Davies-Bouldin: 0.9347.
- **Conclusão:** O baseline HSV apresentou separabilidade superior e foi quase $40\times$ mais rápido na extração que o modelo fundamental SigLIP na GPU.

### 3. Estabilidade da Homografia: Com vs. Sem Suavização EMA
- **Sem Suavização ($\mathbf{H}_{\text{raw}}$)**: Jitter de aceleração métrica = $0.1611\text{ m/frame}^2$.
- **Com Suavização EMA ($\alpha=0.15$)**: Jitter de aceleração métrica = **$0.1013\text{ m/frame}^2$** (**redução de 37.11% de tremor**).
- **Conclusão:** O filtro EMA estabiliza as marcações no gramado e elimina oscilações espúrias no minimapa sem gerar atraso dinâmico.

---

## 🛠️ Ambiente e Instalação

### Requisitos
- Windows 10/11 ou Linux
- **Python 3.11** ou 3.12
- Placa de vídeo NVIDIA com suporte a CUDA (desejável, testado em RTX 3060 12GB)
- PowerShell 5.1+ (Windows)

### Passo a Passo de Instalação (Windows PowerShell)

```powershell
# 1. Clonar o repositório
git clone https://github.com/sepectrok/btlearn_soccer_vision.git
cd btlearn_soccer_vision

# 2. Criar e ativar o ambiente virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Instalar PyTorch com suporte a CUDA (IMPORTANTE: antes dos outros pacotes)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Instalar as demais dependências do projeto
pip install -r requirements.txt

# 5. Validar disponibilidade da GPU
python -c "import torch; print('CUDA disponível:', torch.cuda.is_available())"
```

---

## 🚀 Como Executar

### 1. Testes Unitários
Execute a suíte completa de testes automatizados:
```powershell
python -m pytest -q
# Resultado esperado: 32 passed
```

### 2. Download do Dataset de Treino (Opcional)
```powershell
$env:ROBOFLOW_API_KEY="<sua_chave>"
python -m scripts.prepare_dataset --workspace roboflow-jvuqo --project football-players-detection-3zvbc --version 10
```

### 3. Treinamento do Detector YOLOv8s
```powershell
# Smoke test (1 época rápida)
python -m scripts.train_detector --epochs 1 --name smoke

# Treinamento completo (50 épocas)
python -m scripts.train_detector --name detector
```

### 4. Avaliação e Rastreamento (ByteTrack)
```powershell
python -m scripts.track_video --weights runs/detector/weights/best.pt --video data/121364_0.mp4 --max-frames 150 --output runs/tracking_demo.mp4
```

### 5. Classificação de Equipes e Ablação 2
```powershell
python -m scripts.classify_teams --weights runs/detector/weights/best.pt --video data/121364_0.mp4 --max-frames 150 --output runs/teams_demo.mp4
```

### 6. Avaliação da Homografia e Ablação 3
```powershell
python -m scripts.evaluate_homography --weights-detector runs/detector/weights/best.pt --video data/121364_0.mp4 --max-frames 150 --output runs/homography_demo.mp4
```

### 7. Pipeline Tático Completo (Etapa 5)
Executa a integração total (todas as etapas, minimapa 2D, HUD e métricas):
```powershell
python -m scripts.run_pipeline --video data/121364_0.mp4 --max-frames 150 --output runs/full_tactical_analysis.mp4 --metrics-output runs/tactical_metrics.json
```

O vídeo anotado será gerado em `runs/full_tactical_analysis.mp4` e as métricas consolidadas em `runs/tactical_metrics.json`.

---

## 📁 Estrutura do Repositório

```
configs/
  default.yaml              # Hiperparâmetros, classes, parâmetros da bola e homografia
src/
  detection/                # Dataset YOLO, BallDetector, treino e avaliação
  tracking/                 # PlayerTracker (ByteTrack nativo) e TrackedPlayer
  teams/                    # TeamClassifier (HSV / SigLIP), TeamVotingManager
  homography/               # PitchModel (105x68 m), HomographyEstimator, metrics
  metrics/                  # PhysicalMetricsTracker, BallPossessionTracker, MinimapRenderer
  utils.py                  # Logger, seeds, gerador iter_frames
scripts/
  prepare_dataset.py        # Download e validação do dataset
  train_detector.py         # Treino do detector
  evaluate.py               # Avaliação mAP e ablação da bola
  track_video.py            # Rastreamento isolado
  classify_teams.py         # Classificação e ablação de uniformes
  evaluate_homography.py    # Avaliação e ablação de homografia
  run_pipeline.py           # Pipeline integrado ponta a ponta
tests/                      # 32 testes unitários (pytest)
docs/
  RELATORIO_TECNICO.md      # Relatório técnico completo de 6 a 10 páginas
notebooks/
  pipeline_demonstracao.ipynb # Notebook interativo passo a passo
```

---

## 📜 Licenças e Atribuições

- **Código deste Repositório**: Licenciado sob a licença [MIT](LICENSE).
- **Dataset de Jogadores e Bola**: `roboflow-jvuqo/football-players-detection-3zvbc` (versão 10), disponível sob licença [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Dataset de Marcos do Campo**: `football-project-pifbc/football-field-detection-f07vi-d0ele`, disponível sob licença [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Ultralytics YOLOv8**: Licenciado sob [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
- **SigLIP (Google)**: Licenciado sob [Apache-2.0](https://huggingface.co/google/siglip-base-patch16-224).
- **Vídeo de Demonstração (`121364_0.mp4`)**: Trecho de transmissão esportiva (Bundesliga) proveniente de repositório educacional (`roboflow/sports`). Utilizado exclusivamente para fins didáticos e científicos de pesquisa, não sendo redistribuído ou versionado no repositório.
