# AGENTS.md — contexto do projeto para agentes/colaboradores

Projeto de estudo (primeiro projeto serio de visao computacional): pipeline de analise tatica de futebol por video.
Priorize **clareza e didatica**. Responda em **portugues**. Seja honesto sobre limitacoes e nao invente APIs:
se nao tiver certeza da assinatura de uma funcao, diga isso e indique a doc oficial.

## Objetivo do pipeline (implementar nesta ordem, cada etapa como modulo independente)
1. **Deteccao** (YOLO/Ultralytics): player, goalkeeper, referee, ball. Bola tratada a parte (slicing + interpolacao).
2. **Tracking** (ByteTrack via supervision/Ultralytics) para jogadores/goleiros/arbitros; bola fora do tracker.
3. **Times**: features do torso (SigLIP/CLIP + UMAP/PCA + KMeans k=2) vs baseline HSV; voto temporal por track ID;
   goleiro -> time por proximidade ao centroide; arbitro pela classe.
4. **Homografia**: keypoints do campo -> `cv2.findHomography` + RANSAC (>=4 pontos nao colineares, limiar de confianca),
   suavizacao temporal, reuso da ultima homografia valida com flag de baixa confianca. Campo 105x68 m.
5. **Minimapa + metricas**: minimapa 2D, video com overlay, distancia/velocidade (filtrar > 40 km/h), posse opcional.

Entregas finais: notebook, README (com licencas), relatorio 6-10 paginas, plano de avaliacao e 3 ablacoes
(slicing da bola, HSV vs embeddings, com/sem suavizacao da homografia).

## Forma de trabalho (IMPORTANTE)
- Trabalhar **etapa por etapa**. Ao fim de cada etapa: (a) codigo, (b) decisoes em linguagem simples,
  (c) comando de verificacao com resultado esperado, (d) riscos, (e) **parar e aguardar confirmacao**.
- Se algo for ambiguo, no maximo **uma pergunta por vez**, propondo um padrao razoavel.
- Qualidade: type hints, docstrings, `logging`, seeds fixas (42), configs em YAML, **sem caminhos absolutos**,
  tratamento de erro (video nao abre, homografia degenerada, frame sem deteccao), processamento por geradores
  (nunca carregar o video inteiro na memoria). Informar FPS aproximado e gargalos.
- Preservar comentarios/docstrings existentes que nao tenham relacao com a mudanca.

## Estado atual
| Etapa | Estado |
|---|---|
| 0 Plano | Aprovado (abordagem hibrida: Roboflow para treino, SoccerNet GSR para avaliacao) |
| 1 Deteccao | Codigo + testes prontos (10 passam). **Treino/avaliacao ainda NAO executados** |
| 2-5 | Nao iniciadas |

**Pendencias da Etapa 1:** treinar na GPU (RTX 3060 12 GB), rodar `evaluate detector`, anotar GT da bola
(CSV `frame,x,y`, ~100-300 frames) e rodar a ablacao `evaluate ball` (com/sem slicing). So depois seguir para a Etapa 2.
O smoke test de 1 epoca em CPU nao foi concluido (servidor reiniciou); na GPU rodar de novo.

## Estrutura
```
configs/default.yaml      hiperparametros, classes, parametros da bola, seed
src/utils.py              config, seed, logging, iter_frames (gerador)
src/detection/dataset.py  validador de dataset YOLO
src/detection/ball.py     slicing, select_ball, interpolacao, mediana movel, recall, BallDetector
src/detection/train_eval.py  train_detector / evaluate_detector (wrappers do Ultralytics)
scripts/                  prepare_dataset.py, train_detector.py, evaluate.py (rodar com `python -m scripts.<nome>`)
tests/                    pytest (logica pura, sem GPU)
data/ runs/ weights/      gitignored (datasets, treinos, pesos, videos)
```

## Ambiente
- Windows + **PowerShell 5.1**: `&&` NAO funciona; use `;`. Ativar venv: `.\.venv\Scripts\Activate.ps1`.
- Python 3.11/3.12 recomendado (3.13 funcionou, mas e muito novo).
- **PyTorch com CUDA**: instalar ANTES do resto usando o comando de https://pytorch.org/get-started/locally/
  (o torch do PyPI no Windows e so CPU). Depois `pip install -r requirements.txt`.
- Nao versionar `requirements.lock.txt` gerado em maquina CPU (fixa torch sem CUDA).
- Verificar GPU: `python -c "import torch; print(torch.cuda.is_available())"` -> `True`.

## Comandos
```powershell
python -m pytest -q                                            # 10 passed
$env:ROBOFLOW_API_KEY="<chave>"                                 # NUNCA gravar a chave em arquivo/commit
python -m scripts.prepare_dataset --workspace roboflow-jvuqo --project football-players-detection-3zvbc --version 10
python -m gdown "https://drive.google.com/uc?id=1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu" -O data/121364_0.mp4
python -m scripts.train_detector --epochs 1 --name smoke        # smoke test
python -m scripts.train_detector --name detector                # treino completo (50 epocas)
python -m scripts.evaluate detector --weights runs/detector/weights/best.pt
python -m scripts.evaluate ball --weights runs/detector/weights/best.pt --video data/121364_0.mp4 [--gt data/ball_gt.csv]
```

## Decisoes e fatos verificados
- Dataset de deteccao: `roboflow-jvuqo/football-players-detection-3zvbc`, **versao 10** (CC BY 4.0, classes
  `ball, goalkeeper, player, referee`; 250/43/19 imagens; 272 bolas, 235 goleiros, 6229 jogadores, 718 arbitros).
  Versoes 15/19/20 esticam a imagem para 640/576 sem manter proporcao (prejudica a bola): evitar.
  A URL `suhyeok/soccer-player-ycvnq` foi descartada (so 2 classes: player e ball).
- Keypoints do campo (Etapa 4): `roboflow-jvuqo/football-field-detection-f07vi` (CC BY 4.0, 317 imagens).
- O download via pacote `roboflow` falhou silenciosamente; `prepare_dataset.py` usa a API REST (link de export do zip).
- SoccerNet GSR (`SoccerNet/SN-GSR-2025`, Hugging Face, GPL-3.0, aparentemente sem NDA — **confirmar na pagina**):
  uso previsto para avaliacao (tracking, times, homografia). Formato nao e YOLO e pode ter linhas (nao keypoints) de campo.
- Video de teste: `121364_0.mp4` (1920x1080, 25 fps, 750 frames) do repo `roboflow/sports`; origem provavel DFL Bundesliga.
  **Licenca nao verificada**: uso so para estudo; nao publicar/versionar.
- Slicing da bola e manual (NumPy puro, testavel); SAHI instalado so para comparacao.
- A bola tem poucos exemplos (272): esperar recall baixo; a ablacao com/sem slicing deve quantificar isso.
- Versoes instaladas na maquina CPU (Python 3.13): torch 2.14.1, ultralytics 8.4.184, supervision 0.31.1,
  numpy 2.3.5, transformers 5.20.2. Estas APIs podem ter mudado: **verificar na doc antes de usar**.
- Possivel conflito entre `opencv-python` (puxado pelo roboflow) e `opencv-python-headless`; se ocorrer, manter so o headless.

## Git
- Remote: `https://github.com/sepectrok/btlearn_soccer_vision.git`, branch `main`.
- Identidade deste projeto: `marcospaulopintochaves@gmail.com` (configurar **localmente**: `git config user.email ...`,
  sem `--global`; a global e a corporativa).
- Nunca commitar: `data/`, `runs/`, `weights/`, `*.pt`, `*.mp4`, `.env`, chaves de API.

## Proximos passos
1. Treinar e avaliar o detector na GPU; reportar mAP50, mAP50-95 e P/R por classe.
2. Anotar GT da bola e rodar a ablacao (com/sem slicing + pos-processamento) reportando FPS.
3. Com aprovacao do usuario, iniciar a Etapa 2 (tracking).
