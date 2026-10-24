"""Classificação de times e execução da Ablação 2 (HSV baseline vs SigLIP Embeddings).

Uso:
    # Rodar apenas a ablacao e metricas de clustering (Silhouette / Davies-Bouldin)
    python -m scripts.classify_teams --ablation-only --max-frames 150

    # Gerar video demonstrativo com overlay colorido por time
    python -m scripts.classify_teams --output runs/teams_demo.mp4 --max-frames 150
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.detection.ball import BallDetector, select_ball
from src.teams.classifier import TeamClassifier, TeamVotingManager
from src.tracking.tracker import PlayerTracker, TrackedPlayer, draw_ball
from src.utils import get_logger, iter_frames, load_config, resolve

log = get_logger("classify_teams")


def run_ablation(crops: list[np.ndarray]) -> dict[str, dict[str, float]]:
    """Compara HSV vs SigLIP em tempo de extracao e qualidade de separacao (Silhouette Score)."""
    log.info("Iniciando Ablacao 2 com %d recortes de torso...", len(crops))
    report = {}

    # 1. Baseline HSV
    t0 = time.time()
    clf_hsv = TeamClassifier(method="hsv", dim_reduction="none")
    clf_hsv.fit(crops)
    time_hsv = max(time.time() - t0, 1e-6)
    feats_hsv = clf_hsv.extract_features(crops)
    labels_hsv = clf_hsv.predict(crops)
    sil_hsv = float(silhouette_score(feats_hsv, labels_hsv))
    db_hsv = float(davies_bouldin_score(feats_hsv, labels_hsv))

    report["baseline_hsv"] = {
        "extracao_fps": round(len(crops) / time_hsv, 1),
        "tempo_total_s": round(time_hsv, 3),
        "silhouette_score": round(sil_hsv, 4),        # Maior e melhor [-1, 1]
        "davies_bouldin_index": round(db_hsv, 4),     # Menor e melhor (mais compacto)
    }

    # 2. SigLIP Embeddings + PCA
    t0 = time.time()
    clf_siglip = TeamClassifier(method="siglip", dim_reduction="pca", pca_components=16)
    clf_siglip.fit(crops)
    time_siglip = max(time.time() - t0, 1e-6)
    feats_siglip = clf_siglip.extract_features(crops)
    if clf_siglip.reducer is not None:
        feats_siglip_red = clf_siglip.reducer.transform(feats_siglip)
    else:
        feats_siglip_red = feats_siglip
    labels_siglip = clf_siglip.predict(crops)
    sil_siglip = float(silhouette_score(feats_siglip_red, labels_siglip))
    db_siglip = float(davies_bouldin_score(feats_siglip_red, labels_siglip))

    report["siglip_embeddings"] = {
        "extracao_fps": round(len(crops) / time_siglip, 1),
        "tempo_total_s": round(time_siglip, 3),
        "silhouette_score": round(sil_siglip, 4),
        "davies_bouldin_index": round(db_siglip, 4),
    }

    return report


def draw_team_player(
    frame: np.ndarray,
    player: TrackedPlayer,
    team_id: int,
    team_colors: dict[int, tuple[int, int, int]],
) -> None:
    """Desenha jogador com a cor atribuida ao seu time e rotulo tatico."""
    color = team_colors.get(team_id, (200, 200, 200))
    x1, y1, x2, y2 = [int(v) for v in player.bbox]

    # Caixa delimitadora
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Identificador tático
    if team_id == -1:
        label = f"#{player.track_id} REF"
    elif player.cls_name == "goalkeeper":
        label = f"#{player.track_id} GK (T{team_id})"
    else:
        label = f"#{player.track_id} T{team_id}"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(0, y1)), color, -1)
    # Texto em preto ou branco dependendo do contraste
    text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
    cv2.putText(
        frame,
        label,
        (x1 + 2, max(th, y1 - 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        text_color,
        1,
        cv2.LINE_AA,
    )

    # Ponto no chao (pe)
    bx, by = int(player.bottom_center[0]), int(player.bottom_center[1])
    cv2.circle(frame, (bx, by), 4, color, -1)
    cv2.circle(frame, (bx, by), 5, (0, 0, 0), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Classificacao de times e ablacao HSV vs Embeddings.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--weights", default="runs/detector/weights/best.pt")
    ap.add_argument("--video", default="data/121364_0.mp4")
    ap.add_argument("--output", default="runs/teams_demo.mp4")
    ap.add_argument("--method", choices=["siglip", "hsv"], default="siglip")
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--ablation-only", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    weights_path = resolve(a.weights)
    video_path = resolve(a.video)
    output_path = resolve(a.output)

    if not weights_path.exists() or not video_path.exists():
        log.error("Pesos ou video nao encontrados.")
        return 1

    classes = cfg["detection"]["classes"]
    trk_cfg = cfg.get("tracking", {})
    tms_cfg = cfg.get("teams", {})
    ball_cfg = cfg["ball"]

    tracker = PlayerTracker(
        weights=weights_path,
        all_classes=classes,
        track_classes=trk_cfg.get("track_classes", ["goalkeeper", "player", "referee"]),
        tracker_type=trk_cfg.get("tracker_type", "bytetrack.yaml"),
        conf=trk_cfg.get("conf", 0.25),
        iou=trk_cfg.get("iou", 0.5),
        imgsz=trk_cfg.get("imgsz", 1280),
    )

    # Passo 1: Coleta de recortes de torso dos jogadores nos frames de amostragem
    log.info("Coletando recortes de torso para ajuste do classificador...")
    sample_crops: list[np.ndarray] = []
    frames_cache: list[tuple[int, np.ndarray, list[TrackedPlayer]]] = []

    for idx, frame in iter_frames(video_path):
        if a.max_frames and idx >= a.max_frames:
            break
        tracked = tracker.track_frame(frame, persist=True)
        frames_cache.append((idx, frame, tracked))
        for p in tracked:
            if p.cls_name == "player":
                torso = p.torso_crop(frame)
                if torso.size > 0 and torso.shape[0] >= 10 and torso.shape[1] >= 10:
                    sample_crops.append(torso)
        if len(sample_crops) >= 300:
            break

    if len(sample_crops) < 10:
        log.error("Poucos recortes de torso coletados (%d).", len(sample_crops))
        return 1

    log.info("Total de %d recortes de torso coletados para treino.", len(sample_crops))

    # Executa a Ablação se solicitado
    ablation_results = run_ablation(sample_crops[:200])
    print("\n=== RESULTADOS DA ABLACAO 2 (HSV vs EMBEDDINGS) ===")
    print(json.dumps(ablation_results, indent=2))
    print("===================================================\n")

    if a.ablation_only:
        return 0

    # Passo 2: Treina o classificador escolhido
    log.info("Treinando TeamClassifier com metodo '%s'...", a.method)
    clf = TeamClassifier(
        method=a.method,
        model_name=tms_cfg.get("model_name", "google/siglip-base-patch16-224"),
        dim_reduction=tms_cfg.get("dim_reduction", "pca"),
        pca_components=tms_cfg.get("pca_components", 16),
        k=2,
    )
    clf.fit(sample_crops)

    # Cores vivas distintas para Time 0 e Time 1 para máxima clareza no vídeo
    viz_colors = {
        0: (230, 90, 40),    # Time 0: Azul / Turquesa
        1: (30, 40, 230),    # Time 1: Vermelho / Carmim
        -1: (0, 240, 240),   # Arbitro: Amarelo puro
    }

    # Passo 3: Inicializa detector da bola e gerenciador de voto temporal
    voting_mgr = TeamVotingManager(history_window=60)
    ball_det = BallDetector(
        weights=str(weights_path),
        ball_class_id=classes.index("ball"),
        conf=ball_cfg["conf"],
        imgsz=cfg["detection"]["infer"]["imgsz"],
        slicing=False,
    )

    cap = cv2.VideoCapture(str(video_path))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps_in, (width, height))

    prev_ball: np.ndarray | None = None
    tracker.reset()

    log.info("Processando video com classificacao de times e salvando em %s...", output_path.name)
    n_frames = 0
    t_start = time.time()

    try:
        for idx, frame in iter_frames(video_path):
            if a.max_frames and idx >= a.max_frames:
                break

            tracked = tracker.track_frame(frame, persist=True)

            # 1. Predição dos jogadores de linha do frame atual
            players_line = [p for p in tracked if p.cls_name == "player"]
            crops_line = [p.torso_crop(frame) for p in players_line]

            raw_preds: list[int] = []
            if crops_line:
                raw_preds = list(clf.predict(crops_line))

            # 2. Voto temporal para os jogadores de linha
            assigned_field: list[tuple[TrackedPlayer, int]] = []
            for p, r_team in zip(players_line, raw_preds):
                cons_team = voting_mgr.assign_team(p, raw_team=int(r_team), field_players=[])
                assigned_field.append((p, cons_team))

            # 3. Atribuição de goleiros e árbitros
            frame_assignments: list[tuple[TrackedPlayer, int]] = list(assigned_field)
            for p in tracked:
                if p.cls_name in ("goalkeeper", "referee"):
                    assigned_team = voting_mgr.assign_team(p, raw_team=None, field_players=assigned_field)
                    frame_assignments.append((p, assigned_team))

            # 4. Detecção da bola
            ball_cands = ball_det.candidates(frame)
            best_ball = select_ball(ball_cands, prev_ball, ball_cfg["max_jump_px"])
            ball_pos = None
            if best_ball is not None:
                prev_ball = best_ball[:2]
                ball_pos = (float(best_ball[0]), float(best_ball[1]))

            # 5. Renderização das caixas coloridas por time
            annotated = frame.copy()
            count_t0 = sum(1 for _, t in frame_assignments if t == 0)
            count_t1 = sum(1 for _, t in frame_assignments if t == 1)

            for p, t_id in frame_assignments:
                draw_team_player(annotated, p, t_id, viz_colors)

            annotated = draw_ball(annotated, ball_pos)

            # HUD tático no topo
            hud_text = (
                f"Frame: {idx:04d} | Time 0 (Azul): {count_t0:02d} | "
                f"Time 1 (Vermelho): {count_t1:02d} | Metodo: {a.method.upper()}"
            )
            overlay = annotated.copy()
            cv2.rectangle(overlay, (10, 10), (700, 48), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
            cv2.putText(
                annotated,
                hud_text,
                (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            writer.write(annotated)
            n_frames += 1

            if n_frames % 50 == 0:
                log.info("Renderizados %d/%d frames...", n_frames, a.max_frames)
    finally:
        writer.release()

    total_time = max(time.time() - t_start, 1e-6)
    log.info("Processamento concluido! %d frames em %.2fs (%.1f FPS)", n_frames, total_time, n_frames / total_time)
    log.info("Video anotado com times salvo em: %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
