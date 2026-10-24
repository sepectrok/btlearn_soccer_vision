"""Pipeline completo de análise tática de futebol (Etapas 1 a 5).

Integra de ponta a ponta:
1. Detecção de jogadores, goleiros, árbitros e bola (YOLOv8s).
2. Rastreamento temporal dos jogadores (ByteTrack).
3. Classificação de times com voto temporal e atribuição de goleiro/árbitro.
4. Homografia com modelo de keypoints do campo e suavização temporal EMA.
5. Métricas físicas (velocidade instantânea, filtro >40 km/h, distância percorrida).
6. Métrica tática de posse de bola (proximidade espacial <= 2.0 m).
7. Renderização de minimapa 2D estilo transmissão e painel tático de métricas.

Uso:
    python -m scripts.run_pipeline --video data/121364_0.mp4 --max-frames 150 --output runs/full_tactical_analysis.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.detection.ball import BallDetector, select_ball
from src.homography.estimator import HomographyEstimator, project_point
from src.homography.metrics import draw_pitch_wireframe
from src.homography.pitch import PitchModel
from src.metrics.minimap import MinimapRenderer
from src.metrics.possession import BallPossessionTracker
from src.metrics.speed_distance import PhysicalMetricsTracker
from src.teams.classifier import TeamClassifier, TeamVotingManager
from src.tracking.tracker import PlayerTracker, TrackedPlayer, draw_ball
from src.utils import get_logger, iter_frames, load_config, resolve

log = get_logger("run_pipeline")


def fit_team_classifier(
    video_path: Path,
    tracker: PlayerTracker,
    clf: TeamClassifier,
    warmup_frames: int = 30,
) -> None:
    """Coleta crops de torso nos primeiros frames para ajustar o classificador de times."""
    log.info("Coletando amostras de torso nos primeiros %d frames para ajustar times...", warmup_frames)
    crops = []
    for idx, frame in iter_frames(video_path):
        if idx >= warmup_frames:
            break
        players = tracker.track_frame(frame, persist=False)
        for p in players:
            if p.cls_name == "player":
                crop = p.torso_crop(frame)
                if crop.size > 0 and crop.shape[0] >= 10 and crop.shape[1] >= 10:
                    crops.append(crop)

    if len(crops) < 10:
        log.warning("Poucas amostras de torso (%d); usando inicializacao sintetica padrao.", len(crops))
        dummy = [np.random.randint(0, 255, (30, 20, 3), dtype=np.uint8) for _ in range(10)]
        clf.fit(dummy)
    else:
        log.info("Ajustando classificador KMeans com %d amostras de torso...", len(crops))
        clf.fit(crops)


def draw_hud(
    frame: np.ndarray,
    possession_summary: dict,
    total_dist_t0: float,
    total_dist_t1: float,
    top_speed: float,
    frame_idx: int,
    fps: float,
) -> None:
    """Desenha a barra superior de informações táticas da partida (estilo transmissão)."""
    fh, fw = frame.shape[:2]
    bar_h = 44
    hud = frame[0:bar_h, 0:fw].copy()
    cv2.rectangle(hud, (0, 0), (fw, bar_h), (15, 20, 18), -1)
    frame[0:bar_h, 0:fw] = cv2.addWeighted(hud, 0.85, frame[0:bar_h, 0:fw], 0.15, 0)
    cv2.line(frame, (0, bar_h), (fw, bar_h), (80, 95, 85), 1)

    t0_pct = possession_summary.get("team_0_pct", 50.0)
    t1_pct = possession_summary.get("team_1_pct", 50.0)
    time_str = f"{int(frame_idx / fps):02d}:{int(frame_idx % fps * 4):02d}"

    # 1. Título do jogo e tempo
    cv2.putText(frame, "ANALISE TATICA", (20, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"TEMPO {time_str}", (210, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 200, 190), 1, cv2.LINE_AA)

    # 2. Posse de bola central
    cx = fw // 2 - 160
    cv2.putText(frame, f"TIME A  {t0_pct:.0f}%", (cx - 110, 28), cv2.FONT_HERSHEY_DUPLEX, 0.52, (80, 80, 240), 1, cv2.LINE_AA)
    cv2.putText(frame, "POSSE", (cx + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 210, 205), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{t1_pct:.0f}%  TIME B", (cx + 80, 28), cv2.FONT_HERSHEY_DUPLEX, 0.52, (60, 230, 80), 1, cv2.LINE_AA)

    # 3. Métricas físicas acumuladas
    rx = fw - 380
    info_text = f"Dist: A {total_dist_t0:.0f}m | B {total_dist_t1:.0f}m   Top: {top_speed:.1f} km/h"
    cv2.putText(frame, info_text, (rx, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(description="Executa o pipeline tático completo de futebol.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--video", default="data/121364_0.mp4")
    ap.add_argument("--weights-detector", default="runs/detector/weights/best.pt")
    ap.add_argument("--output", default="runs/full_tactical_analysis.mp4")
    ap.add_argument("--metrics-output", default="runs/tactical_metrics.json")
    ap.add_argument("--max-frames", type=int, default=150)
    a = ap.parse_args()

    cfg = load_config(a.config)
    video_path = resolve(a.video)
    det_weights = resolve(a.weights_detector)
    output_path = resolve(a.output)
    metrics_path = resolve(a.metrics_output)

    if not video_path.exists():
        log.error("Vídeo não encontrado: %s", video_path)
        return 1
    if not det_weights.exists():
        log.error("Pesos do detector não encontrados: %s", det_weights)
        return 1

    # Inicialização dos módulos
    cap = cv2.VideoCapture(str(video_path))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    h_cfg = cfg.get("homography", {})
    m_cfg = cfg.get("metrics", {})
    pitch = PitchModel(
        length=h_cfg.get("pitch_length", 105.0),
        width=h_cfg.get("pitch_width", 68.0),
    )

    log.info("Inicializando componentes do pipeline...")
    # Tracker (Etapa 2)
    tracker = PlayerTracker(
        weights=det_weights,
        all_classes=cfg["detection"]["classes"],
        track_classes=cfg["tracking"]["track_classes"],
        conf=cfg["tracking"]["conf"],
        imgsz=cfg["tracking"]["imgsz"],
    )

    # Ball Detector (Etapa 1)
    ball_cfg = cfg.get("ball", {})
    ball_detector = BallDetector(
        weights=str(det_weights),
        ball_class_id=cfg["detection"]["classes"].index("ball"),
        conf=ball_cfg.get("conf", 0.10),
        imgsz=cfg["detection"]["infer"]["imgsz"],
        slicing=False,
    )
    prev_ball_pos: np.ndarray | None = None

    # Team Classifier (Etapa 3)
    team_cfg = cfg.get("teams", {})
    clf = TeamClassifier(
        method=team_cfg.get("method", "hsv"),
        dim_reduction=team_cfg.get("dim_reduction", "none"),
        k=team_cfg.get("k", 2),
    )
    fit_team_classifier(video_path, tracker, clf, warmup_frames=25)
    voting_mgr = TeamVotingManager(history_window=50)

    # Homography Estimator (Etapa 4)
    estimator = HomographyEstimator(
        pitch_model=pitch,
        conf_threshold=h_cfg.get("conf_threshold", 0.35),
        ransac_threshold=h_cfg.get("ransac_threshold", 6.0),
        min_keypoints=h_cfg.get("min_keypoints", 4),
        smooth_alpha=h_cfg.get("smooth_alpha", 0.15),
        max_fallback_frames=h_cfg.get("max_fallback_frames", 30),
        imgsz=h_cfg.get("imgsz", 640),
    )

    # Physical & Tactical Metrics (Etapa 5)
    phys_tracker = PhysicalMetricsTracker(
        fps=fps_in,
        max_speed_kmh=m_cfg.get("max_speed_kmh", 40.0),
        smooth_window=m_cfg.get("speed_smooth_window", 5),
    )
    poss_tracker = BallPossessionTracker(
        possession_radius_m=m_cfg.get("possession_radius_m", 2.0)
    )

    # Minimap Renderer (Etapa 5)
    mini_cfg = m_cfg.get("minimap", {})
    minimap_renderer = MinimapRenderer(
        pitch_model=pitch,
        width=mini_cfg.get("width", 350),
        height=mini_cfg.get("height", 227),
        alpha=mini_cfg.get("alpha", 0.85),
        margin=mini_cfg.get("margin", 20),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps_in, (width, height))

    team_colors = {
        0: (50, 50, 230),    # Time 0 (Vermelho)
        1: (50, 220, 70),    # Time 1 (Verde)
        -1: (220, 220, 220), # Árbitro
    }

    log.info("Iniciando processamento integrado do vídeo (%d frames)...", a.max_frames)
    t_start = time.time()
    processed_count = 0
    top_match_speed = 0.0

    try:
        for idx, frame in iter_frames(video_path):
            if a.max_frames and idx >= a.max_frames:
                break
            processed_count += 1

            # 1. Homografia do campo
            h_res = estimator.estimate(frame, use_smoothing=True)
            H = h_res.H

            # 2. Tracking dos jogadores e árbitros
            tracked_players = tracker.track_frame(frame, persist=True)

            # 3. Classificação e votação de time
            frame_crops = []
            crop_players = []
            for p in tracked_players:
                if p.cls_name == "player":
                    c = p.torso_crop(frame)
                    if c.size > 0 and c.shape[0] >= 10 and c.shape[1] >= 10:
                        frame_crops.append(c)
                        crop_players.append(p)

            pred_teams: dict[int, int] = {}
            if frame_crops and clf.is_fitted:
                preds = clf.predict(frame_crops)
                for p, team_pred in zip(crop_players, preds):
                    pred_teams[p.track_id] = int(team_pred)

            # Resolve primeiro jogadores de linha
            field_players: list[tuple[TrackedPlayer, int]] = []
            assigned_teams: dict[int, int] = {}
            for p in tracked_players:
                if p.cls_name == "player":
                    raw = pred_teams.get(p.track_id)
                    tid = voting_mgr.assign_team(p, raw, [])
                    assigned_teams[p.track_id] = tid
                    field_players.append((p, tid))

            # Em seguida goleiros e árbitros
            for p in tracked_players:
                if p.cls_name != "player":
                    tid = voting_mgr.assign_team(p, None, field_players)
                    assigned_teams[p.track_id] = tid

            # 4. Projeção métrica no campo e cálculo de velocidade/distância
            player_info_list = []
            total_dist_t0 = 0.0
            total_dist_t1 = 0.0

            for p in tracked_players:
                t_id = assigned_teams.get(p.track_id)
                pos_m = (float("nan"), float("nan"))
                speed_kmh = 0.0
                dist_m = 0.0

                if H is not None:
                    pos_m = project_point(p.bottom_center, H)
                    speed_kmh, dist_m = phys_tracker.update(p.track_id, pos_m, idx)
                    if speed_kmh > top_match_speed:
                        top_match_speed = speed_kmh

                    if t_id == 0:
                        total_dist_t0 += phys_tracker.get_distance(p.track_id)
                    elif t_id == 1:
                        total_dist_t1 += phys_tracker.get_distance(p.track_id)

                player_info_list.append({
                    "track_id": p.track_id,
                    "team_id": t_id,
                    "pos_m": pos_m,
                    "speed_kmh": speed_kmh,
                    "dist_m": dist_m,
                    "cls_name": p.cls_name,
                    "is_referee": (p.cls_name == "referee"),
                    "is_goalkeeper": (p.cls_name == "goalkeeper"),
                    "bbox": p.bbox,
                    "bottom_center": p.bottom_center,
                })

            # 5. Detecção e projeção da bola
            ball_cands = ball_detector.candidates(frame)
            best_ball = select_ball(ball_cands, prev_ball_pos, ball_cfg.get("max_jump_px", 150))
            ball_pos = None
            ball_pos_m = None
            if best_ball is not None:
                ball_pos = (float(best_ball[0]), float(best_ball[1]))
                prev_ball_pos = np.array(ball_pos)
                if H is not None:
                    ball_pos_m = project_point(ball_pos, H)

            # 6. Atualização de posse de bola
            poss_summary = poss_tracker.update(ball_pos_m, player_info_list)

            # 7. Renderização das anotações no frame
            annotated = frame.copy()

            # Wireframe suave das linhas do campo
            if H is not None:
                annotated = draw_pitch_wireframe(annotated, H, pitch, color=(0, 220, 100), thickness=2)

            # Jogadores: caixas, cores do time e HUD individual (Velocidade e Distância)
            for p_info in player_info_list:
                t_id = p_info["team_id"]
                color = team_colors.get(t_id, (200, 200, 200))
                if p_info["is_goalkeeper"]:
                    color = (0, 215, 255)
                elif p_info["is_referee"]:
                    color = (230, 230, 230)

                x1, y1, x2, y2 = [int(v) for v in p_info["bbox"]]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Rótulo de identificação
                if p_info["is_referee"]:
                    lbl = f"REF #{p_info['track_id']}"
                elif p_info["is_goalkeeper"]:
                    lbl = f"GK (T{t_id})"
                else:
                    lbl = f"T{t_id} #{p_info['track_id']}"

                (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                cv2.rectangle(annotated, (x1, max(0, y1 - lh - 4)), (x1 + lw + 2, max(0, y1)), color, -1)
                txt_col = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
                cv2.putText(annotated, lbl, (x1 + 1, max(lh, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, txt_col, 1, cv2.LINE_AA)

                # Rótulo inferior com velocidade e distância física
                bx, by = int(p_info["bottom_center"][0]), int(p_info["bottom_center"][1])
                cv2.circle(annotated, (bx, by), 4, color, -1)
                if not math.isnan(p_info["pos_m"][0]):
                    stat_lbl = f"{p_info['speed_kmh']:.1f}km/h | {p_info['dist_m']:.1f}m"
                    cv2.putText(annotated, stat_lbl, (bx - 25, by + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

            # Bola destacada
            if ball_pos is not None:
                annotated = draw_ball(annotated, ball_pos)

            # Minimapa 2D
            radar_img = minimap_renderer.render(player_info_list, ball_pos_m, poss_summary)
            annotated = minimap_renderer.overlay_on_frame(annotated, radar_img, position=mini_cfg.get("position", "top_right"))

            # Barra HUD superior estilo transmissão
            draw_hud(annotated, poss_summary, total_dist_t0, total_dist_t1, top_match_speed, idx, fps_in)

            writer.write(annotated)

            if processed_count % 30 == 0:
                log.info("Processados %d/%d frames...", processed_count, a.max_frames)

    finally:
        writer.release()

    elapsed = max(time.time() - t_start, 1e-6)
    fps_pipe = processed_count / elapsed
    log.info("Pipeline concluído em %.2fs (%d frames, %.1f FPS).", elapsed, processed_count, fps_pipe)
    log.info("Vídeo final salvo em: %s", output_path)

    # Relatório de estatísticas finais
    final_report = {
        "frames_processados": processed_count,
        "tempo_total_s": round(elapsed, 2),
        "fps_pipeline": round(fps_pipe, 1),
        "posse_de_bola": {
            "time_0_pct": poss_summary.get("team_0_pct", 50.0),
            "time_1_pct": poss_summary.get("team_1_pct", 50.0),
            "tempo_disputado_pct": poss_summary.get("contested_pct", 0.0),
        },
        "distancia_total_equipes_m": {
            "time_0": round(total_dist_t0, 1),
            "time_1": round(total_dist_t1, 1),
        },
        "velocidade_maxima_jogo_kmh": round(top_match_speed, 1),
        "jogadores": phys_tracker.get_all_metrics(),
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    log.info("Métricas táticas salvas em: %s", metrics_path)

    print("\n" + "=" * 60)
    print("=== RELATÓRIO DO PIPELINE TÁTICO COMPLETO (ETAPA 5) ===")
    print(json.dumps({k: v for k, v in final_report.items() if k != "jogadores"}, indent=2, ensure_ascii=False))
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
