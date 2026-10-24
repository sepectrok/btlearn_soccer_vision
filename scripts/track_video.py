"""Processa um video gerando o rastreamento dos jogadores e da bola com overlay.

Uso:
    python -m scripts.track_video --weights runs/detector/weights/best.pt --video data/121364_0.mp4 --max-frames 300
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.detection.ball import BallDetector, select_ball
from src.tracking.tracker import PlayerTracker, draw_ball, draw_player_tracks
from src.utils import get_logger, iter_frames, load_config, resolve

log = get_logger("track_video")


def main() -> int:
    ap = argparse.ArgumentParser(description="Executa tracking em video e salva demo anotado.")
    ap.add_argument("--config", default="configs/default.yaml", help="Caminho do arquivo YAML.")
    ap.add_argument("--weights", default="runs/detector/weights/best.pt", help="Pesos do detector treinado.")
    ap.add_argument("--video", default="data/121364_0.mp4", help="Caminho do video de entrada.")
    ap.add_argument("--output", default="runs/tracking_demo.mp4", help="Caminho do video de saida.")
    ap.add_argument("--max-frames", type=int, default=None, help="Limite maximo de frames a processar.")
    ap.add_argument("--stride", type=int, default=1, help="Pular frames (stride=1 para todos).")
    a = ap.parse_args()

    cfg = load_config(a.config)
    weights_path = resolve(a.weights)
    video_path = resolve(a.video)
    output_path = resolve(a.output)

    if not weights_path.exists():
        log.error("Pesos nao encontrados: %s", weights_path)
        return 1
    if not video_path.exists():
        log.error("Video nao encontrado: %s", video_path)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Informações do vídeo de entrada
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error("Nao foi possivel abrir o video: %s", video_path)
        return 1
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    classes = cfg["detection"]["classes"]
    trk_cfg = cfg.get("tracking", {})
    ball_cfg = cfg["ball"]

    log.info("Inicializando PlayerTracker (ByteTrack)...")
    tracker = PlayerTracker(
        weights=weights_path,
        all_classes=classes,
        track_classes=trk_cfg.get("track_classes", ["goalkeeper", "player", "referee"]),
        tracker_type=trk_cfg.get("tracker_type", "bytetrack.yaml"),
        conf=trk_cfg.get("conf", 0.25),
        iou=trk_cfg.get("iou", 0.5),
        imgsz=trk_cfg.get("imgsz", 1280),
    )

    log.info("Inicializando BallDetector...")
    ball_det = BallDetector(
        weights=str(weights_path),
        ball_class_id=classes.index("ball"),
        conf=ball_cfg["conf"],
        imgsz=cfg["detection"]["infer"]["imgsz"],
        slicing=False,  # Decisao da Etapa 1: sem slicing para garantir tempo real
    )

    # Configuração do VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps_in / a.stride, (width, height))

    prev_ball_pos: np.ndarray | None = None
    all_seen_ids: set[int] = set()
    frames_processed = 0
    t_start = time.time()

    log.info(
        "Iniciando tracking em %s (%dx%d, %.1f FPS)...",
        video_path.name,
        width,
        height,
        fps_in,
    )

    try:
        for idx, frame in iter_frames(video_path, stride=a.stride):
            if a.max_frames and idx >= a.max_frames:
                break

            t_frame = time.time()

            # 1. Tracking dos jogadores (ByteTrack)
            tracked_players = tracker.track_frame(frame, persist=True)
            for p in tracked_players:
                all_seen_ids.add(p.track_id)

            # 2. Deteccao da bola (online com filtro de salto maximo)
            ball_cands = ball_det.candidates(frame)
            best_ball = select_ball(ball_cands, prev_ball_pos, ball_cfg["max_jump_px"])
            ball_pos = None
            if best_ball is not None:
                prev_ball_pos = best_ball[:2]
                ball_pos = (float(best_ball[0]), float(best_ball[1]))

            # 3. Desenho de anotações
            annotated = draw_player_tracks(
                frame,
                tracked_players,
                history=tracker.history,
                draw_trails=True,
                draw_feet=True,
            )
            annotated = draw_ball(annotated, ball_pos)

            # 4. Banner informativo no topo
            dt = max(time.time() - t_frame, 1e-6)
            current_fps = 1.0 / dt
            hud_text = (
                f"Frame: {idx:04d} | Tracks Ativos: {len(tracked_players):02d} | "
                f"Total IDs: {len(all_seen_ids):02d} | FPS: {current_fps:4.1f}"
            )
            # Fundo semitransparente para o HUD
            overlay = annotated.copy()
            cv2.rectangle(overlay, (10, 10), (620, 48), (20, 20, 20), -1)
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
            frames_processed += 1

            if frames_processed % 50 == 0:
                elapsed = time.time() - t_start
                avg_fps = frames_processed / max(elapsed, 1e-6)
                log.info(
                    "Progresso: %d/%s frames | FPS medio: %.1f | IDs unicos: %d",
                    frames_processed,
                    a.max_frames or total_video_frames,
                    avg_fps,
                    len(all_seen_ids),
                )
    finally:
        writer.release()

    total_time = max(time.time() - t_start, 1e-6)
    pipeline_fps = frames_processed / total_time

    log.info("Tracking concluido com sucesso!")
    log.info("Frames processados: %d", frames_processed)
    log.info("Tempo total: %.2fs (%.1f FPS)", total_time, pipeline_fps)
    log.info("Total de IDs unicos identificados: %d", len(all_seen_ids))
    log.info("Video salvo em: %s", output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
