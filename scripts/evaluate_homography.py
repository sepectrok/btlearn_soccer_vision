"""Avaliação de homografia, execução da Ablação 3 (Com vs Sem suavização temporal) e demo visual.

Uso:
    python -m scripts.evaluate_homography --video data/121364_0.mp4 --max-frames 150 --output runs/homography_demo.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.homography.estimator import HomographyEstimator, project_point
from src.homography.metrics import compute_trajectory_jitter, draw_pitch_wireframe
from src.homography.pitch import PitchModel
from src.tracking.tracker import PlayerTracker
from src.utils import get_logger, iter_frames, load_config, resolve

log = get_logger("evaluate_homography")


def main() -> int:
    ap = argparse.ArgumentParser(description="Avalia homografia e executa Ablacao 3.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--video", default="data/121364_0.mp4")
    ap.add_argument("--weights-detector", default="runs/detector/weights/best.pt")
    ap.add_argument("--output", default="runs/homography_demo.mp4")
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--ablation-only", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    video_path = resolve(a.video)
    output_path = resolve(a.output)
    det_weights = resolve(a.weights_detector)

    if not video_path.exists() or not det_weights.exists():
        log.error("Arquivo de video ou pesos do detector nao encontrados.")
        return 1

    h_cfg = cfg.get("homography", {})
    pitch = PitchModel(
        length=h_cfg.get("pitch_length", 105.0),
        width=h_cfg.get("pitch_width", 68.0),
    )

    log.info("Inicializando estimador de homografia e modelo de keypoints...")
    estimator = HomographyEstimator(
        pitch_model=pitch,
        conf_threshold=h_cfg.get("conf_threshold", 0.35),
        ransac_threshold=h_cfg.get("ransac_threshold", 6.0),
        min_keypoints=h_cfg.get("min_keypoints", 4),
        smooth_alpha=h_cfg.get("smooth_alpha", 0.15),
        max_fallback_frames=h_cfg.get("max_fallback_frames", 30),
        imgsz=h_cfg.get("imgsz", 640),
    )

    log.info("Inicializando rastreador de jogadores...")
    tracker = PlayerTracker(
        weights=det_weights,
        all_classes=cfg["detection"]["classes"],
        track_classes=["goalkeeper", "player", "referee"],
        conf=0.25,
        imgsz=1280,
    )

    cap = cv2.VideoCapture(str(video_path))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps_in, (width, height))

    # Coleta de trajetórias projetadas para a Ablação 3 (com e sem suavização)
    traj_raw: dict[int, list[tuple[float, float]]] = defaultdict(list)
    traj_smooth: dict[int, list[tuple[float, float]]] = defaultdict(list)

    total_frames = 0
    valid_frames = 0
    fallback_frames = 0
    total_keypoints = 0
    total_inliers = 0

    log.info("Iniciando processamento de homografia no video (%d frames)...", a.max_frames)
    t0 = time.time()

    try:
        for idx, frame in iter_frames(video_path):
            if a.max_frames and idx >= a.max_frames:
                break

            total_frames += 1

            # 1. Estima homografia com suavização (para o vídeo) e obtém H_raw (para a ablação)
            res = estimator.estimate(frame, use_smoothing=True)
            H_smooth = res.H
            H_raw = res.H_raw if res.H_raw is not None else H_smooth

            if H_smooth is not None:
                valid_frames += 1
                if res.is_fallback:
                    fallback_frames += 1
                else:
                    total_keypoints += res.num_keypoints
                    total_inliers += res.inliers_count

            # 2. Tracking dos jogadores no frame
            players = tracker.track_frame(frame, persist=True)

            # 3. Projeta os pés dos jogadores para coordenadas métricas do campo (X, Y)
            for p in players:
                foot = p.bottom_center
                if H_raw is not None:
                    p_raw = project_point(foot, H_raw)
                    traj_raw[p.track_id].append(p_raw)
                if H_smooth is not None:
                    p_smooth = project_point(foot, H_smooth)
                    traj_smooth[p.track_id].append(p_smooth)

            # 4. Renderização do vídeo demonstrativo
            annotated = frame.copy()

            # Desenha wireframe do campo reprojetado sobre o gramado real
            if H_smooth is not None:
                annotated = draw_pitch_wireframe(annotated, H_smooth, pitch, color=(0, 255, 120), thickness=2)

            # Desenha marcos detectados no frame
            if res.keypoints_xy is not None and len(res.keypoints_xy) > 0:
                for k_i, (k_xy, k_c) in enumerate(zip(res.keypoints_xy, res.keypoints_conf)):
                    if k_c >= 0.35:
                        kx, ky = int(k_xy[0]), int(k_xy[1])
                        cv2.circle(annotated, (kx, ky), 5, (0, 165, 255), -1)
                        cv2.circle(annotated, (kx, ky), 6, (0, 0, 0), 1)

            # Desenha jogadores e suas coordenadas métricas no campo
            for p in players:
                bx, by = int(p.bottom_center[0]), int(p.bottom_center[1])
                cv2.circle(annotated, (bx, by), 4, (255, 255, 0), -1)
                if H_smooth is not None:
                    m_pt = project_point(p.bottom_center, H_smooth)
                    if pitch.is_inside(m_pt[0], m_pt[1], margin=5.0):
                        coord_lbl = f"{m_pt[0]:.1f}m,{m_pt[1]:.1f}m"
                        cv2.putText(
                            annotated,
                            coord_lbl,
                            (bx - 20, by + 14),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.38,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

            # HUD tático no topo
            status_txt = "FALLBACK" if res.is_fallback else ("OK" if H_smooth is not None else "LOST")
            hud = (
                f"Frame: {idx:04d} | Homografia: {status_txt} | "
                f"Kpts: {res.num_keypoints:02d} | Inliers: {res.inliers_count:02d}"
            )
            overlay = annotated.copy()
            cv2.rectangle(overlay, (10, 10), (620, 48), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
            cv2.putText(annotated, hud, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            writer.write(annotated)

            if total_frames % 50 == 0:
                log.info("Processados %d/%d frames...", total_frames, a.max_frames)
    finally:
        writer.release()

    total_time = max(time.time() - t0, 1e-6)
    fps_homography = total_frames / total_time

    # Cálculo da Ablação 3: Jitter com e sem suavização temporal
    jitters_raw = [compute_trajectory_jitter(np.array(pts)) for pts in traj_raw.values() if len(pts) >= 5]
    jitters_smooth = [compute_trajectory_jitter(np.array(pts)) for pts in traj_smooth.values() if len(pts) >= 5]

    mean_j_raw = float(np.mean(jitters_raw)) if jitters_raw else 0.0
    mean_j_smooth = float(np.mean(jitters_smooth)) if jitters_smooth else 0.0
    jitter_reduction_pct = (
        float((mean_j_raw - mean_j_smooth) / max(mean_j_raw, 1e-6) * 100.0) if mean_j_raw > 0 else 0.0
    )

    n_direct = valid_frames - fallback_frames
    avg_kpts = float(total_keypoints / max(n_direct, 1))
    avg_inliers = float(total_inliers / max(n_direct, 1))

    report = {
        "frames_totais": total_frames,
        "cobertura_homografia": round(valid_frames / max(total_frames, 1), 4),
        "taxa_fallback": round(fallback_frames / max(valid_frames, 1), 4),
        "media_keypoints_por_frame": round(avg_kpts, 1),
        "media_inliers_por_frame": round(avg_inliers, 1),
        "fps_homografia": round(fps_homography, 2),
        "ablacao_3_jitter": {
            "sem_suavizacao_m_per_frame2": round(mean_j_raw, 4),
            "com_suavizacao_m_per_frame2": round(mean_j_smooth, 4),
            "reducao_de_jitter_percentual": f"{jitter_reduction_pct:.2f}%",
        },
    }

    print("\n=== RESULTADOS DA ABLACAO 3 (ESTABILIDADE DA HOMOGRAFIA) ===")
    print(json.dumps(report, indent=2))
    print("============================================================\n")

    log.info("Video com wireframe e coordenadas salvo em: %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
