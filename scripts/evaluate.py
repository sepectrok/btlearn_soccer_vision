"""Avalia o detector (mAP) e/ou faz a ablacao da bola (com/sem slicing + pos-processamento).

Exemplos:
  python -m scripts.evaluate detector --weights runs/detector/weights/best.pt
  python -m scripts.evaluate ball --weights runs/detector/weights/best.pt \
      --video data/clip.mp4 [--gt data/ball_gt.csv] [--max-frames 300]

CSV de GT da bola: colunas frame,x,y (frames sem bola podem ser omitidos).
Sem GT, reportamos apenas a taxa de frames com bola detectada ("cobertura"), que NAO e recall.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.detection.ball import BallDetector, ball_recall, postprocess_trajectory, select_ball
from src.detection.train_eval import evaluate_detector
from src.utils import get_logger, iter_frames, load_config, resolve

log = get_logger("evaluate")


def run_ball(det: BallDetector, video: Path, max_jump: float, max_frames: int | None) -> tuple[np.ndarray, float]:
    traj, prev, t0, n = [], None, time.time(), 0
    for idx, frame in iter_frames(video):
        if max_frames and idx >= max_frames:
            break
        best = select_ball(det.candidates(frame), prev, max_jump)
        if best is not None:
            prev = best[:2]
        traj.append(best[:2] if best is not None else [np.nan, np.nan])
        n += 1
    return np.array(traj, dtype=float), n / max(time.time() - t0, 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["detector", "ball"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video")
    ap.add_argument("--gt")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    cfg = load_config(a.config)

    if a.mode == "detector":
        res = evaluate_detector(resolve(a.weights), resolve(cfg["paths"]["dataset_dir"]) / "data.yaml", a.split)
        print(json.dumps(res, indent=2))
        return

    if not a.video:
        ap.error("--video e obrigatorio no modo ball")
    classes = cfg["detection"]["classes"]
    b, report = cfg["ball"], {}
    for label, slicing in (("sem_slicing", False), ("com_slicing", True)):
        det = BallDetector(str(resolve(a.weights)), classes.index("ball"), b["conf"], cfg["detection"]["infer"]["imgsz"],
                           slicing, tuple(b["slicing"]["slice_wh"]), b["slicing"]["overlap"])
        raw, fps = run_ball(det, a.video, b["max_jump_px"], a.max_frames)
        post, interp = postprocess_trajectory(raw, b["max_jump_px"], b["max_gap_frames"], b["smooth_window"])
        row = {"fps": round(fps, 2),
               "cobertura_bruta": float((~np.isnan(raw).any(axis=1)).mean()),
               "cobertura_pos": float((~np.isnan(post).any(axis=1)).mean()),
               "frames_interpolados": int(interp.sum())}
        if a.gt:
            data = np.genfromtxt(resolve(a.gt), delimiter=",", names=True)
            gt = np.full((len(raw), 2), np.nan)
            for r in np.atleast_1d(data):
                if int(r["frame"]) < len(gt):
                    gt[int(r["frame"])] = (r["x"], r["y"])
            row["recall_bruto@20px"] = ball_recall(raw, gt)
            row["recall_pos@20px"] = ball_recall(post, gt)
        report[label] = row
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
