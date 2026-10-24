"""Estimação, suavização temporal e projeção por matriz de homografia (campo 105x68 m)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.homography.pitch import PitchModel
from src.utils import get_logger, resolve

log = get_logger(__name__)


@dataclass
class HomographyResult:
    """Resultado da estimação de homografia para um frame."""

    H: np.ndarray | None                     # Matriz 3x3 suavizada (ou fallback)
    H_raw: np.ndarray | None                 # Matriz 3x3 bruta deste frame
    is_fallback: bool                        # True se reusando ultima matriz valida
    num_keypoints: int                       # Quantidade de pontos detectados
    inliers_count: int                       # Inliers confirmados pelo RANSAC
    keypoints_xy: np.ndarray                 # Shape (N, 2)
    keypoints_conf: np.ndarray               # Shape (N,)


def project_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Projeta pontos 2D usando a matriz de homografia H (3x3).

    Args:
        points: Array de shape (N, 2) com coordenadas (x, y).
        H: Matriz de homografia 3x3.

    Returns:
        Array de shape (N, 2) com pontos projetados. Pontos degenerados retornam NaN.
    """
    if points is None or len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)

    pts = np.atleast_2d(points).astype(np.float64)
    n = len(pts)

    # Coordenadas homogeneas: [x, y, 1]
    homog = np.hstack([pts, np.ones((n, 1), dtype=np.float64)])
    # Transformacao: p' = H * p
    transformed = (H @ homog.T).T

    w = transformed[:, 2]
    valid = np.abs(w) > 1e-7

    out = np.full((n, 2), np.nan, dtype=np.float32)
    out[valid, 0] = (transformed[valid, 0] / w[valid]).astype(np.float32)
    out[valid, 1] = (transformed[valid, 1] / w[valid]).astype(np.float32)

    return out


def project_point(point: tuple[float, float], H: np.ndarray) -> tuple[float, float]:
    """Projeta um único ponto (x, y) com a matriz H."""
    pts = np.array([point], dtype=np.float32)
    proj = project_points(pts, H)
    return (float(proj[0, 0]), float(proj[0, 1]))


class HomographyEstimator:
    """Gerencia a detecção de marcos do campo, cálculo de homografia e estabilização temporal."""

    def __init__(
        self,
        weights: str | Path | None = None,
        pitch_model: PitchModel | None = None,
        conf_threshold: float = 0.35,
        ransac_threshold: float = 6.0,
        min_keypoints: int = 4,
        smooth_alpha: float = 0.15,
        max_fallback_frames: int = 30,
        imgsz: int = 640,
    ) -> None:
        from ultralytics import YOLO

        self.pitch = pitch_model or PitchModel()
        self.conf_threshold = conf_threshold
        self.ransac_threshold = ransac_threshold
        self.min_keypoints = min_keypoints
        self.smooth_alpha = smooth_alpha
        self.max_fallback_frames = max_fallback_frames
        self.imgsz = imgsz

        # Obtém pesos do modelo YOLO pose treinado no campo
        if weights is None:
            from huggingface_hub import hf_hub_download

            weights_file = hf_hub_download(
                repo_id="martinjolif/yolo-football-pitch-detection",
                filename="yolo-football-pitch-detection.pt",
            )
            self.model = YOLO(weights_file)
        else:
            p = resolve(weights)
            if not p.exists():
                raise FileNotFoundError(f"Pesos do modelo de campo nao encontrados: {p}")
            self.model = YOLO(str(p))

        # Estado temporal
        self.last_valid_H: np.ndarray | None = None
        self.smoothed_H: np.ndarray | None = None
        self.fallback_count: int = 0

    def estimate(
        self,
        frame: np.ndarray,
        use_smoothing: bool = True,
    ) -> HomographyResult:
        """Processa um frame e estima a homografia de imagem para metros no campo."""
        if frame is None or frame.size == 0:
            return self._handle_failure()

        results = self.model.predict(frame, conf=0.2, imgsz=self.imgsz, verbose=False)
        if not results or len(results) == 0 or results[0].keypoints is None:
            return self._handle_failure()

        kpts = results[0].keypoints
        if kpts.xy is None or len(kpts.xy) == 0:
            return self._handle_failure()

        xy_all = kpts.xy[0].cpu().numpy()       # (32, 2)
        conf_all = kpts.conf[0].cpu().numpy()   # (32,)

        # Filtra pontos confiáveis
        valid_mask = conf_all >= self.conf_threshold
        num_valid = int(valid_mask.sum())

        if num_valid < self.min_keypoints:
            return self._handle_failure(num_valid, xy_all, conf_all)

        src_pts = xy_all[valid_mask]
        dst_pts = self.pitch.vertices[valid_mask]

        # Calcula a homografia com RANSAC: pixels -> metros no campo
        H_raw, inliers = cv2.findHomography(
            srcPoints=src_pts,
            dstPoints=dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
        )

        n_inliers = int(inliers.sum()) if inliers is not None else 0

        # Validação de não-degeneração
        if (
            H_raw is None
            or H_raw.shape != (3, 3)
            or not np.isfinite(H_raw).all()
            or abs(np.linalg.det(H_raw)) < 1e-9
            or n_inliers < self.min_keypoints
        ):
            return self._handle_failure(num_valid, xy_all, conf_all)

        # Normaliza a matriz para que H[2, 2] == 1.0
        if abs(H_raw[2, 2]) > 1e-9:
            H_raw = H_raw / H_raw[2, 2]

        # Suavização temporal exponencial (EMA)
        if use_smoothing:
            if self.smoothed_H is None:
                self.smoothed_H = H_raw.copy()
            else:
                # Alinhamento de sinal para evitar cancelamento destrutivo (H e -H representam a mesma homografia)
                if np.dot(H_raw.flatten(), self.smoothed_H.flatten()) < 0:
                    H_raw = -H_raw
                # Interpolação ponderada
                self.smoothed_H = self.smooth_alpha * H_raw + (1.0 - self.smooth_alpha) * self.smoothed_H
                if abs(self.smoothed_H[2, 2]) > 1e-9:
                    self.smoothed_H = self.smoothed_H / self.smoothed_H[2, 2]
            H_final = self.smoothed_H.copy()
        else:
            H_final = H_raw.copy()

        # Atualiza histórico de validade
        self.last_valid_H = H_final.copy()
        self.fallback_count = 0

        return HomographyResult(
            H=H_final,
            H_raw=H_raw,
            is_fallback=False,
            num_keypoints=num_valid,
            inliers_count=n_inliers,
            keypoints_xy=xy_all,
            keypoints_conf=conf_all,
        )

    def _handle_failure(
        self,
        num_kpts: int = 0,
        xy: np.ndarray | None = None,
        conf: np.ndarray | None = None,
    ) -> HomographyResult:
        """Aplica estratégia de fallback reutilizando a última homografia válida com flag."""
        if xy is None:
            xy = np.empty((0, 2), dtype=np.float32)
        if conf is None:
            conf = np.empty(0, dtype=np.float32)

        if self.last_valid_H is not None and self.fallback_count < self.max_fallback_frames:
            self.fallback_count += 1
            return HomographyResult(
                H=self.last_valid_H.copy(),
                H_raw=None,
                is_fallback=True,
                num_keypoints=num_kpts,
                inliers_count=0,
                keypoints_xy=xy,
                keypoints_conf=conf,
            )

        return HomographyResult(
            H=None,
            H_raw=None,
            is_fallback=False,
            num_keypoints=num_kpts,
            inliers_count=0,
            keypoints_xy=xy,
            keypoints_conf=conf,
        )

    def reset(self) -> None:
        """Limpa o estado temporal acumulado."""
        self.last_valid_H = None
        self.smoothed_H = None
        self.fallback_count = 0
