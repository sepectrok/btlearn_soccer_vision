"""Métricas de estabilidade de homografia (Jitter) e projeção de wireframe do campo."""
from __future__ import annotations

import cv2
import numpy as np

from src.homography.estimator import project_points
from src.homography.pitch import PitchModel


def compute_trajectory_jitter(trajectory: np.ndarray) -> float:
    """Calcula o jitter médio (magnitude da aceleração temporal / segunda derivada).

    Args:
        trajectory: Array de coordenadas (N, 2) em metros no campo ao longo do tempo.

    Returns:
        Valor escalar do jitter em metros/frame^2. Valores menores indicam trajetórias mais estáveis.
    """
    if trajectory is None or len(trajectory) < 3:
        return 0.0

    traj = np.atleast_2d(trajectory).astype(np.float64)
    # Segunda derivada finita: a_t = p_{t+2} - 2*p_{t+1} + p_t
    diff1 = np.diff(traj, axis=0)
    accel = np.diff(diff1, axis=0)

    # Filtra NaNs
    valid = ~np.isnan(accel).any(axis=1)
    if not valid.any():
        return 0.0

    accel_magnitudes = np.linalg.norm(accel[valid], axis=1)
    return float(np.mean(accel_magnitudes))


def draw_pitch_wireframe(
    frame: np.ndarray,
    H: np.ndarray | None,
    pitch: PitchModel,
    color: tuple[int, int, int] = (0, 255, 120),
    thickness: int = 2,
) -> np.ndarray:
    """Projeta as linhas do modelo de campo de volta sobre a imagem da câmera.

    Permite inspeção visual direta da qualidade e do alinhamento da calibração.
    """
    if H is None or frame is None or frame.size == 0:
        return frame

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return frame

    out = frame.copy()
    h, w = frame.shape[:2]

    # 1. Projeta os vértices físicos para pixels na imagem
    verts_pitch = pitch.vertices  # (32, 2)
    verts_img = project_points(verts_pitch, H_inv)

    # 2. Desenha as linhas que conectam os vértices
    for i1, i2 in pitch.edges:
        pt1 = verts_img[i1]
        pt2 = verts_img[i2]
        if np.isnan(pt1).any() or np.isnan(pt2).any():
            continue
        p1 = (int(pt1[0]), int(pt1[1]))
        p2 = (int(pt2[0]), int(pt2[1]))

        # Desenha apenas se pelo menos um ponto estiver próximo do campo visual
        if (-500 <= p1[0] <= w + 500 and -500 <= p1[1] <= h + 500) or (
            -500 <= p2[0] <= w + 500 and -500 <= p2[1] <= h + 500
        ):
            cv2.line(out, p1, p2, color, thickness, cv2.LINE_AA)

    # 3. Desenha círculo central interpolado
    c_x, c_y = pitch.length / 2.0, pitch.width / 2.0
    r = pitch.centre_circle_radius
    angles = np.linspace(0, 2 * np.pi, 36)
    circle_pts_pitch = np.stack([c_x + r * np.cos(angles), c_y + r * np.sin(angles)], axis=1)
    circle_pts_img = project_points(circle_pts_pitch, H_inv)

    for i in range(len(circle_pts_img) - 1):
        pt1, pt2 = circle_pts_img[i], circle_pts_img[i + 1]
        if np.isnan(pt1).any() or np.isnan(pt2).any():
            continue
        cv2.line(
            out,
            (int(pt1[0]), int(pt1[1])),
            (int(pt2[0]), int(pt2[1])),
            color,
            thickness,
            cv2.LINE_AA,
        )

    return out
