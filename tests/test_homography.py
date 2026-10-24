import numpy as np
import pytest

from src.homography.estimator import project_point, project_points
from src.homography.metrics import compute_trajectory_jitter
from src.homography.pitch import PitchModel


def test_pitch_model_geometry():
    pitch = PitchModel()
    verts = pitch.vertices
    assert verts.shape == (32, 2)
    # Cantos
    assert np.allclose(verts[0], (0.0, 0.0))          # Superior esquerdo
    assert np.allclose(verts[5], (0.0, 68.0))         # Inferior esquerdo
    assert np.allclose(verts[24], (105.0, 0.0))       # Superior direito
    assert np.allclose(verts[29], (105.0, 68.0))      # Inferior direito
    # Meio de campo
    assert np.allclose(verts[13], (52.5, 0.0))
    assert np.allclose(verts[16], (52.5, 68.0))

    # Limites
    assert pitch.is_inside(50.0, 30.0)
    assert not pitch.is_inside(110.0, 30.0, margin=0.0)
    assert pitch.is_inside(106.0, 30.0, margin=2.0)
    assert pitch.clip_point(110.0, -5.0) == (105.0, 0.0)


def test_project_points_identity_and_scaling():
    # Matriz identidade 3x3
    H_eye = np.eye(3, dtype=np.float32)
    pts = np.array([[10.0, 20.0], [50.0, 30.0]], dtype=np.float32)
    proj = project_points(pts, H_eye)
    assert np.allclose(proj, pts)

    # Matriz de escala: X * 2, Y * 0.5
    H_scale = np.diag([2.0, 0.5, 1.0]).astype(np.float32)
    proj_scale = project_points(pts, H_scale)
    assert np.allclose(proj_scale, [[20.0, 10.0], [100.0, 15.0]])


def test_project_points_roundtrip():
    # Homografia perspectica sintética bem condicionada
    H = np.array([
        [1.2, 0.1, -50.0],
        [-0.05, 1.1, -30.0],
        [0.0002, 0.0001, 1.0],
    ], dtype=np.float64)

    pts = np.array([[100.0, 200.0], [500.0, 400.0], [800.0, 600.0]], dtype=np.float32)
    proj = project_points(pts, H)
    assert not np.isnan(proj).any()

    # Inversa: deve recuperar os pontos originais com precisão
    H_inv = np.linalg.inv(H)
    pts_recovered = project_points(proj, H_inv)
    assert np.allclose(pts_recovered, pts, atol=1e-3)


def test_project_points_degeneracy_handling():
    # Ponto no infinito / denominador zero
    H_bad = np.zeros((3, 3), dtype=np.float32)
    pts = np.array([[10.0, 20.0]])
    proj = project_points(pts, H_bad)
    assert np.isnan(proj).all()


def test_compute_trajectory_jitter():
    # 1. Movimento retilíneo uniforme (velocidade constante): aceleracao zero -> jitter zero
    t_smooth = np.array([[i * 1.0, i * 2.0] for i in range(10)], dtype=np.float32)
    jitter_smooth = compute_trajectory_jitter(t_smooth)
    assert np.isclose(jitter_smooth, 0.0)

    # 2. Trajetória com ruído/jitter oscilatório
    t_noisy = t_smooth.copy()
    t_noisy[1:-1] += np.random.normal(0, 1.5, size=t_noisy[1:-1].shape)
    jitter_noisy = compute_trajectory_jitter(t_noisy)
    assert jitter_noisy > 0.5

    # 3. Trajetória com menos de 3 pontos retorna 0.0
    assert compute_trajectory_jitter(np.array([[1.0, 1.0], [2.0, 2.0]])) == 0.0
