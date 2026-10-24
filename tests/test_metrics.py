"""Testes unitários para o módulo de métricas táticas e físicas (Etapa 5)."""
import numpy as np
import pytest

from src.homography.pitch import PitchModel
from src.metrics.minimap import MinimapRenderer
from src.metrics.possession import BallPossessionTracker
from src.metrics.speed_distance import PhysicalMetricsTracker


def test_speed_distance_normal_movement():
    # 25 fps -> 1 frame = 0.04s
    tracker = PhysicalMetricsTracker(fps=25.0, max_speed_kmh=40.0, smooth_window=5)

    # Frame 0: Jogador surge na posição (10.0, 10.0)
    speed, dist = tracker.update(track_id=1, pos_m=(10.0, 10.0), frame_idx=0)
    assert speed == 0.0
    assert dist == 0.0

    # Frame 25 (após exatamente 1 segundo): Jogador se deslocou 5 metros em linha reta (15.0, 10.0)
    # v = (5m / 1s) * 3.6 = 18.0 km/h
    speed, dist = tracker.update(track_id=1, pos_m=(15.0, 10.0), frame_idx=25)
    assert np.isclose(dist, 5.0, atol=1e-2)
    assert np.isclose(speed, 18.0, atol=1e-2)
    assert tracker.get_distance(1) == pytest.approx(5.0, abs=1e-2)


def test_speed_distance_noise_filtering():
    """Verifica se saltos irreais de detecção (> 40 km/h) são filtrados."""
    tracker = PhysicalMetricsTracker(fps=25.0, max_speed_kmh=40.0)

    tracker.update(track_id=1, pos_m=(10.0, 10.0), frame_idx=0)

    # No frame seguinte (0.04s depois), jogador salta 20 metros (teletransporte / erro de detecção)
    # v = (20m / 0.04s) * 3.6 = 1800 km/h >> 40 km/h
    speed, dist = tracker.update(track_id=1, pos_m=(30.0, 10.0), frame_idx=1)

    # O salto foi filtrado: a distância acumulada não deve ter aumentado 20 metros
    assert dist == 0.0
    assert speed == 0.0
    assert tracker.get_distance(1) == 0.0


def test_speed_distance_long_absence_reset():
    """Verifica se ausência longa (> 1s) reseta o rastreamento sem acumular salto."""
    tracker = PhysicalMetricsTracker(fps=25.0, max_speed_kmh=40.0)

    tracker.update(track_id=1, pos_m=(10.0, 10.0), frame_idx=0)
    tracker.update(track_id=1, pos_m=(12.0, 10.0), frame_idx=10) # 2 metros acumulados
    dist_before = tracker.get_distance(1)
    assert np.isclose(dist_before, 2.0, atol=1e-2)

    # Jogador desaparece por 100 frames (> 4 segundos) e reaparece em outro lugar
    speed, dist_after = tracker.update(track_id=1, pos_m=(50.0, 30.0), frame_idx=110)
    # Distância não deve incluir o salto de 40m
    assert dist_after == dist_before


def test_ball_possession_attribution():
    tracker = BallPossessionTracker(possession_radius_m=2.0)

    # Jogador 1 (Time 0) a 1.0m da bola
    # Jogador 2 (Time 1) a 4.0m da bola
    # Árbitro a 0.5m da bola (deve ser ignorado)
    players = [
        {"track_id": 1, "team_id": 0, "pos_m": (50.0, 30.0), "is_referee": False},
        {"track_id": 2, "team_id": 1, "pos_m": (54.0, 30.0), "is_referee": False},
        {"track_id": 99, "team_id": None, "pos_m": (50.5, 30.0), "is_referee": True},
    ]

    ball_pos = (50.8, 30.0) # dist p/ player 1: 0.8m (< 2.0m), dist p/ árbitro: 0.3m

    res = tracker.update(ball_pos, players)
    assert res["current_team"] == 0
    assert res["current_player_id"] == 1
    assert res["current_distance_m"] == pytest.approx(0.8, abs=1e-1)
    assert res["team_0_pct"] == 100.0
    assert res["team_1_pct"] == 0.0


def test_ball_possession_contested():
    tracker = BallPossessionTracker(possession_radius_m=2.0)

    players = [
        {"track_id": 1, "team_id": 0, "pos_m": (50.0, 30.0), "is_referee": False},
        {"track_id": 2, "team_id": 1, "pos_m": (60.0, 30.0), "is_referee": False},
    ]

    # Bola isolada a 5.0m do jogador mais próximo
    ball_pos = (55.0, 30.0)
    res = tracker.update(ball_pos, players)

    assert res["current_team"] is None
    assert res["current_player_id"] is None
    assert res["contested_pct"] == 100.0


def test_minimap_renderer():
    pitch = PitchModel(105.0, 68.0)
    renderer = MinimapRenderer(pitch_model=pitch, width=350, height=227)

    players = [
        {"track_id": 1, "team_id": 0, "pos_m": (10.0, 20.0), "cls_name": "player"},
        {"track_id": 2, "team_id": 1, "pos_m": (60.0, 40.0), "cls_name": "player"},
        {"track_id": 3, "team_id": 0, "pos_m": (2.0, 34.0), "cls_name": "goalkeeper"},
        {"track_id": 4, "team_id": None, "pos_m": (52.5, 34.0), "cls_name": "referee"},
    ]

    radar = renderer.render(players, ball_pos_m=(50.0, 35.0), possession_info={"team_0_pct": 55.0, "team_1_pct": 45.0})
    assert radar.shape == (227, 350, 3)
    assert radar.dtype == np.uint8

    # Testa sobreposição no frame do vídeo
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    overlaid = renderer.overlay_on_frame(dummy_frame, radar, position="top_right")
    assert overlaid.shape == (1080, 1920, 3)
    # Confirma que a região superior direita não está toda preta
    assert np.any(overlaid[20:247, 1920-350-20:1920-20] > 0)
