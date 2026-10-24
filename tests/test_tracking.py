import numpy as np
import pytest

from src.tracking.tracker import (
    CLASS_COLORS,
    PlayerTracker,
    TrackedPlayer,
    draw_ball,
    draw_player_tracks,
)


def test_tracked_player_geometry():
    p = TrackedPlayer(
        track_id=1,
        bbox=(100.0, 200.0, 160.0, 300.0),
        cls_id=2,
        cls_name="player",
        conf=0.92,
    )
    assert p.width == 60.0
    assert p.height == 100.0
    assert p.center == (130.0, 250.0)
    # Ponto no chao: x medio, y inferior
    assert p.bottom_center == (130.0, 300.0)


def test_tracked_player_crops():
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    frame[200:300, 100:160] = 255
    p = TrackedPlayer(
        track_id=1,
        bbox=(100.0, 200.0, 160.0, 300.0),
        cls_id=2,
        cls_name="player",
        conf=0.92,
    )
    crop = p.crop(frame)
    assert crop.shape == (100, 60, 3)

    torso = p.torso_crop(frame)
    # Altura do torso e 40% da altura total (55% - 15%)
    assert torso.shape[0] == 40
    assert torso.shape[1] == 60


def test_tracked_player_crop_clipping():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Bbox parcialmente fora da imagem
    p = TrackedPlayer(
        track_id=2,
        bbox=(-20.0, -10.0, 150.0, 120.0),
        cls_id=2,
        cls_name="player",
        conf=0.85,
    )
    crop = p.crop(frame)
    assert crop.shape == (100, 100, 3)


def test_draw_player_tracks_and_ball():
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    players = [
        TrackedPlayer(track_id=10, bbox=(50.0, 50.0, 100.0, 150.0), cls_id=2, cls_name="player", conf=0.9),
        TrackedPlayer(track_id=11, bbox=(200.0, 80.0, 250.0, 180.0), cls_id=1, cls_name="goalkeeper", conf=0.88),
    ]
    history = {10: [(75.0, 150.0), (76.0, 150.0)]}

    drawn = draw_player_tracks(canvas, players, history=history, draw_trails=True, draw_feet=True)
    assert drawn.shape == canvas.shape
    assert drawn.dtype == np.uint8
    # Canvas nao deve ficar completamente preto apos o desenho
    assert drawn.sum() > 0

    # Desenho da bola valida
    with_ball = draw_ball(drawn, (300.0, 200.0))
    assert with_ball.sum() > drawn.sum()

    # Bola com NaN ou None nao altera a imagem
    assert np.array_equal(draw_ball(drawn, None), drawn)
    assert np.array_equal(draw_ball(drawn, (np.nan, np.nan)), drawn)


def test_player_tracker_target_classes():
    # Verifica que classes de tracking excluem a bola
    all_classes = ["ball", "goalkeeper", "player", "referee"]
    track_classes = ["goalkeeper", "player", "referee"]
    
    # Com arquivo inexistente deve levantar FileNotFoundError
    with pytest.raises(FileNotFoundError):
        PlayerTracker(
            weights="runs/detector/weights/non_existent.pt",
            all_classes=all_classes,
            track_classes=track_classes,
        )
