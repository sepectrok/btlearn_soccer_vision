import numpy as np
import pytest

from src.teams.classifier import TeamClassifier, TeamVotingManager
from src.teams.features import extract_hsv_features
from src.tracking.tracker import TrackedPlayer


def test_hsv_feature_extraction():
    # Patch vermelho (BGR: [0, 0, 255])
    red_crop = np.zeros((40, 40, 3), dtype=np.uint8)
    red_crop[:, :] = [0, 0, 255]

    # Patch azul (BGR: [255, 0, 0])
    blue_crop = np.zeros((40, 40, 3), dtype=np.uint8)
    blue_crop[:, :] = [255, 0, 0]

    feats = extract_hsv_features([red_crop, blue_crop], bins=(16, 8))
    assert feats.shape == (2, 128)
    # Vetores devem ser normalizados L2
    assert np.isclose(np.linalg.norm(feats[0]), 1.0)
    assert np.isclose(np.linalg.norm(feats[1]), 1.0)
    # Histogramas de vermelho e azul devem ser bem distintos (distancia euclidiana alta)
    dist = np.linalg.norm(feats[0] - feats[1])
    assert dist > 0.5


def test_hsv_grass_filtering():
    # Patch com metade grama verde (BGR: [0, 180, 0]) e metade camisa amarela (BGR: [0, 255, 255])
    crop = np.zeros((40, 40, 3), dtype=np.uint8)
    crop[:20, :] = [0, 180, 0]       # verde
    crop[20:, :] = [0, 255, 255]     # amarelo

    feat_filtered = extract_hsv_features([crop], filter_grass=True)
    feat_unfiltered = extract_hsv_features([crop], filter_grass=False)
    # Os histogramas devem ser diferentes pois o verde foi suprimido no primeiro
    assert not np.allclose(feat_filtered, feat_unfiltered)


def test_team_classifier_hsv_clustering():
    # Cria 4 amostras do Time Vermelho e 4 amostras do Time Azul com leve ruido
    red_crops = []
    blue_crops = []
    np.random.seed(42)

    for _ in range(6):
        r = np.zeros((30, 30, 3), dtype=np.uint8)
        r[:, :] = [0, 0, 240 + np.random.randint(-10, 10)]
        red_crops.append(r)

        b = np.zeros((30, 30, 3), dtype=np.uint8)
        b[:, :] = [240 + np.random.randint(-10, 10), 0, 0]
        blue_crops.append(b)

    all_crops = red_crops + blue_crops
    clf = TeamClassifier(method="hsv", dim_reduction="none")
    clf.fit(all_crops)

    preds = clf.predict(all_crops)
    # Todos do time vermelho devem ter o mesmo label
    assert len(set(preds[:6])) == 1
    # Todos do time azul devem ter o mesmo label
    assert len(set(preds[6:])) == 1
    # Os labels dos dois times devem ser diferentes (um 0, o outro 1)
    assert preds[0] != preds[6]


def test_team_voting_manager():
    mgr = TeamVotingManager(history_window=10)
    # Jogador 5 recebe maioria de votos 0 com ruído ocasional 1
    for vote in [0, 0, 0, 1, 0, 0, 1, 0]:
        consensus = mgr.update_player(track_id=5, raw_team=vote)
    # O consenso deve ser 0
    assert consensus == 0
    assert mgr.get_consensus(track_id=5) == 0


def test_goalkeeper_spatial_assignment():
    mgr = TeamVotingManager()

    # Time 0 com jogadores na esquerda (X em torno de 200)
    # Time 1 com jogadores na direita (X em torno de 800)
    field_players = [
        (TrackedPlayer(track_id=1, bbox=(180, 200, 220, 300), cls_id=2, cls_name="player", conf=0.9), 0),
        (TrackedPlayer(track_id=2, bbox=(190, 400, 230, 500), cls_id=2, cls_name="player", conf=0.9), 0),
        (TrackedPlayer(track_id=3, bbox=(780, 200, 820, 300), cls_id=2, cls_name="player", conf=0.9), 1),
        (TrackedPlayer(track_id=4, bbox=(790, 400, 830, 500), cls_id=2, cls_name="player", conf=0.9), 1),
    ]

    # Goleiro A no lado esquerdo (X=100) -> deve ser Time 0
    gk_left = TrackedPlayer(track_id=10, bbox=(90, 300, 110, 400), cls_id=1, cls_name="goalkeeper", conf=0.95)
    assert mgr.assign_goalkeeper_spatially(gk_left, field_players) == 0

    # Goleiro B no lado direito (X=900) -> deve ser Time 1
    gk_right = TrackedPlayer(track_id=11, bbox=(890, 300, 910, 400), cls_id=1, cls_name="goalkeeper", conf=0.95)
    assert mgr.assign_goalkeeper_spatially(gk_right, field_players) == 1


def test_referee_assignment():
    mgr = TeamVotingManager()
    ref = TrackedPlayer(track_id=99, bbox=(500, 300, 520, 400), cls_id=3, cls_name="referee", conf=0.9)
    team = mgr.assign_team(ref, raw_team=None, field_players=[])
    assert team == -1  # Arbitro e sempre classe neutra (-1)
