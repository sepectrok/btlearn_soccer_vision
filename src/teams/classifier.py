"""Classificador de times via K-Means e gerenciador de voto temporal por track ID."""
from __future__ import annotations

from collections import Counter
from typing import Sequence

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.teams.features import SigLIPEmbedder, extract_hsv_features
from src.tracking.tracker import TrackedPlayer
from src.utils import get_logger

log = get_logger(__name__)

# Cores padrão BGR para visualização
DEFAULT_TEAM_COLORS: dict[int, tuple[int, int, int]] = {
    0: (230, 80, 50),     # Time A: Azul vibrante (BGR)
    1: (40, 50, 230),     # Time B: Vermelho/Laranja vibrante (BGR)
    -1: (0, 255, 255),    # Árbitro: Amarelo
}


class TeamClassifier:
    """Classificador não supervisionado de times (K-Means, k=2)."""

    def __init__(
        self,
        method: str = "siglip",
        model_name: str = "google/siglip-base-patch16-224",
        dim_reduction: str = "pca",
        pca_components: int = 16,
        k: int = 2,
        seed: int = 42,
    ) -> None:
        self.method = method.lower()
        self.dim_reduction = dim_reduction.lower()
        self.pca_components = pca_components
        self.k = k
        self.seed = seed

        self.embedder: SigLIPEmbedder | None = None
        if self.method == "siglip":
            self.embedder = SigLIPEmbedder(model_name=model_name)

        self.reducer: PCA | None = None
        self.kmeans = KMeans(n_clusters=self.k, random_state=self.seed, n_init=10)
        self.is_fitted = False
        self.team_colors: dict[int, tuple[int, int, int]] = dict(DEFAULT_TEAM_COLORS)

    def extract_features(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Extrai características com base no método configurado (siglip ou hsv)."""
        if self.method == "siglip":
            if self.embedder is None:
                raise RuntimeError("SigLIPEmbedder nao inicializado.")
            return self.embedder.extract_features(crops)
        elif self.method == "hsv":
            return extract_hsv_features(crops)
        else:
            raise ValueError(f"Metodo desconhecido: {self.method}")

    def fit(self, crops: Sequence[np.ndarray]) -> TeamClassifier:
        """Ajusta o redutor dimensional e o KMeans nas características dos torsos."""
        if len(crops) < self.k:
            raise ValueError(f"Pelo menos {self.k} amostras necessarias para o fit.")

        feats = self.extract_features(crops)

        # Redução dimensional opcional
        if self.dim_reduction == "pca" and feats.shape[1] > self.pca_components:
            n_comp = min(self.pca_components, len(crops), feats.shape[1])
            self.reducer = PCA(n_components=n_comp, random_state=self.seed)
            feats_reduced = self.reducer.fit_transform(feats)
        else:
            self.reducer = None
            feats_reduced = feats

        labels = self.kmeans.fit_predict(feats_reduced)
        self.is_fitted = True

        # Determina a cor media BGR representativa para cada time
        for cluster_id in range(self.k):
            cluster_crops = [crops[i] for i, lbl in enumerate(labels) if lbl == cluster_id and crops[i].size > 0]
            if cluster_crops:
                means = [c.mean(axis=(0, 1)) for c in cluster_crops]
                avg_bgr = np.mean(means, axis=0).astype(int)
                self.team_colors[cluster_id] = (int(avg_bgr[0]), int(avg_bgr[1]), int(avg_bgr[2]))
            else:
                self.team_colors[cluster_id] = DEFAULT_TEAM_COLORS.get(cluster_id, (200, 200, 200))

        log.info("TeamClassifier ajustado com sucesso. Cores medias calculadas: %s", self.team_colors)
        return self

    def predict(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Prediz o time (0 ou 1) para cada recorte de torso."""
        if not self.is_fitted:
            raise RuntimeError("TeamClassifier precisa ser ajustado com fit() antes de predict().")
        if not crops:
            return np.empty(0, dtype=int)

        feats = self.extract_features(crops)
        if self.reducer is not None:
            feats = self.reducer.transform(feats)

        return self.kmeans.predict(feats)


class TeamVotingManager:
    """Gerencia a consistência temporal de times por track ID e regras de goleiro/árbitro."""

    def __init__(self, history_window: int = 50) -> None:
        self.history_window = history_window
        # Mapeamento track_id -> histórico de predições de time [0, 1, 0, 0, ...]
        self.player_votes: dict[int, list[int]] = {}

    def update_player(self, track_id: int, raw_team: int) -> int:
        """Adiciona uma predição instantânea e retorna o time pelo voto por maioria."""
        if track_id not in self.player_votes:
            self.player_votes[track_id] = []

        self.player_votes[track_id].append(raw_team)
        if len(self.player_votes[track_id]) > self.history_window:
            self.player_votes[track_id].pop(0)

        return self.get_consensus(track_id)

    def get_consensus(self, track_id: int, default: int = 0) -> int:
        """Retorna o time consensual (mais votado) para o track ID."""
        votes = self.player_votes.get(track_id, [])
        if not votes:
            return default
        counts = Counter(votes)
        # Mais comum; em caso de empate, o mais recente
        return counts.most_common(1)[0][0]

    def assign_goalkeeper_spatially(
        self,
        gk: TrackedPlayer,
        field_players: list[tuple[TrackedPlayer, int]],
    ) -> int:
        """Atribui o goleiro ao time cujo centroide médio dos jogadores está mais próximo espacialmente."""
        if not field_players:
            return 0

        # Separa posições X dos pés dos jogadores de cada time no frame
        team_0_x = [p.bottom_center[0] for p, tid in field_players if tid == 0]
        team_1_x = [p.bottom_center[0] for p, tid in field_players if tid == 1]

        if not team_0_x and not team_1_x:
            return 0
        if not team_0_x:
            return 1
        if not team_1_x:
            return 0

        c0 = float(np.mean(team_0_x))
        c1 = float(np.mean(team_1_x))
        gk_x = gk.bottom_center[0]

        # O goleiro joga no lado de defesa de sua própria equipe
        dist_0 = abs(gk_x - c0)
        dist_1 = abs(gk_x - c1)

        return 0 if dist_0 < dist_1 else 1

    def assign_team(
        self,
        player: TrackedPlayer,
        raw_team: int | None,
        field_players: list[tuple[TrackedPlayer, int]],
    ) -> int:
        """Aplica regras táticas para atribuir o time definitivo a qualquer ator rastreado.

        - referee: time -1 (árbitro)
        - goalkeeper: proximidade espacial com o time
        - player: voto temporal por track_id
        """
        if player.cls_name == "referee":
            return -1

        if player.cls_name == "goalkeeper":
            return self.assign_goalkeeper_spatially(player, field_players)

        # Jogador de linha
        if raw_team is not None:
            return self.update_player(player.track_id, raw_team)
        return self.get_consensus(player.track_id, default=0)
