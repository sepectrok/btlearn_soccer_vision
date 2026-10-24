"""Módulo de rastreamento de posse de bola por proximidade espacial no campo.

Atribui a posse de bola ao jogador/time mais próximo dentro de um raio de 2.0 metros
e computa o percentual acumulado de posse para cada equipe ao longo do tempo.
"""
from __future__ import annotations

import math


class BallPossessionTracker:
    """Rastreador de posse de bola baseado em distância euclidiana no gramado."""

    def __init__(self, possession_radius_m: float = 2.0) -> None:
        """Inicializa o rastreador de posse.

        Args:
            possession_radius_m: Raio máximo em metros entre a bola e os pés de um jogador
                                 para considerar que ele detém o controle da posse (padrão 2.0 m).
        """
        self.possession_radius_m = float(possession_radius_m)
        self.team_frames: dict[int, int] = {0: 0, 1: 0}
        self.contested_frames: int = 0
        self.total_frames: int = 0

        self.current_team: int | None = None
        self.current_player_id: int | None = None
        self.current_distance_m: float | None = None

    def update(
        self,
        ball_pos_m: tuple[float, float] | None,
        players: list[dict],
    ) -> dict[str, float | int | None]:
        """Atualiza a estimativa de posse no frame atual.

        Args:
            ball_pos_m: Coordenadas métricas (X, Y) da bola no campo, ou None se não detectada.
            players: Lista de dicionários contendo informações dos jogadores no frame:
                     [{'track_id': int, 'team_id': int | None, 'pos_m': (x, y), 'is_referee': bool}, ...]

        Returns:
            Dicionário com a posse atual e as porcentagens acumuladas do jogo.
        """
        self.total_frames += 1

        if (
            ball_pos_m is None
            or math.isnan(ball_pos_m[0])
            or math.isnan(ball_pos_m[1])
            or not players
        ):
            self.contested_frames += 1
            self.current_team = None
            self.current_player_id = None
            self.current_distance_m = None
            return self.get_summary()

        bx, by = ball_pos_m
        best_dist = float("inf")
        best_player: dict | None = None

        for p in players:
            # Árbitros não disputam posse
            if p.get("is_referee", False):
                continue

            # Se o jogador não tem time definido (ainda em voting), desconsidera
            if p.get("team_id") not in (0, 1):
                continue

            px, py = p["pos_m"]
            if math.isnan(px) or math.isnan(py):
                continue

            d = math.hypot(px - bx, py - by)
            if d < best_dist:
                best_dist = d
                best_player = p

        if best_player is not None and best_dist <= self.possession_radius_m:
            t_id = best_player["team_id"]
            self.team_frames[t_id] += 1
            self.current_team = t_id
            self.current_player_id = best_player["track_id"]
            self.current_distance_m = round(best_dist, 2)
        else:
            self.contested_frames += 1
            self.current_team = None
            self.current_player_id = None
            self.current_distance_m = round(best_dist, 2) if best_player is not None else None

        return self.get_summary()

    def get_summary(self) -> dict[str, float | int | None]:
        """Retorna o estado atual e o percentual de posse de bola acumulado."""
        controlled_frames = self.team_frames[0] + self.team_frames[1]

        if controlled_frames > 0:
            team_0_share = round((self.team_frames[0] / controlled_frames) * 100.0, 1)
            team_1_share = round((self.team_frames[1] / controlled_frames) * 100.0, 1)
        else:
            team_0_share = 50.0
            team_1_share = 50.0

        if self.total_frames > 0:
            contested_pct = round((self.contested_frames / self.total_frames) * 100.0, 1)
        else:
            contested_pct = 0.0

        return {
            "current_team": self.current_team,
            "current_player_id": self.current_player_id,
            "current_distance_m": self.current_distance_m,
            "team_0_pct": team_0_share,
            "team_1_pct": team_1_share,
            "contested_pct": contested_pct,
            "total_frames": self.total_frames,
        }
