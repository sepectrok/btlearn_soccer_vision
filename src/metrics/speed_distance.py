"""Módulo de cálculo de métricas físicas de jogadores (velocidade e distância).

Calcula a velocidade instantânea em km/h com filtro para ruídos físicos (> 40 km/h)
e acumula a distância percorrida em metros por track_id.
"""
from __future__ import annotations

from collections import defaultdict, deque
import math


class PhysicalMetricsTracker:
    """Rastreador de métricas físicas para jogadores a partir de coordenadas métricas do campo."""

    def __init__(
        self,
        fps: float = 25.0,
        max_speed_kmh: float = 40.0,
        smooth_window: int = 5,
    ) -> None:
        """Inicializa o rastreador de métricas.

        Args:
            fps: Taxa de quadros do vídeo (padrão 25.0).
            max_speed_kmh: Velocidade máxima fisicamente plausível para um jogador (padrão 40.0 km/h).
            smooth_window: Tamanho da janela móvel para suavizar a velocidade exibida.
        """
        self.fps = max(1.0, float(fps))
        self.max_speed_kmh = float(max_speed_kmh)
        self.smooth_window = max(1, int(smooth_window))

        # Estado por track_id:
        # last_pos: (x_m, y_m)
        self._last_pos: dict[int, tuple[float, float]] = {}
        self._last_frame_idx: dict[int, int] = {}
        self._total_distance: dict[int, float] = defaultdict(float)
        self._recent_speeds: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.smooth_window)
        )

    def update(
        self,
        track_id: int,
        pos_m: tuple[float, float],
        frame_idx: int,
    ) -> tuple[float, float]:
        """Atualiza a posição métrica de um jogador e retorna (velocidade_kmh, distancia_m).

        Args:
            track_id: Identificador único do jogador pelo tracker.
            pos_m: Posição (X, Y) em metros no sistema de coordenadas do campo.
            frame_idx: Índice sequencial do frame.

        Returns:
            Tupla (velocidade_suavizada_kmh, distancia_acumulada_m).
        """
        if math.isnan(pos_m[0]) or math.isnan(pos_m[1]):
            return (self.get_speed(track_id), self._total_distance[track_id])

        if track_id not in self._last_pos:
            self._last_pos[track_id] = pos_m
            self._last_frame_idx[track_id] = frame_idx
            return (0.0, 0.0)

        prev_pos = self._last_pos[track_id]
        prev_frame = self._last_frame_idx[track_id]
        delta_frames = frame_idx - prev_frame

        if delta_frames <= 0:
            return (self.get_speed(track_id), self._total_distance[track_id])

        # Se o jogador ficou ausente por mais de 1 segundo (> 25 frames), não acumula salto de teletransporte
        if delta_frames > int(self.fps):
            self._last_pos[track_id] = pos_m
            self._last_frame_idx[track_id] = frame_idx
            return (0.0, self._total_distance[track_id])

        delta_t = delta_frames / self.fps
        dx = pos_m[0] - prev_pos[0]
        dy = pos_m[1] - prev_pos[1]
        dist_m = math.hypot(dx, dy)

        speed_kmh = (dist_m / delta_t) * 3.6

        # Filtro de ruído físico: sprinters de elite atingem ~36-38 km/h.
        # Deslocamentos > 40 km/h decorrem de trocas de ID ou jitter de detecção.
        if speed_kmh <= self.max_speed_kmh:
            self._total_distance[track_id] += dist_m
            self._recent_speeds[track_id].append(speed_kmh)
        else:
            # Salto irreal descartado
            pass

        self._last_pos[track_id] = pos_m
        self._last_frame_idx[track_id] = frame_idx

        return (self.get_speed(track_id), self._total_distance[track_id])

    def get_speed(self, track_id: int) -> float:
        """Retorna a velocidade média recente suavizada em km/h para o jogador."""
        speeds = self._recent_speeds.get(track_id)
        if not speeds:
            return 0.0
        return float(sum(speeds) / len(speeds))

    def get_distance(self, track_id: int) -> float:
        """Retorna a distância total percorrida em metros pelo jogador."""
        return float(self._total_distance.get(track_id, 0.0))

    def get_all_metrics(self) -> dict[int, dict[str, float]]:
        """Retorna o resumo de métricas para todos os jogadores registrados."""
        result = {}
        for tid in self._last_pos:
            result[tid] = {
                "speed_kmh": round(self.get_speed(tid), 2),
                "total_distance_m": round(self.get_distance(tid), 2),
            }
        return result
