"""Modelo geométrico oficial do campo de futebol (105m x 68m, padrão FIFA)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class PitchModel:
    """Dimensões físicas e geometria dos 32 marcos padrão de um campo de futebol."""

    length: float = 105.0             # metros [X: 0 a 105]
    width: float = 68.0               # metros [Y: 0 a 68]
    penalty_box_length: float = 16.5  # metros
    penalty_box_width: float = 40.32  # metros
    goal_box_length: float = 5.5      # metros
    goal_box_width: float = 18.32     # metros
    centre_circle_radius: float = 9.15# metros
    penalty_spot_distance: float = 11.0# metros

    @property
    def vertices(self) -> np.ndarray:
        """Coordenadas físicas (X, Y) em metros dos 32 marcos padrão."""
        W, L = self.width, self.length
        pbl, pbw = self.penalty_box_length, self.penalty_box_width
        gbl, gbw = self.goal_box_length, self.goal_box_width
        r = self.centre_circle_radius
        ps = self.penalty_spot_distance

        v = [
            (0.0, 0.0),                           # 0: Label 01 - Canto superior esquerdo
            (0.0, (W - pbw) / 2.0),               # 1: Label 02 - Linha de fundo / grande area esq (topo)
            (0.0, (W - gbw) / 2.0),               # 2: Label 03 - Linha de fundo / pequena area esq (topo)
            (0.0, (W + gbw) / 2.0),               # 3: Label 04 - Linha de fundo / pequena area esq (base)
            (0.0, (W + pbw) / 2.0),               # 4: Label 05 - Linha de fundo / grande area esq (base)
            (0.0, W),                             # 5: Label 06 - Canto inferior esquerdo
            (gbl, (W - gbw) / 2.0),               # 6: Label 07 - Vertice frontal pequena area esq (topo)
            (gbl, (W + gbw) / 2.0),               # 7: Label 08 - Vertice frontal pequena area esq (base)
            (ps, W / 2.0),                        # 8: Label 09 - Marca do penal esquerdo
            (pbl, (W - pbw) / 2.0),               # 9: Label 10 - Vertice frontal grande area esq (topo)
            (pbl, (W - gbw) / 2.0),               # 10: Label 11 - Intersecao frontal grande area / pequena area (topo)
            (pbl, (W + gbw) / 2.0),               # 11: Label 12 - Intersecao frontal grande area / pequena area (base)
            (pbl, (W + pbw) / 2.0),               # 12: Label 13 - Vertice frontal grande area esq (base)
            (L / 2.0, 0.0),                       # 13: Label 15 - Meio de campo / lateral superior
            (L / 2.0, W / 2.0 - r),               # 14: Label 16 - Meio de campo / circulo central (topo)
            (L / 2.0, W / 2.0 + r),               # 15: Label 17 - Meio de campo / circulo central (base)
            (L / 2.0, W),                         # 16: Label 18 - Meio de campo / lateral inferior
            (L - pbl, (W - pbw) / 2.0),           # 17: Label 20 - Vertice frontal grande area dir (topo)
            (L - pbl, (W - gbw) / 2.0),           # 18: Label 21 - Intersecao frontal grande area / pequena area dir (topo)
            (L - pbl, (W + gbw) / 2.0),           # 19: Label 22 - Intersecao frontal grande area / pequena area dir (base)
            (L - pbl, (W + pbw) / 2.0),           # 20: Label 23 - Vertice frontal grande area dir (base)
            (L - ps, W / 2.0),                    # 21: Label 24 - Marca do penal direito
            (L - gbl, (W - gbw) / 2.0),           # 22: Label 25 - Vertice frontal pequena area dir (topo)
            (L - gbl, (W + gbw) / 2.0),           # 23: Label 26 - Vertice frontal pequena area dir (base)
            (L, 0.0),                             # 24: Label 27 - Canto superior direito
            (L, (W - pbw) / 2.0),                 # 25: Label 28 - Linha de fundo / grande area dir (topo)
            (L, (W - gbw) / 2.0),                 # 26: Label 29 - Linha de fundo / pequena area dir (topo)
            (L, (W + gbw) / 2.0),                 # 27: Label 30 - Linha de fundo / pequena area dir (base)
            (L, (W + pbw) / 2.0),                 # 28: Label 31 - Linha de fundo / grande area dir (base)
            (L, W),                               # 29: Label 32 - Canto inferior direito
            (L / 2.0 - r, W / 2.0),               # 30: Label 14 - Circulo central (extremo esquerdo)
            (L / 2.0 + r, W / 2.0),               # 31: Label 19 - Circulo central (extremo direito)
        ]
        return np.array(v, dtype=np.float32)

    @property
    def edges(self) -> list[tuple[int, int]]:
        """Pares de indices (0 a 31) conectando linhas reais no campo."""
        return [
            # Linha de fundo esquerda e laterais
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
            (0, 13), (13, 24),                    # Lateral superior
            (5, 16), (16, 29),                    # Lateral inferior
            # Linha de fundo direita
            (24, 25), (25, 26), (26, 27), (27, 28), (28, 29),
            # Linha de meio de campo
            (13, 14), (14, 15), (15, 16),
            # Grande area esquerda
            (1, 9), (9, 10), (10, 11), (11, 12), (12, 4),
            # Pequena area esquerda
            (2, 6), (6, 7), (7, 3),
            # Grande area direita
            (25, 17), (17, 18), (18, 19), (19, 20), (20, 28),
            # Pequena area direita
            (26, 22), (22, 23), (23, 27),
        ]

    def is_inside(self, x: float, y: float, margin: float = 2.0) -> bool:
        """Verifica se um ponto métrico (X, Y) está dentro das quatro linhas com margem opcional."""
        if np.isnan(x) or np.isnan(y):
            return False
        return (-margin <= x <= self.length + margin) and (-margin <= y <= self.width + margin)

    def clip_point(self, x: float, y: float, margin: float = 0.0) -> tuple[float, float]:
        """Restringe o ponto aos limites físicos do gramado."""
        cx = float(np.clip(x, -margin, self.length + margin))
        cy = float(np.clip(y, -margin, self.width + margin))
        return (cx, cy)
