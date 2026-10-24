"""Módulo de renderização de minimapa 2D (radar tático) e overlay no vídeo.

Desenha um campo 2D estilo transmissão esportiva com proporções FIFA (105x68 m),
plotando jogadores por equipe, goleiros, árbitros, bola e barra de posse de bola.
"""
from __future__ import annotations

import math
import cv2
import numpy as np

from src.homography.pitch import PitchModel


class MinimapRenderer:
    """Renderizador de radar tático 2D e compositor visual sobre frames de vídeo."""

    def __init__(
        self,
        pitch_model: PitchModel | None = None,
        width: int = 350,
        height: int = 227,
        alpha: float = 0.85,
        margin: int = 20,
    ) -> None:
        """Inicializa o renderizador do minimapa.

        Args:
            pitch_model: Modelo com dimensões físicas do campo (padrão 105x68 m).
            width: Largura do minimapa em pixels (padrão 350).
            height: Altura do minimapa em pixels (padrão 227).
            alpha: Transparência do minimapa sobreposto no vídeo (0.0 a 1.0).
            margin: Distância em pixels da borda do vídeo.
        """
        self.pitch = pitch_model or PitchModel()
        self.w = int(width)
        self.h = int(height)
        self.alpha = float(alpha)
        self.margin = int(margin)

        # Margens internas dentro do minimapa para as linhas do campo
        self.pad_x = 16
        self.pad_y = 16
        self.pitch_w = self.w - 2 * self.pad_x
        self.pitch_h = self.h - 2 * self.pad_y

        # Escala: metros -> pixels
        self.scale_x = self.pitch_w / self.pitch.length
        self.scale_y = self.pitch_h / self.pitch.width

        # Cores BGR padrão
        self.color_bg = (24, 32, 28)           # Verde escuro / asfalto suave
        self.color_border = (80, 100, 90)      # Borda externa suave
        self.color_lines = (220, 230, 225)     # Linhas brancas do campo
        self.color_team0 = (50, 50, 230)       # Vermelho vibrante
        self.color_team1 = (50, 220, 70)       # Verde neon vibrante
        self.color_gk = (0, 215, 255)          # Amarelo/Ouro
        self.color_ref = (240, 240, 240)       # Branco/Prata com borda preta
        self.color_ball = (255, 255, 255)      # Branco puro

        # Pré-renderiza a base estática do campo
        self._base_pitch = self._create_base_pitch()

    def _to_radar_px(self, x_m: float, y_m: float) -> tuple[int, int]:
        """Converte coordenadas métricas (X, Y) do campo para pixels do radar."""
        rx = int(self.pad_x + x_m * self.scale_x)
        ry = int(self.pad_y + y_m * self.scale_y)
        return (rx, ry)

    def _create_base_pitch(self) -> np.ndarray:
        """Desenha a imagem estática do gramado e todas as marcações oficiais do campo."""
        img = np.full((self.h, self.w, 3), self.color_bg, dtype=np.uint8)

        # Borda externa arredondada / moldura do radar
        cv2.rectangle(img, (0, 0), (self.w - 1, self.h - 1), self.color_border, 1)

        # 1. Quatro linhas do campo
        x0, y0 = self._to_radar_px(0.0, 0.0)
        x1, y1 = self._to_radar_px(self.pitch.length, self.pitch.width)
        cv2.rectangle(img, (x0, y0), (x1, y1), self.color_lines, 1, cv2.LINE_AA)

        # 2. Linha de meio de campo
        xm, _ = self._to_radar_px(self.pitch.length / 2.0, 0.0)
        cv2.line(img, (xm, y0), (xm, y1), self.color_lines, 1, cv2.LINE_AA)

        # 3. Círculo central e marca central
        cx, cy = self._to_radar_px(self.pitch.length / 2.0, self.pitch.width / 2.0)
        radius_px = int(self.pitch.centre_circle_radius * self.scale_x)
        cv2.circle(img, (cx, cy), radius_px, self.color_lines, 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 2, self.color_lines, -1)

        # 4. Grande área esquerda e direita
        W = self.pitch.width
        pbw = self.pitch.penalty_box_width
        pbl = self.pitch.penalty_box_length
        gbl = self.pitch.goal_box_length
        gbw = self.pitch.goal_box_width
        ps = self.pitch.penalty_spot_distance

        # Grande área esq
        p1 = self._to_radar_px(0.0, (W - pbw) / 2.0)
        p2 = self._to_radar_px(pbl, (W + pbw) / 2.0)
        cv2.rectangle(img, p1, p2, self.color_lines, 1, cv2.LINE_AA)

        # Pequena área esq
        g1 = self._to_radar_px(0.0, (W - gbw) / 2.0)
        g2 = self._to_radar_px(gbl, (W + gbw) / 2.0)
        cv2.rectangle(img, g1, g2, self.color_lines, 1, cv2.LINE_AA)

        # Ponto penal esquerdo
        ps_l = self._to_radar_px(ps, W / 2.0)
        cv2.circle(img, ps_l, 2, self.color_lines, -1)

        # Grande área dir
        L = self.pitch.length
        p1_r = self._to_radar_px(L - pbl, (W - pbw) / 2.0)
        p2_r = self._to_radar_px(L, (W + pbw) / 2.0)
        cv2.rectangle(img, p1_r, p2_r, self.color_lines, 1, cv2.LINE_AA)

        # Pequena área dir
        g1_r = self._to_radar_px(L - gbl, (W - gbw) / 2.0)
        g2_r = self._to_radar_px(L, (W + gbw) / 2.0)
        cv2.rectangle(img, g1_r, g2_r, self.color_lines, 1, cv2.LINE_AA)

        # Ponto penal direito
        ps_r = self._to_radar_px(L - ps, W / 2.0)
        cv2.circle(img, ps_r, 2, self.color_lines, -1)

        return img

    def render(
        self,
        players: list[dict],
        ball_pos_m: tuple[float, float] | None = None,
        possession_info: dict | None = None,
    ) -> np.ndarray:
        """Gera a imagem do minimapa com as posições dinâmicas e estatísticas do jogo.

        Args:
            players: Lista de dicionários dos jogadores:
                     [{'track_id': int, 'team_id': int | None, 'pos_m': (x, y),
                       'cls_name': str, 'is_referee': bool, 'is_goalkeeper': bool}, ...]
            ball_pos_m: Coordenadas métricas da bola (X, Y) ou None.
            possession_info: Dicionário retornado pelo BallPossessionTracker.

        Returns:
            Imagem BGR uint8 de dimensões (h, w, 3).
        """
        radar = self._base_pitch.copy()

        # 1. Renderiza jogadores
        for p in players:
            pos = p.get("pos_m")
            if pos is None or math.isnan(pos[0]) or math.isnan(pos[1]):
                continue

            rx, ry = self._to_radar_px(pos[0], pos[1])
            # Garante que o ponto está dentro do radar
            if not (0 <= rx < self.w and 0 <= ry < self.h):
                continue

            cls_name = p.get("cls_name", "player")
            team_id = p.get("team_id")

            if cls_name == "referee":
                color = self.color_ref
                border_color = (0, 0, 0)
                radius = 4
            elif cls_name == "goalkeeper":
                color = self.color_gk
                border_color = (0, 0, 0)
                radius = 5
            elif team_id == 0:
                color = self.color_team0
                border_color = (255, 255, 255)
                radius = 5
            elif team_id == 1:
                color = self.color_team1
                border_color = (0, 0, 0)
                radius = 5
            else:
                color = (180, 180, 180)
                border_color = (0, 0, 0)
                radius = 4

            cv2.circle(radar, (rx, ry), radius, color, -1, cv2.LINE_AA)
            cv2.circle(radar, (rx, ry), radius, border_color, 1, cv2.LINE_AA)

        # 2. Renderiza a bola
        if ball_pos_m is not None and not (math.isnan(ball_pos_m[0]) or math.isnan(ball_pos_m[1])):
            bx, by = self._to_radar_px(ball_pos_m[0], ball_pos_m[1])
            if 0 <= bx < self.w and 0 <= by < self.h:
                cv2.circle(radar, (bx, by), 4, (0, 140, 255), -1, cv2.LINE_AA)
                cv2.circle(radar, (bx, by), 3, self.color_ball, -1, cv2.LINE_AA)
                cv2.circle(radar, (bx, by), 4, (0, 0, 0), 1, cv2.LINE_AA)

        # 3. Renderiza barra de posse de bola na parte inferior
        if possession_info is not None:
            t0_pct = float(possession_info.get("team_0_pct", 50.0))
            t1_pct = float(possession_info.get("team_1_pct", 50.0))

            bar_y = self.h - 10
            bar_h = 6
            bar_w = self.w - 2 * self.pad_x

            # Largura proporcional do Time 0
            w0 = int((t0_pct / 100.0) * bar_w)
            w1 = bar_w - w0

            # Desenha barra Time 0 (Vermelho) e Time 1 (Verde)
            cv2.rectangle(
                radar,
                (self.pad_x, bar_y),
                (self.pad_x + w0, bar_y + bar_h),
                self.color_team0,
                -1,
            )
            cv2.rectangle(
                radar,
                (self.pad_x + w0, bar_y),
                (self.pad_x + bar_w, bar_y + bar_h),
                self.color_team1,
                -1,
            )

            # Legenda de texto compacta (Ex: 54% | 46%)
            label = f"{t0_pct:.0f}%  POSSE  {t1_pct:.0f}%"
            cv2.putText(
                radar,
                label,
                (self.w // 2 - 38, bar_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.28,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        return radar

    def overlay_on_frame(
        self,
        frame: np.ndarray,
        minimap: np.ndarray,
        position: str = "top_right",
    ) -> np.ndarray:
        """Sobrepõe o minimapa no frame do vídeo com transparência alfa.

        Args:
            frame: Imagem original do frame do vídeo (BGR uint8).
            minimap: Imagem renderizada do minimapa (BGR uint8).
            position: Posição no frame ("top_right", "top_left", "bottom_right", "bottom_left").

        Returns:
            Frame anotado com o minimapa incorporado.
        """
        fh, fw = frame.shape[:2]
        mh, mw = minimap.shape[:2]

        if position == "top_right":
            x1 = fw - mw - self.margin
            y1 = self.margin
        elif position == "top_left":
            x1 = self.margin
            y1 = self.margin
        elif position == "bottom_left":
            x1 = self.margin
            y1 = fh - mh - self.margin
        else:  # bottom_right
            x1 = fw - mw - self.margin
            y1 = fh - mh - self.margin

        x2, y2 = x1 + mw, y1 + mh

        # Região de interesse (ROI)
        roi = frame[y1:y2, x1:x2]
        # Mistura alfa: out = alpha * minimap + (1 - alpha) * roi
        blended = cv2.addWeighted(minimap, self.alpha, roi, 1.0 - self.alpha, 0)
        frame[y1:y2, x1:x2] = blended

        # Contorno sutil ao redor do minimapa sobreposto
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1, cv2.LINE_AA)

        return frame
