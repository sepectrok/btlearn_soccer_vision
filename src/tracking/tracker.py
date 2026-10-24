"""Rastreamento de jogadores, goleiros e arbitros usando ByteTrack (Ultralytics).

A bola e mantida fora deste tracker e tratada separadamente pelo BallDetector,
conforme definido na arquitetura do projeto.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.utils import get_logger, resolve

log = get_logger(__name__)

# Cores BGR padrão por classe
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "player": (255, 180, 0),       # Ciano/Azul claro (BGR)
    "goalkeeper": (0, 140, 255),   # Laranja
    "referee": (0, 255, 255),      # Amarelo
    "ball": (0, 0, 255),           # Vermelho
}


@dataclass
class TrackedPlayer:
    """Representa um jogador, goleiro ou arbitro rastreado em um frame."""

    track_id: int
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) em pixels
    cls_id: int
    cls_name: str
    conf: float

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Ponto de contato com o chao (pe do jogador).

        Essencial para homografia e projecao 2D no campo.
        """
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    @property
    def center(self) -> tuple[float, float]:
        """Centro geometrico do bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> float:
        """Largura do bounding box em pixels."""
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        """Altura do bounding box em pixels."""
        return max(0.0, self.bbox[3] - self.bbox[1])

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Recorta o bounding box do jogador na imagem com recorte seguro."""
        h, w = frame.shape[:2]
        x1 = int(np.clip(self.bbox[0], 0, w - 1))
        y1 = int(np.clip(self.bbox[1], 0, h - 1))
        x2 = int(np.clip(self.bbox[2], 0, w))
        y2 = int(np.clip(self.bbox[3], 0, h))
        return frame[y1:y2, x1:x2].copy()

    def torso_crop(self, frame: np.ndarray) -> np.ndarray:
        """Recorta a regiao do tronco (15% a 55% da altura).

        Utilizado para extracao de cor da camisa / embeddings na Etapa 3.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.bbox
        bh = y2 - y1
        ty1 = int(np.clip(y1 + 0.15 * bh, 0, h - 1))
        ty2 = int(np.clip(y1 + 0.55 * bh, 0, h))
        tx1 = int(np.clip(x1, 0, w - 1))
        tx2 = int(np.clip(x2, 0, w))
        return frame[ty1:ty2, tx1:tx2].copy()


class PlayerTracker:
    """Gerencia a inferencia de deteccao e rastreamento com ByteTrack."""

    def __init__(
        self,
        weights: str | Path,
        all_classes: list[str],
        track_classes: list[str] | None = None,
        tracker_type: str = "bytetrack.yaml",
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 1280,
    ) -> None:
        from ultralytics import YOLO

        self.weights_path = resolve(weights)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Pesos nao encontrados: {self.weights_path}")

        self.all_classes = all_classes
        self.track_classes = track_classes or ["goalkeeper", "player", "referee"]
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.tracker_type = tracker_type

        # Indices das classes permitidas para rastreamento (garante que ball fica de fora)
        self.target_class_ids = [
            self.all_classes.index(c) for c in self.track_classes if c in self.all_classes
        ]

        self.model = YOLO(str(self.weights_path))
        # Historico de trajetorias dos ultimos frames: track_id -> list[(x, y)]
        self.history: dict[int, list[tuple[float, float]]] = {}

    def track_frame(self, frame: np.ndarray, persist: bool = True) -> list[TrackedPlayer]:
        """Processa um frame BGR e retorna a lista de jogadores com track_id."""
        if frame is None or frame.size == 0:
            return []

        results = self.model.track(
            source=frame,
            tracker=self.tracker_type,
            persist=persist,
            classes=self.target_class_ids,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )

        tracked_players: list[TrackedPlayer] = []
        if not results or len(results) == 0:
            return tracked_players

        res = results[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            return tracked_players

        xyxy_arr = boxes.xyxy.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()
        cls_arr = boxes.cls.cpu().numpy().astype(int)
        ids_arr = boxes.id.cpu().numpy().astype(int)

        for i in range(len(boxes)):
            tid = int(ids_arr[i])
            cls_id = int(cls_arr[i])
            cls_name = self.all_classes[cls_id] if cls_id < len(self.all_classes) else str(cls_id)
            bbox = (float(xyxy_arr[i][0]), float(xyxy_arr[i][1]), float(xyxy_arr[i][2]), float(xyxy_arr[i][3]))
            conf_val = float(conf_arr[i])

            player = TrackedPlayer(
                track_id=tid,
                bbox=bbox,
                cls_id=cls_id,
                cls_name=cls_name,
                conf=conf_val,
            )
            tracked_players.append(player)

            # Atualiza historico de rastreamento do pe
            if tid not in self.history:
                self.history[tid] = []
            self.history[tid].append(player.bottom_center)
            if len(self.history[tid]) > 30:
                self.history[tid].pop(0)

        return tracked_players

    def reset(self) -> None:
        """Limpa o historico de trajetorias e reinicializa o tracker."""
        self.history.clear()
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            if hasattr(self.model.predictor, "trackers"):
                for t in self.model.predictor.trackers:
                    if hasattr(t, "reset"):
                        t.reset()


def draw_player_tracks(
    frame: np.ndarray,
    players: list[TrackedPlayer],
    history: dict[int, list[tuple[float, float]]] | None = None,
    draw_trails: bool = True,
    draw_feet: bool = True,
) -> np.ndarray:
    """Desenha bounding boxes, IDs e rastros nos jogadores no frame."""
    out = frame.copy()

    for p in players:
        color = CLASS_COLORS.get(p.cls_name, (200, 200, 200))
        x1, y1, x2, y2 = [int(v) for v in p.bbox]

        # Caixa delimitadora
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label com fundo destacado: #ID classe
        label = f"#{p.track_id} {p.cls_name[:2].upper()}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(0, y1)), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 2, max(th, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        # Ponto no pe do jogador (contato com o chao)
        if draw_feet:
            bx, by = int(p.bottom_center[0]), int(p.bottom_center[1])
            cv2.circle(out, (bx, by), 4, color, -1)
            cv2.circle(out, (bx, by), 5, (0, 0, 0), 1)

        # Rastro da trajetoria recente
        if draw_trails and history and p.track_id in history:
            pts = history[p.track_id]
            if len(pts) >= 2:
                for j in range(1, len(pts)):
                    pt1 = (int(pts[j - 1][0]), int(pts[j - 1][1]))
                    pt2 = (int(pts[j][0]), int(pts[j][1]))
                    alpha = j / len(pts)
                    thickness = max(1, int(2 * alpha))
                    cv2.line(out, pt1, pt2, color, thickness)

    return out


def draw_ball(
    frame: np.ndarray,
    ball_coord: tuple[float, float] | None,
    radius: int = 6,
) -> np.ndarray:
    """Desenha a bola com circulo de destaque."""
    if ball_coord is None or np.isnan(ball_coord[0]) or np.isnan(ball_coord[1]):
        return frame
    out = frame.copy()
    bx, by = int(ball_coord[0]), int(ball_coord[1])
    color = CLASS_COLORS["ball"]
    cv2.circle(out, (bx, by), radius + 2, (255, 255, 255), 2)
    cv2.circle(out, (bx, by), radius, color, -1)
    cv2.putText(
        out,
        "BALL",
        (bx - 12, by - radius - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out
