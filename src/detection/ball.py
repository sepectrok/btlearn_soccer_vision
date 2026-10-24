"""Logica da bola: fatiamento (slicing), escolha da melhor deteccao, interpolacao e suavizacao.

Funcoes puras (numpy) para serem testaveis sem GPU/modelo.
Trajetoria da bola: array (N, 2) com NaN onde nao ha posicao.
"""
from __future__ import annotations

import numpy as np

from src.utils import get_logger

log = get_logger(__name__)


def slice_boxes(width: int, height: int, slice_wh: tuple[int, int], overlap: float) -> list[tuple[int, int, int, int]]:
    """Gera janelas (x1, y1, x2, y2) cobrindo a imagem com sobreposicao.

    A ultima janela de cada eixo e encostada na borda, entao nunca passa da imagem.
    """
    if not 0 <= overlap < 1:
        raise ValueError("overlap deve estar em [0, 1)")
    sw, sh = min(slice_wh[0], width), min(slice_wh[1], height)

    def starts(total: int, size: int) -> list[int]:
        step = max(1, int(size * (1 - overlap)))
        s = list(range(0, max(total - size, 0) + 1, step))
        if s[-1] + size < total:
            s.append(total - size)
        return s

    return [(x, y, x + sw, y + sh) for y in starts(height, sh) for x in starts(width, sw)]


def select_ball(
    candidates: np.ndarray,
    prev_xy: np.ndarray | None,
    max_jump_px: float,
) -> np.ndarray | None:
    """Escolhe UMA bola entre candidatas (M, 3) = [cx, cy, conf].

    Regra: se ha posicao anterior, descarta candidatas alem de `max_jump_px`
    (fisicamente implausiveis) e pega a de maior confianca; sem anterior, pega a de maior confianca.
    Retorna [cx, cy, conf] ou None.
    """
    if candidates is None or len(candidates) == 0:
        return None
    c = np.asarray(candidates, dtype=float)
    if prev_xy is not None and not np.isnan(prev_xy).any():
        dist = np.hypot(c[:, 0] - prev_xy[0], c[:, 1] - prev_xy[1])
        c = c[dist <= max_jump_px]
        if len(c) == 0:
            return None
    return c[np.argmax(c[:, 2])]


def reject_outliers(traj: np.ndarray, max_jump_px: float, frames_per_gap: int = 1) -> np.ndarray:
    """Remove pontos que saltam demais em relacao ao ultimo ponto valido (escala com o buraco)."""
    out = traj.copy()
    last_i: int | None = None
    for i in range(len(out)):
        if np.isnan(out[i]).any():
            continue
        if last_i is not None:
            gap = (i - last_i) * frames_per_gap
            if np.hypot(*(out[i] - out[last_i])) > max_jump_px * gap:
                out[i] = np.nan
                continue
        last_i = i
    return out


def interpolate_gaps(traj: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Interpolacao linear apenas em buracos de ate `max_gap` frames (entre dois pontos validos).

    Retorna (trajetoria, mascara_interpolado). Buracos maiores permanecem NaN
    (melhor nao inventar posicao da bola).
    """
    out = traj.copy()
    interp = np.zeros(len(traj), dtype=bool)
    valid = np.where(~np.isnan(traj).any(axis=1))[0]
    for a, b in zip(valid[:-1], valid[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            for k in (0, 1):
                out[a + 1 : b, k] = np.linspace(traj[a, k], traj[b, k], gap + 2)[1:-1]
            interp[a + 1 : b] = True
    return out, interp


def smooth_median(traj: np.ndarray, window: int) -> np.ndarray:
    """Mediana movel (robusta a ruido) ignorando NaN; NaN originais continuam NaN."""
    if window <= 1:
        return traj.copy()
    half = window // 2
    out = traj.copy()
    for i in range(len(traj)):
        if np.isnan(traj[i]).any():
            continue
        seg = traj[max(0, i - half) : i + half + 1]
        seg = seg[~np.isnan(seg).any(axis=1)]
        out[i] = np.median(seg, axis=0)
    return out


def postprocess_trajectory(
    traj: np.ndarray, max_jump_px: float, max_gap: int, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pipeline de pos-processamento: outliers -> interpolacao -> suavizacao."""
    t = reject_outliers(traj, max_jump_px)
    t, interp = interpolate_gaps(t, max_gap)
    t = smooth_median(t, window)
    return t, interp


def ball_recall(traj: np.ndarray, gt: np.ndarray, tol_px: float = 20.0) -> float:
    """Fracao de frames com bola GT em que a predicao esta a ate `tol_px` do GT."""
    has_gt = ~np.isnan(gt).any(axis=1)
    if not has_gt.any():
        return float("nan")
    ok = ~np.isnan(traj).any(axis=1) & has_gt
    d = np.full(len(gt), np.inf)
    d[ok] = np.hypot(*(traj[ok] - gt[ok]).T)
    return float((d[has_gt] <= tol_px).mean())


class BallDetector:
    """Detector de bola sobre um modelo Ultralytics YOLO, com ou sem slicing.

    A API usada (`YOLO(...)`, `model.predict(img, conf=, imgsz=, verbose=)`,
    `result.boxes.xyxy/conf/cls`) e a documentada pela Ultralytics; confirme em
    https://docs.ultralytics.com/modes/predict/ se atualizar a versao.
    """

    def __init__(self, weights: str, ball_class_id: int, conf: float = 0.10,
                 imgsz: int = 1280, slicing: bool = True,
                 slice_wh: tuple[int, int] = (640, 640), overlap: float = 0.2) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.ball_id, self.conf, self.imgsz = ball_class_id, conf, imgsz
        self.slicing, self.slice_wh, self.overlap = slicing, slice_wh, overlap

    def _predict(self, img: np.ndarray, dx: int = 0, dy: int = 0) -> list[list[float]]:
        res = self.model.predict(img, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        out = []
        for (x1, y1, x2, y2), c, k in zip(res.boxes.xyxy.cpu().numpy(),
                                           res.boxes.conf.cpu().numpy(),
                                           res.boxes.cls.cpu().numpy()):
            if int(k) == self.ball_id:
                out.append([(x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, float(c)])
        return out

    def candidates(self, frame: np.ndarray) -> np.ndarray:
        """Candidatas [cx, cy, conf] no frame inteiro (+ fatias se slicing ativo)."""
        cands = self._predict(frame)
        if self.slicing:
            h, w = frame.shape[:2]
            for x1, y1, x2, y2 in slice_boxes(w, h, self.slice_wh, self.overlap):
                cands += self._predict(frame[y1:y2, x1:x2], x1, y1)
        return np.array(cands, dtype=float).reshape(-1, 3)
