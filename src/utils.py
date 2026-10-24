"""Utilitarios compartilhados: config, seed, logging, leitura de video."""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_logger(name: str) -> logging.Logger:
    """Logger com formato padrao (configura o root uma unica vez)."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    return logging.getLogger(name)


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Le o YAML. Caminhos relativos sao resolvidos a partir da raiz do projeto."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Config nao encontrada: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Resolve um caminho relativo em relacao a raiz do projeto."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


# Locais candidatos para os pesos do detector (ordem de prioridade apos o caminho explicito).
DETECTOR_WEIGHT_CANDIDATES = (
    "weights/detector_best.pt",
    "runs/detector/weights/best.pt",
)


def resolve_detector_weights(path: str | Path | None = None) -> Path:
    """Encontra os pesos do detector YOLO treinado.

    Tenta ``path`` (se dado) e depois os locais em ``DETECTOR_WEIGHT_CANDIDATES``.

    Raises:
        FileNotFoundError: com instrucoes de como obter/gerar os pesos.
    """
    tried: list[Path] = []
    candidates = ([path] if path else []) + list(DETECTOR_WEIGHT_CANDIDATES)
    for c in candidates:
        p = resolve(c)
        tried.append(p)
        if p.exists():
            return p
    raise FileNotFoundError(
        "Pesos do detector nao encontrados. Locais testados:\n  "
        + "\n  ".join(str(t) for t in tried)
        + "\nColoque o arquivo em weights/detector_best.pt ou treine com "
        "`python -m scripts.train_detector --name detector` (veja README, secao 'Avaliacao')."
    )


def set_seed(seed: int) -> None:
    """Fixa seeds de random/numpy/torch (torch e opcional na importacao)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def iter_frames(video_path: str | Path, stride: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Gerador de frames BGR (nao carrega o video na memoria).

    Raises:
        IOError: se o video nao puder ser aberto.
    """
    import cv2

    p = resolve(video_path)
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise IOError(f"Nao foi possivel abrir o video: {p}")
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()
