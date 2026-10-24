"""Extração de características visuais do torso: HSV baseline vs SigLIP Embeddings."""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from src.utils import get_logger

log = get_logger(__name__)


def extract_hsv_features(
    crops: Sequence[np.ndarray],
    bins: tuple[int, int] = (16, 8),
    filter_grass: bool = True,
) -> np.ndarray:
    """Extrai histogramas 2D de Matiz (H) e Saturacao (S) a partir dos recortes de torso.

    Esta e a abordagem baseline (leves, sem deep learning) para a Ablacao 2.

    Args:
        crops: Lista de imagens BGR dos torsos dos jogadores.
        bins: Quantidade de bins para (Hue, Saturation).
        filter_grass: Se True, mascara pixels verdes tipicos do gramado de fundo.

    Returns:
        Matriz numpy de shape (len(crops), bins[0] * bins[1]) com vetores normalizados L2.
    """
    dim = bins[0] * bins[1]
    if not crops:
        return np.empty((0, dim), dtype=np.float32)

    features = []
    for crop in crops:
        if crop is None or crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            features.append(np.zeros(dim, dtype=np.float32))
            continue

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = None

        if filter_grass:
            # Verde no OpenCV HSV: H tipicamente entre 35 e 85, com saturacao minima
            is_green = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] >= 40)
            mask_non_green = (~is_green).astype(np.uint8) * 255
            # So utiliza a mascara se sobrarem pelo menos 10% dos pixels para nao zerar tudo
            if mask_non_green.mean() > 25:
                mask = mask_non_green

        hist = cv2.calcHist(
            [hsv],
            channels=[0, 1],
            mask=mask,
            histSize=[bins[0], bins[1]],
            ranges=[0, 180, 0, 256],
        ).flatten()

        norm = np.linalg.norm(hist)
        if norm > 1e-6:
            hist = hist / norm
        features.append(hist.astype(np.float32))

    return np.vstack(features)


class SigLIPEmbedder:
    """Extrai embeddings densos de visao utilizando o backbone SigLIP da Hugging Face."""

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoImageProcessor, SiglipVisionModel

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        log.info("Carregando modelo SigLIP (%s) em %s...", model_name, self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SiglipVisionModel.from_pretrained(model_name).to(self.device).eval()

    def extract_features(
        self,
        crops: Sequence[np.ndarray],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Processa os recortes de torso e retorna embeddings normalizados L2 (768-D)."""
        import torch

        if not crops:
            return np.empty((0, 768), dtype=np.float32)

        valid_indices = []
        valid_rgbs = []
        fallback_vec = np.zeros(768, dtype=np.float32)

        for i, crop in enumerate(crops):
            if crop is not None and crop.size > 0 and crop.shape[0] >= 4 and crop.shape[1] >= 4:
                valid_indices.append(i)
                valid_rgbs.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        all_embs = np.zeros((len(crops), 768), dtype=np.float32)
        if not valid_rgbs:
            return all_embs

        embs_list = []
        for start in range(0, len(valid_rgbs), batch_size):
            batch = valid_rgbs[start : start + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)

            with torch.no_grad():
                out = self.model(**inputs)
                pool = out.pooler_output
                pool = torch.nn.functional.normalize(pool, p=2, dim=-1)
                embs_list.append(pool.cpu().numpy().astype(np.float32))

        batch_embs = np.vstack(embs_list)
        for idx, orig_i in enumerate(valid_indices):
            all_embs[orig_i] = batch_embs[idx]

        return all_embs
