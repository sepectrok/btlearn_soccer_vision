"""Prepara os artefatos externos necessarios para rodar o pipeline em uma maquina limpa.

Itens verificados/baixados:
  1. Video de teste ``data/121364_0.mp4`` (Google Drive, via gdown) - licenca nao verificada, uso so academico.
  2. Modelo de keypoints do campo (Hugging Face ``martinjolif/yolo-football-pitch-detection``).
  3. Pesos do detector (``weights/detector_best.pt`` ou ``runs/detector/weights/best.pt``) - apenas verificados.

Uso:
    python -m scripts.setup_assets
"""
from __future__ import annotations

import sys

from src.utils import get_logger, load_config, resolve, resolve_detector_weights

log = get_logger("setup_assets")

VIDEO_GDRIVE_ID = "1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu"
VIDEO_PATH = "data/121364_0.mp4"


def ensure_video() -> bool:
    """Garante a existencia do video de teste; baixa via gdown se ausente."""
    dest = resolve(VIDEO_PATH)
    if dest.exists() and dest.stat().st_size > 0:
        log.info("[OK] Video presente: %s", dest)
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown

        log.info("Baixando video de teste (Google Drive)...")
        gdown.download(id=VIDEO_GDRIVE_ID, output=str(dest), quiet=False)
    except Exception as exc:  # noqa: BLE001 - rede/Drive podem falhar de varias formas
        log.error("Falha ao baixar o video: %s", exc)
    ok = dest.exists() and dest.stat().st_size > 0
    if not ok:
        log.error("[FALTA] Coloque um video em %s (qualquer clipe de transmissao de futebol serve).", dest)
    return ok


def ensure_pitch_model() -> bool:
    """Baixa (e deixa em cache) o modelo de keypoints do campo."""
    cfg = load_config()["homography"]
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=cfg["model_repo"], filename=cfg["model_filename"])
        log.info("[OK] Modelo de campo em cache: %s", path)
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("[FALTA] Nao foi possivel baixar o modelo de campo (%s). Verifique a internet.", exc)
        return False


def check_detector() -> bool:
    """Verifica os pesos do detector (nao e possivel baixar automaticamente)."""
    try:
        log.info("[OK] Pesos do detector: %s", resolve_detector_weights())
        return True
    except FileNotFoundError as exc:
        log.error("[FALTA] %s", exc)
        return False


def main() -> int:
    results = {"video": ensure_video(), "pitch_model": ensure_pitch_model(), "detector": check_detector()}
    log.info("Resumo: %s", results)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
