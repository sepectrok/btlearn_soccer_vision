"""Baixa (Roboflow) ou valida um dataset YOLO.

Uso:
  # validar um dataset ja extraido em data/roboflow_players
  python -m scripts.prepare_dataset --validate-only

  # baixar via API (precisa de ROBOFLOW_API_KEY no ambiente)
  python -m scripts.prepare_dataset --workspace <ws> --project <proj> --version <n>

Workspace/projeto/versao aparecem na URL da pagina do dataset no Roboflow Universe
(universe.roboflow.com/<workspace>/<project>/dataset/<version>).
A chamada `Roboflow(api_key).workspace().project().version().download(fmt, location)` segue
a doc do pacote `roboflow`; confirme em https://docs.roboflow.com/ se der erro de assinatura.
"""
from __future__ import annotations

import argparse
import os
import sys

from src.detection.dataset import validate_yolo_dataset
from src.utils import get_logger, load_config, resolve

log = get_logger("prepare_dataset")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--workspace")
    ap.add_argument("--project")
    ap.add_argument("--version", type=int)
    ap.add_argument("--validate-only", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    dest = resolve(cfg["paths"]["dataset_dir"])

    if not a.validate_only:
        key = os.environ.get("ROBOFLOW_API_KEY")
        if not (key and a.workspace and a.project and a.version):
            log.error("Informe ROBOFLOW_API_KEY, --workspace, --project e --version (ou use --validate-only).")
            return 2
        import io
        import zipfile

        import requests

        dest.mkdir(parents=True, exist_ok=True)
        url = f"https://api.roboflow.com/{a.workspace}/{a.project}/{a.version}/yolov8"
        meta = requests.get(url, params={"api_key": key}, timeout=60)
        meta.raise_for_status()
        link = (meta.json().get("export") or {}).get("link")
        if not link:
            log.error("A API nao retornou link de export: %s", meta.text[:300])
            return 2
        log.info("Baixando zip do dataset...")
        z = requests.get(link, timeout=600)
        z.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
            zf.extractall(dest)

    rep = validate_yolo_dataset(dest, cfg["detection"]["classes"])
    log.info("Classes: %s", rep.names)
    log.info("Imagens por split: %s", rep.counts)
    log.info("Instancias por classe: %s", rep.class_instances)
    for p in rep.problems[:20]:
        log.warning(p)
    log.info("Dataset %s", "OK" if rep.ok else f"COM {len(rep.problems)} PROBLEMAS")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
