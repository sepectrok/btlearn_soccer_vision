"""Treino e avaliacao do detector (wrappers finos sobre a API do Ultralytics)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils import get_logger, resolve, set_seed

log = get_logger(__name__)


def train_detector(cfg: dict[str, Any], data_yaml: Path | None = None, name: str = "detector") -> Path:
    """Fine-tuning a partir de pesos pre-treinados. Retorna o caminho do best.pt.

    Parametros mapeiam 1:1 para `YOLO.train(...)`; veja
    https://docs.ultralytics.com/usage/cfg/ para a lista oficial.
    """
    from ultralytics import YOLO

    seed = cfg["project"]["seed"]
    set_seed(seed)
    data_yaml = data_yaml or resolve(cfg["paths"]["dataset_dir"]) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} nao existe. Rode scripts/prepare_dataset.py primeiro.")
    t = cfg["detection"]["train"]
    model = YOLO(cfg["detection"]["base_weights"])
    model.train(
        data=str(data_yaml), epochs=t["epochs"], imgsz=t["imgsz"], batch=t["batch"],
        patience=t["patience"], lr0=t["lr0"], optimizer=t["optimizer"], workers=t["workers"],
        seed=seed, deterministic=True,
        project=str(resolve(cfg["paths"]["runs_dir"])), name=name, exist_ok=True,
    )
    best = resolve(cfg["paths"]["runs_dir"]) / name / "weights" / "best.pt"
    log.info("Treino concluido: %s", best)
    return best


def evaluate_detector(weights: Path, data_yaml: Path, split: str = "test", imgsz: int = 960) -> dict[str, Any]:
    """mAP50, mAP50-95 globais e P/R/AP50 por classe via `model.val`."""
    from ultralytics import YOLO

    m = YOLO(str(weights)).val(data=str(data_yaml), split=split, imgsz=imgsz, verbose=False)
    out: dict[str, Any] = {"map50": float(m.box.map50), "map50_95": float(m.box.map), "per_class": {}}
    for i, cls_id in enumerate(m.box.ap_class_index):
        out["per_class"][m.names[int(cls_id)]] = {
            "precision": float(m.box.p[i]), "recall": float(m.box.r[i]),
            "ap50": float(m.box.ap50[i]), "ap50_95": float(m.box.ap[i]),
        }
    return out
