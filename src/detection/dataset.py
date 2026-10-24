"""Validacao da estrutura de um dataset em formato YOLO (Roboflow)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.utils import get_logger

log = get_logger(__name__)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class DatasetReport:
    names: list[str]
    counts: dict[str, int] = field(default_factory=dict)       # imagens por split
    class_instances: dict[str, int] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _find_split_dir(root: Path, split: str) -> Path | None:
    for cand in (root / split, root / "images" / split):
        if cand.exists():
            return cand
    return None


def validate_yolo_dataset(root: Path, expected_classes: list[str] | None = None) -> DatasetReport:
    """Valida data.yaml, pareamento imagem/label, formato das linhas e classes.

    Aceita os layouts do Roboflow (`train/images`, `train/labels`) e do Ultralytics
    (`images/train`, `labels/train`). Nao levanta excecao para problemas de dados:
    eles vao em `report.problems`. Levanta FileNotFoundError se nao houver data.yaml.
    """
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml nao encontrado em {root}")
    with yaml_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_names = cfg.get("names", [])
    names = [raw_names[k] for k in sorted(raw_names)] if isinstance(raw_names, dict) else list(raw_names)
    rep = DatasetReport(names=names, class_instances={n: 0 for n in names})

    if expected_classes is not None and sorted(names) != sorted(expected_classes):
        rep.problems.append(f"Classes do dataset {names} != esperadas {expected_classes}")

    for split in ("train", "valid", "test"):
        split_dir = _find_split_dir(root, split) or (_find_split_dir(root, "val") if split == "valid" else None)
        if split_dir is None:
            if split != "test":
                rep.problems.append(f"Split ausente: {split}")
            continue
        img_dir = split_dir / "images" if (split_dir / "images").exists() else split_dir
        lbl_dir = (split_dir / "labels") if (split_dir / "labels").exists() else Path(
            str(img_dir).replace("images", "labels")
        )
        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT]
        rep.counts[split] = len(imgs)
        for img in imgs:
            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.exists():
                continue  # imagem sem label = fundo (valido em YOLO)
            for ln, line in enumerate(lbl.read_text().splitlines(), 1):
                parts = line.split()
                if not parts:
                    continue
                try:
                    cls = int(parts[0])
                    coords = [float(x) for x in parts[1:5]]
                except ValueError:
                    rep.problems.append(f"{lbl.name}:{ln} linha invalida")
                    continue
                if len(parts) != 5 or not all(0.0 <= c <= 1.0 for c in coords):
                    rep.problems.append(f"{lbl.name}:{ln} formato/coords fora de [0,1] (nao e bbox YOLO?)")
                    continue
                if not 0 <= cls < len(names):
                    rep.problems.append(f"{lbl.name}:{ln} classe {cls} fora do range")
                    continue
                rep.class_instances[names[cls]] += 1
    return rep
