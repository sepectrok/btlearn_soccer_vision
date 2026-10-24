"""Treina o detector. Uso: python -m scripts.train_detector [--epochs N]"""
from __future__ import annotations

import argparse

from src.detection.train_eval import train_detector
from src.utils import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--epochs", type=int, help="sobrescreve o YAML (use 1-2 para smoke test)")
    ap.add_argument("--name", default="detector")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.epochs:
        cfg["detection"]["train"]["epochs"] = a.epochs
    train_detector(cfg, name=a.name)


if __name__ == "__main__":
    main()
