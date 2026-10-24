"""Testes do resolvedor de pesos do detector e da validacao de saidas."""
from __future__ import annotations

import pytest

from src import utils


def test_resolve_detector_weights_explicit(tmp_path):
    w = tmp_path / "meu.pt"
    w.write_bytes(b"x")
    assert utils.resolve_detector_weights(w) == w


def test_resolve_detector_weights_fallback(tmp_path, monkeypatch):
    (tmp_path / "weights").mkdir()
    w = tmp_path / "weights" / "detector_best.pt"
    w.write_bytes(b"x")
    monkeypatch.setattr(utils, "PROJECT_ROOT", tmp_path)
    assert utils.resolve_detector_weights() == w


def test_resolve_detector_weights_missing_has_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "PROJECT_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="train_detector"):
        utils.resolve_detector_weights()
