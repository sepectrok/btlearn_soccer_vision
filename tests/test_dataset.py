from pathlib import Path

import pytest

from src.detection.dataset import validate_yolo_dataset


def _make(root: Path, label_line: str, names="['ball','goalkeeper','player','referee']"):
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    (root / "valid" / "images").mkdir(parents=True)
    (root / "valid" / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text(f"names: {names}\n")
    for split in ("train", "valid"):
        (root / split / "images" / "a.jpg").write_bytes(b"x")
        (root / split / "labels" / "a.txt").write_text(label_line)


def test_valid_dataset(tmp_path):
    _make(tmp_path, "2 0.5 0.5 0.1 0.2\n")
    rep = validate_yolo_dataset(tmp_path, ["ball", "goalkeeper", "player", "referee"])
    assert rep.ok and rep.class_instances["player"] == 2


def test_bad_class_and_polygon(tmp_path):
    _make(tmp_path, "9 0.5 0.5 0.1 0.2\n2 0.1 0.1 0.2 0.2 0.3 0.3\n")
    rep = validate_yolo_dataset(tmp_path)
    assert not rep.ok and any("fora do range" in p for p in rep.problems)


def test_wrong_classes_and_missing_yaml(tmp_path):
    _make(tmp_path, "0 0.5 0.5 0.1 0.2\n", names="['a','b']")
    assert not validate_yolo_dataset(tmp_path, ["ball", "player"]).ok
    with pytest.raises(FileNotFoundError):
        validate_yolo_dataset(tmp_path / "nope")
