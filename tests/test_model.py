import pytest

from src import config
from src.model import best_weights_path, build_model, load_best_model


def test_build_model_from_architecture_yaml():
    # A YAML builds an untrained network without downloading pretrained weights.
    assert build_model("classify", "yolo11n-cls.yaml").task == "classify"
    assert build_model("segment", "yolo11n-seg.yaml").task == "segment"


def test_build_model_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        build_model("classify", "yolo11n-seg.yaml")


def test_build_model_rejects_unknown_task():
    with pytest.raises(ValueError):
        build_model("detect")


def test_best_weights_live_in_models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    assert best_weights_path("classify") == tmp_path / "classify_best.pt"
    with pytest.raises(FileNotFoundError, match="src.train"):
        load_best_model("classify")
