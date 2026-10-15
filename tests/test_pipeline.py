"""End-to-end smoke test: train for one epoch on a tiny dataset, then evaluate and predict."""

import pytest

from src import config
from src.evaluate import evaluate
from src.predict import predict
from src.train import train


@pytest.mark.slow
def test_train_evaluate_predict(processed_dataset, raw_dataset):
    weights = train(
        "classify",
        weights="yolo11n-cls.yaml",
        device="cpu",
        epochs=1,
        imgsz=32,
        batch=8,
        workers=0,
        plots=False,
    )
    assert weights == config.MODELS_DIR / "classify_best.pt"
    assert weights.is_file()

    results = evaluate("classify", imgsz=32, device="cpu", plots=False)
    assert 0.0 <= results["metrics/accuracy_top1"] <= 1.0

    train_dir, _ = raw_dataset
    predictions = predict(train_dir / "glioma")
    assert len(predictions) == 10
    assert all(p["label"] in config.CLASS_NAMES for p in predictions)
    assert all(abs(sum(p["probabilities"].values()) - 1) < 1e-3 for p in predictions)
