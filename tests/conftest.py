import random
from pathlib import Path

import pytest
from PIL import Image

from src import config


def make_raw_dataset(root: Path, images_per_class: int) -> tuple[Path, Path]:
    """Create a tiny fake dataset mimicking data/{Train,Test}/<class>/ with mixed sizes and modes."""
    rng = random.Random(0)
    for split in ("Train", "Test"):
        for class_name in config.CLASS_NAMES:
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True)
            for i in range(images_per_class):
                size = rng.choice([(64, 64), (80, 48), (40, 72)])
                mode = rng.choice(["L", "RGB"])
                Image.new(mode, size, color=rng.randrange(256) if mode == "L" else (rng.randrange(256),) * 3).save(
                    class_dir / f"img_{i}{rng.choice(['.jpg', '.JPG', '.jpeg'])}"
                )
    return root / "Train", root / "Test"


@pytest.fixture
def raw_dataset(tmp_path):
    return make_raw_dataset(tmp_path / "raw", images_per_class=10)


@pytest.fixture
def processed_dataset(tmp_path, raw_dataset, monkeypatch):
    """Build a processed dataset and point config at temporary data/models/runs folders."""
    from src.preprocess import build_classify_dataset

    out_dir = tmp_path / "processed"
    build_classify_dataset(*raw_dataset, out_dir=out_dir, size=32, val_fraction=0.2)
    monkeypatch.setattr(config, "CLASSIFY_DATA_DIR", out_dir)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    return out_dir
