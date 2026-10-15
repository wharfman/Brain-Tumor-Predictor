import pytest
from PIL import Image

from src import config
from src.preprocess import build_classify_dataset, split_train_val


def test_split_train_val_is_deterministic_and_disjoint(tmp_path):
    files = [tmp_path / f"{i}.jpg" for i in range(20)]
    train, val = split_train_val(files, 0.25, seed=1)

    assert len(val) == 5
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(files)
    assert split_train_val(files, 0.25, seed=1) == (train, val)


def test_build_classify_dataset_layout_and_counts(tmp_path, raw_dataset):
    out_dir = tmp_path / "processed"
    counts = build_classify_dataset(*raw_dataset, out_dir=out_dir, size=32, val_fraction=0.2)

    for class_name in config.CLASS_NAMES:
        assert counts["train"][class_name] == 8
        assert counts["val"][class_name] == 2
        assert counts["test"][class_name] == 10

    output = next((out_dir / "train" / config.CLASS_NAMES[0]).iterdir())
    assert output.suffix == ".jpg"
    with Image.open(output) as image:
        assert image.size == (32, 32)
        assert image.mode == "L"


def test_build_classify_dataset_refuses_to_overwrite(tmp_path, raw_dataset):
    out_dir = tmp_path / "processed"
    build_classify_dataset(*raw_dataset, out_dir=out_dir, size=32)

    with pytest.raises(FileExistsError):
        build_classify_dataset(*raw_dataset, out_dir=out_dir, size=32)
    build_classify_dataset(*raw_dataset, out_dir=out_dir, size=32, overwrite=True)
