import shutil

import pytest

from src import config, dataloader


def test_get_classify_data_accepts_processed_dataset(processed_dataset):
    assert dataloader.get_data("classify") == processed_dataset
    assert dataloader.count_images(processed_dataset / "test") == {name: 10 for name in config.CLASS_NAMES}


def test_get_classify_data_reports_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.preprocess"):
        dataloader.get_classify_data(tmp_path / "nope")


def test_get_classify_data_reports_missing_class(processed_dataset):
    shutil.rmtree(processed_dataset / "val" / "glioma")
    with pytest.raises(ValueError, match="glioma"):
        dataloader.get_classify_data()


def test_get_segment_data_explains_missing_masks(tmp_path):
    with pytest.raises(FileNotFoundError, match="masks"):
        dataloader.get_segment_data(tmp_path / "data.yaml")


def test_get_segment_data_accepts_yolo_layout(tmp_path):
    root = tmp_path / "segmentation"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        (root / "labels" / split / "a.txt").write_text("0 0.1 0.1 0.5 0.1 0.5 0.5\n")
    yaml_path = root / "data.yaml"
    yaml_path.write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: tumor\n")

    assert dataloader.get_segment_data(yaml_path) == yaml_path


def test_get_data_rejects_unknown_task():
    with pytest.raises(ValueError):
        dataloader.get_data("detect")
