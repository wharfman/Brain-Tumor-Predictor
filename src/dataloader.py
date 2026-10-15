"""Locate and validate the datasets that the YOLO trainers load.

Ultralytics builds its own PyTorch DataLoaders (with augmentation) from a dataset path, so
this module's job is to resolve that path for a task and fail early with a clear message
if the dataset is missing or malformed.
"""

from pathlib import Path

import yaml

from src import config

CLASSIFY_SPLITS = ("train", "val", "test")


def count_images(split_dir: Path) -> dict[str, int]:
    """Count images per class folder in a classification split."""
    return {
        class_dir.name: sum(1 for p in class_dir.iterdir() if p.suffix.lower() in config.IMAGE_EXTENSIONS)
        for class_dir in sorted(split_dir.iterdir())
        if class_dir.is_dir()
    }


def get_classify_data(data_dir: Path | None = None) -> Path:
    """Return the processed classification dataset directory after checking its layout."""
    data_dir = data_dir or config.CLASSIFY_DATA_DIR
    if not data_dir.is_dir():
        raise FileNotFoundError(f"No processed dataset at {data_dir}. Run `python -m src.preprocess` first.")

    for split in CLASSIFY_SPLITS:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split folder {split_dir}. Re-run `python -m src.preprocess --overwrite`.")
        counts = count_images(split_dir)
        missing = [name for name in config.CLASS_NAMES if counts.get(name, 0) == 0]
        if missing:
            raise ValueError(f"Split '{split}' has no images for classes: {', '.join(missing)}")
    return data_dir


def get_segment_data(yaml_path: Path | None = None) -> Path:
    """Return the YOLO segmentation dataset YAML after checking its images and labels exist.

    Expected layout (the standard Ultralytics/Roboflow export):
        data/segmentation/data.yaml
        data/segmentation/images/{train,val}/*.jpg
        data/segmentation/labels/{train,val}/*.txt   # one polygon per line: class x1 y1 x2 y2 ...
    """
    yaml_path = yaml_path or config.SEGMENT_DATA_YAML
    if not yaml_path.is_file():
        raise FileNotFoundError(
            f"No segmentation dataset at {yaml_path}. The current dataset only has class labels; "
            "segmentation needs per-image tumor masks/polygons (see README)."
        )

    spec = yaml.safe_load(yaml_path.read_text())
    root = Path(spec.get("path", "."))
    if not root.is_absolute():
        root = yaml_path.parent / root

    for split in ("train", "val"):
        if split not in spec:
            raise ValueError(f"{yaml_path} is missing the '{split}' key")
        images_dir = root / spec[split]
        labels_dir = Path(str(images_dir).replace("images", "labels"))
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Missing segmentation images folder {images_dir}")
        if not labels_dir.is_dir() or not any(labels_dir.glob("*.txt")):
            raise FileNotFoundError(f"No label .txt files in {labels_dir}")
    return yaml_path


def get_data(task: str) -> Path:
    if task == "classify":
        return get_classify_data()
    if task == "segment":
        return get_segment_data()
    raise ValueError(f"Unknown task '{task}'. Choose from {config.TASKS}.")
