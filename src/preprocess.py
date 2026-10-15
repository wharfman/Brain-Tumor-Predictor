"""Resize the raw MRI scans and arrange them in the folder layout YOLO classification expects.

    data/Train/<class>/*  ->  data/processed/classify/train/<class>/*.jpg
                          ->  data/processed/classify/val/<class>/*.jpg   (stratified hold-out)
    data/Test/<class>/*   ->  data/processed/classify/test/<class>/*.jpg

Usage:
    python -m src.preprocess [--size 224] [--val-fraction 0.15] [--overwrite]
"""

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, ImageOps

from src import config


def letterbox(image: Image.Image, size: int) -> Image.Image:
    """Pad an image to a square with black borders, then resize it.

    Padding instead of stretching keeps the anatomy (and tumor shape) undistorted, and
    black matches the MRI background.
    """
    image = ImageOps.exif_transpose(image).convert("RGB")
    side = max(image.size)
    canvas = Image.new("RGB", (side, side))
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def list_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in config.IMAGE_EXTENSIONS)


def split_train_val(files: list[Path], val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    n_val = round(len(shuffled) * val_fraction)
    return shuffled[n_val:], shuffled[:n_val]


def process_images(files: list[Path], dest_dir: Path, size: int) -> int:
    """Letterbox each image into dest_dir as JPEG. Returns the number written."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in files:
        try:
            with Image.open(path) as image:
                letterbox(image, size).save(dest_dir / f"{path.stem}.jpg", quality=95)
            written += 1
        except OSError as error:
            print(f"Skipping unreadable image {path}: {error}")
    return written


def build_classify_dataset(
    raw_train_dir: Path = config.RAW_TRAIN_DIR,
    raw_test_dir: Path = config.RAW_TEST_DIR,
    out_dir: Path = config.CLASSIFY_DATA_DIR,
    size: int = config.IMAGE_SIZE,
    val_fraction: float = config.VAL_FRACTION,
    seed: int = config.SEED,
    overwrite: bool = False,
) -> dict[str, dict[str, int]]:
    """Build the processed dataset. Returns image counts as {split: {class: count}}."""
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} already exists. Pass overwrite=True (--overwrite) to rebuild it.")
        shutil.rmtree(out_dir)

    counts: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    for class_name in config.CLASS_NAMES:
        train_files, val_files = split_train_val(list_images(raw_train_dir / class_name), val_fraction, seed)
        test_files = list_images(raw_test_dir / class_name)

        for split, files in (("train", train_files), ("val", val_files), ("test", test_files)):
            counts[split][class_name] = process_images(files, out_dir / split / class_name, size)
            print(f"{split:>5} / {class_name:<10} {counts[split][class_name]:>5} images")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=config.IMAGE_SIZE, help="output image side length in pixels")
    parser.add_argument("--val-fraction", type=float, default=config.VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--overwrite", action="store_true", help="delete and rebuild an existing processed dataset")
    args = parser.parse_args()

    build_classify_dataset(size=args.size, val_fraction=args.val_fraction, seed=args.seed, overwrite=args.overwrite)
    print(f"Processed dataset written to {config.CLASSIFY_DATA_DIR}")


if __name__ == "__main__":
    main()
