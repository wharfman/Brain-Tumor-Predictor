"""Enlarge the training split with synthetic copies of its images.

Each copy is always rotated and flipped, and may also be cropped, have its brightness, contrast
or gamma changed, and have noise added. Every copy draws its own random settings. Copies are
saved next to the originals as <stem>__aug<k>.jpg, so Ultralytics trains on both. Originals and
the val/test splits are never changed.

Usage:
    python -m src.augment [--copies 1]     # (re)create copies in data/processed/classify/train
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src import config

AUG_MARKER = "__aug"

ROTATION_DEGREES = (5, 30)  # magnitude; the direction is random
ZOOM = (0.85, 1.2)  # below 1 zooms out, above 1 zooms in
MAX_SHIFT = 0.05  # fraction of the image side
CROP_SIDE = (0.80, 0.95)  # fraction of the side kept by a random crop
BRIGHTNESS_SHIFT = (15, 40)  # magnitude in gray levels; up or down
CONTRAST = (0.8, 1.2)
GAMMA = (0.7, 1.4)
GAUSSIAN_SIGMA = (5, 15)
SALT_PEPPER_FRACTION = (0.005, 0.02)
OPTIONAL_STEP_PROBABILITY = 0.5  # for crop, intensity change and noise


def rotate_zoom_shift(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = gray.shape
    angle = rng.uniform(*ROTATION_DEGREES) * rng.choice([-1, 1])
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, rng.uniform(*ZOOM))
    matrix[:, 2] += rng.uniform(-MAX_SHIFT, MAX_SHIFT, size=2) * (width, height)
    return cv2.warpAffine(gray, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=0)


def flip(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # OpenCV flip codes: 1 horizontal, 0 vertical, -1 both
    return cv2.flip(gray, int(rng.choice([1, 0, -1])))


def random_crop(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = gray.shape
    side = rng.uniform(*CROP_SIDE)
    crop_h, crop_w = round(height * side), round(width * side)
    top, left = rng.integers(0, height - crop_h + 1), rng.integers(0, width - crop_w + 1)
    return cv2.resize(gray[top : top + crop_h, left : left + crop_w], (width, height), interpolation=cv2.INTER_CUBIC)


def change_intensity(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pixels = gray.astype(np.float32)
    kind = rng.choice(["brightness", "contrast", "gamma"])
    if kind == "brightness":
        pixels += rng.uniform(*BRIGHTNESS_SHIFT) * rng.choice([-1, 1])
    elif kind == "contrast":
        pixels = (pixels - pixels.mean()) * rng.uniform(*CONTRAST) + pixels.mean()
    else:
        pixels = 255 * (pixels / 255) ** rng.uniform(*GAMMA)
    return np.clip(pixels, 0, 255).astype(np.uint8)


def add_noise(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        noisy = gray + rng.normal(0, rng.uniform(*GAUSSIAN_SIGMA), gray.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    noisy = gray.copy()
    hit = rng.random(gray.shape) < rng.uniform(*SALT_PEPPER_FRACTION)
    noisy[hit] = rng.choice([0, 255], size=int(hit.sum()))
    return noisy


def augment_image(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a randomly altered copy of a grayscale image. It is always rotated and flipped."""
    out = flip(rotate_zoom_shift(gray, rng), rng)
    for step in (random_crop, change_intensity, add_noise):
        if rng.random() < OPTIONAL_STEP_PROBABILITY:
            out = step(out, rng)
    return out


def is_augmented(path: Path) -> bool:
    return AUG_MARKER in path.stem


def augment_split(train_dir: Path, copies: int = config.AUGMENT_COPIES, seed: int = config.SEED) -> dict[str, int]:
    """Replace any existing copies in train_dir/<class>/ with `copies` new ones per original.

    Returns the number of copies written per class.
    """
    rng = np.random.default_rng(seed)
    written: dict[str, int] = {}
    for class_dir in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        images = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in config.IMAGE_EXTENSIONS)
        for old_copy in filter(is_augmented, images):
            old_copy.unlink()

        written[class_dir.name] = 0
        for path in (p for p in images if not is_augmented(p)):
            try:
                with Image.open(path) as image:
                    gray = np.asarray(image.convert("L"))
            except OSError as error:
                print(f"Skipping unreadable image {path}: {error}")
                continue
            for k in range(1, copies + 1):
                copy = Image.fromarray(augment_image(gray, rng))
                copy.save(class_dir / f"{path.stem}{AUG_MARKER}{k}.jpg", quality=95)
                written[class_dir.name] += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--copies", type=int, default=config.AUGMENT_COPIES, help="synthetic copies per training image")
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    train_dir = config.CLASSIFY_DATA_DIR / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"No training split at {train_dir}. Run `python -m src.preprocess` first.")
    for class_name, n in augment_split(train_dir, args.copies, args.seed).items():
        print(f"{class_name:<10} {n:>5} copies")


if __name__ == "__main__":
    main()
