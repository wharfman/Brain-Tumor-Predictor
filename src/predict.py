"""Run a trained model on new MRI scans.

Usage:
    python -m src.predict path/to/scan.jpg
    python -m src.predict path/to/folder --task segment --save
"""

import argparse
from pathlib import Path

from PIL import Image

from src import config
from src.model import load_best_model
from src.preprocess import letterbox


def collect_images(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.suffix.lower() in config.IMAGE_EXTENSIONS)
    if source.is_file():
        return [source]
    raise FileNotFoundError(f"No image or folder at {source}")


def load_for_classification(path: Path) -> Image.Image:
    """Apply the same letterbox resize used in preprocessing so inputs match training data."""
    with Image.open(path) as image:
        return letterbox(image, config.IMAGE_SIZE)


def predict(
    source: str | Path,
    task: str = "classify",
    weights: str | None = None,
    conf: float = 0.25,
    save: bool = False,
) -> list[dict]:
    """Predict on an image or folder. Returns one result dict per image."""
    model = load_best_model(task, weights)
    paths = collect_images(Path(source))
    if not paths:
        raise FileNotFoundError(f"No images found in {source}")

    if task == "classify":
        inputs = [load_for_classification(p) for p in paths]
    else:
        inputs = [str(p) for p in paths]

    predictions = []
    for path, result in zip(paths, model.predict(inputs, conf=conf, verbose=False, stream=True)):
        if save:
            output_dir = config.RUNS_DIR / task / "predict"
            output_dir.mkdir(parents=True, exist_ok=True)
            result.save(filename=str(output_dir / path.name))
        predictions.append({"image": str(path), **summarize(result)})
    return predictions


def summarize(result) -> dict:
    names = result.names
    if result.probs is not None:  # classification
        probabilities = result.probs.data.tolist()
        return {
            "label": names[result.probs.top1],
            "confidence": round(float(result.probs.top1conf), 4),
            "probabilities": {names[i]: round(p, 4) for i, p in enumerate(probabilities)},
        }
    # segmentation: one entry per detected tumor region
    regions = [
        {"label": names[int(cls)], "confidence": round(float(score), 4), "box_xyxy": [round(v, 1) for v in box]}
        for cls, score, box in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy.tolist())
    ]
    return {"regions": regions, "has_masks": result.masks is not None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="image file or folder of images")
    parser.add_argument("--task", choices=config.TASKS, default="classify")
    parser.add_argument("--weights", help="checkpoint to use (default: models/<task>_best.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="minimum confidence for segmentation regions")
    parser.add_argument("--save", action="store_true", help="save annotated images to runs/<task>/predict/")
    args = parser.parse_args()

    for prediction in predict(args.source, task=args.task, weights=args.weights, conf=args.conf, save=args.save):
        if args.task == "classify":
            print(f"{prediction['image']}: {prediction['label']} ({prediction['confidence']:.1%})")
        else:
            labels = ", ".join(f"{r['label']} {r['confidence']:.0%}" for r in prediction["regions"]) or "no tumor found"
            print(f"{prediction['image']}: {labels}")


if __name__ == "__main__":
    main()
