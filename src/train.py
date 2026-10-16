"""Fine-tune a YOLO model and save its best weights to models/.

Full run outputs (every checkpoint, curves, confusion matrices) go to runs/<task>/<name>/;
only the best checkpoint is copied to models/<task>_best.pt.

Usage:
    python -m src.train --task classify [--epochs 50] [--batch 32] [--weights yolo11s-cls.pt]
    python -m src.train --task segment
"""

import argparse
import shutil
from contextlib import nullcontext
from pathlib import Path

from src import config, dataloader, progress
from src.model import best_weights_path, build_model


def train(
    task: str = "classify",
    weights: str | None = None,
    device: str | None = None,
    name: str | None = None,
    **overrides,
) -> Path:
    """Train a model for `task` and return the path of the saved best weights.

    `overrides` are passed to Ultralytics' trainer on top of config.TRAIN_DEFAULTS
    (e.g. epochs=10, batch=64, lr0=0.001).
    """
    data = dataloader.get_data(task)
    model = build_model(task, weights)

    # Classification gets readable per-epoch progress lines with accuracy (see src/progress.py).
    with progress.attach(model) if task == "classify" else nullcontext():
        model.train(
            data=str(data),
            project=str(config.RUNS_DIR / task),
            name=name or task,
            device=device,
            seed=config.SEED,
            **{**config.TRAIN_DEFAULTS[task], **overrides},
        )

    best = Path(model.trainer.best)
    destination = best_weights_path(task)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, destination)
    print(f"Best weights saved to {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=config.TASKS, default="classify")
    parser.add_argument("--weights", help="starting checkpoint or architecture YAML (default: config.BASE_WEIGHTS)")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device", help="e.g. 0 for the first GPU, or cpu (default: auto)")
    parser.add_argument("--name", help="run folder name under runs/<task>/")
    args = parser.parse_args()

    overrides = {k: v for k, v in {"epochs": args.epochs, "batch": args.batch, "imgsz": args.imgsz}.items() if v}
    train(args.task, weights=args.weights, device=args.device, name=args.name, **overrides)


if __name__ == "__main__":
    main()
