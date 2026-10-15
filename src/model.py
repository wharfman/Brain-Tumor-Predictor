"""Build and load YOLO models (PyTorch CNNs via Ultralytics) for each task."""

import platform
from pathlib import Path

import torch
from ultralytics import YOLO

from src import config

# On ROCm, torch.backends.cudnn controls MIOpen. See config.DISABLE_MIOPEN.
if config.DISABLE_MIOPEN and torch.version.hip and platform.system() == "Windows":
    torch.backends.cudnn.enabled = False


def build_model(task: str, weights: str | Path | None = None) -> YOLO:
    """Create a model to train.

    `weights` can be a pretrained checkpoint (e.g. yolo11s-cls.pt, downloaded on first use)
    or an architecture YAML (e.g. yolo11n-cls.yaml) to train from scratch. Defaults to the
    pretrained checkpoint for the task in config.BASE_WEIGHTS.
    """
    if task not in config.TASKS:
        raise ValueError(f"Unknown task '{task}'. Choose from {config.TASKS}.")
    return _check_task(YOLO(str(weights or config.BASE_WEIGHTS[task])), task)


def best_weights_path(task: str) -> Path:
    return config.MODELS_DIR / f"{task}_best.pt"


def load_best_model(task: str, weights: str | Path | None = None) -> YOLO:
    """Load trained weights for inference, defaulting to models/<task>_best.pt."""
    path = Path(weights) if weights else best_weights_path(task)
    if not path.is_file():
        raise FileNotFoundError(f"No trained weights at {path}. Run `python -m src.train --task {task}` first.")
    return _check_task(YOLO(str(path)), task)


def _check_task(model: YOLO, task: str) -> YOLO:
    # Ultralytics infers the task from the checkpoint/YAML; passing task= would silently override it.
    if model.task != task:
        raise ValueError(f"'{model.ckpt_path or model.cfg}' is a {model.task} model, not {task}.")
    return model
