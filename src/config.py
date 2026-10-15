"""Central configuration: paths, class names and default hyperparameters."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw dataset as downloaded: data/{Train,Test}/<class>/*.jpg
DATA_DIR = PROJECT_ROOT / "data"
RAW_TRAIN_DIR = DATA_DIR / "Train"
RAW_TEST_DIR = DATA_DIR / "Test"

# Resized dataset in YOLO classification layout: {train,val,test}/<class>/*.jpg
CLASSIFY_DATA_DIR = DATA_DIR / "processed" / "classify"

# YOLO segmentation dataset (images + polygon labels). Not included in the current dataset.
SEGMENT_DATA_YAML = DATA_DIR / "segmentation" / "data.yaml"

# Best weights from training are copied here; everything else from a run stays in RUNS_DIR.
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"

CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

IMAGE_SIZE = 224
VAL_FRACTION = 0.15
SEED = 42

TASKS = ("classify", "segment")

# AMD's Windows ROCm build can't compile MIOpen kernels at runtime (no C++ standard headers),
# so batch norm crashes with miopenStatusUnknownError. With MIOpen off, PyTorch's own GPU
# kernels are used instead. Set to False if a future ROCm release fixes this.
DISABLE_MIOPEN = True

# Pretrained Ultralytics checkpoints used as the starting point for fine-tuning.
BASE_WEIGHTS = {
    "classify": "yolo11n-cls.pt",
    "segment": "yolo11n-seg.pt",
}

TRAIN_DEFAULTS = {
    "classify": {"epochs": 50, "imgsz": IMAGE_SIZE, "batch": 32, "patience": 10},
    "segment": {"epochs": 100, "imgsz": 640, "batch": 16, "patience": 20},
}
