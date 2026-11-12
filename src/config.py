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

IMAGE_SIZE = 640
VAL_FRACTION = 0.15
SEED = 42

# Image pipeline settings (see src/transforms.py)
CROP_THRESHOLD = 45  # pixels brighter than this count as head when cropping off the black border
BLUR_KERNEL = (5, 5)
MORPH_KERNEL = 3

# Synthetic copies made per training image by src/augment.py (val/test are never augmented)
AUGMENT_COPIES = 1

TASKS = ("classify", "segment")

# AMD's Windows ROCm build can't compile MIOpen kernels at runtime (no C++ standard headers),
# so batch norm crashes with miopenStatusUnknownError. With MIOpen off, PyTorch's own GPU
# kernels are used instead. Set to False if a future ROCm release fixes this.
DISABLE_MIOPEN = True

# Windows ROCm caps pinned (page-locked) host memory at ~9.6 GB on this machine. Ultralytics always
# pins its dataloader batches and PyTorch caches freed pinned blocks, so a run fills the cap after a
# few epochs and dies with "CUDA error: out of memory" in the pin memory thread. With pinning off,
# batches are copied to the GPU from ordinary memory, which is slightly slower.
DISABLE_PIN_MEMORY = True

# Pretrained Ultralytics checkpoints used as the starting point for fine-tuning.
BASE_WEIGHTS = {
    # Small (5.5M params) rather than nano (1.6M): more capacity, and it still fits a 16 GB GPU at 640 px.
    "classify": "yolo11s-cls.pt",
    "segment": "yolo11n-seg.pt",
}

# CBAM attention (see src/cbam.py) after these classification backbone layers: the C3k2 stages at
# 1/8, 1/16 and 1/32 resolution (80x80, 40x40, 20x20 for a 640 input). Layer 2 (1/4 resolution, mostly
# edges) is skipped for cost, and layer 9 (C2PSA) already has self-attention. () turns CBAM off.
CBAM_LAYERS = (4, 6, 8)
CBAM_REDUCTION = 16  # channel-attention MLP hidden size is channels / CBAM_REDUCTION
CBAM_KERNEL = 7  # spatial-attention convolution size

# SPPF+ multi-scale pooling (see src/sppf_plus.py), inserted in front of this classification backbone
# layer: 9 is C2PSA, the same spot as SPPF in the detection model, on the 20x20 map for a 640 input.
# None turns SPPF+ off. The max pools run in series, so kernels 5, 9, 13 see 5, 13 and 25 px windows.
SPPF_PLUS_LAYER = 9
SPPF_PLUS_KERNELS = (5, 9, 13)

# BiFPN head (see src/bifpn.py) over the outputs of these layers, replacing the Classify head: the
# 80x80, 40x40 and 20x20 maps (P3, P4, P5) for a 640 input. Layer 9 is C2PSA (after SPPF+). () turns
# BiFPN off. BIFPN_REPEATS is how many times the BiFPN layer is stacked (EfficientDet-D0 uses 3).
BIFPN_LAYERS = (4, 6, 9)
BIFPN_REPEATS = 2

TRAIN_DEFAULTS = {
    # erasing=0: random erasing blacks out patches of the scan and can hide the tumor itself.
    # scale=0.2: the random crop keeps at least 80% of the image area (default 0.5 keeps 50%, which can
    #   also cut the tumor out). Offline copies from src/augment.py already add rotation, zoom and crops.
    # AdamW + cosine schedule: optimizer=auto picks MuSGD at lr 0.01, which made validation accuracy
    #   swing by up to 14 points between epochs when fine-tuning the pretrained checkpoint.
    "classify": {
        "epochs": 50,
        "imgsz": IMAGE_SIZE,
        "batch": 32,
        "patience": 15,
        "erasing": 0.0,
        "scale": 0.2,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "cos_lr": True,
    },
    "segment": {"epochs": 100, "imgsz": 640, "batch": 32, "patience": 15},
}
