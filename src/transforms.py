"""Image pipeline applied to every MRI scan, both when building the dataset and at prediction time.

    1. grayscale
    2. crop to the brain: find its outer contour and drop the black border around it
    3. letterbox resize to IMAGE_SIZE x IMAGE_SIZE (pad to square, so anatomy isn't stretched)
    4. Gaussian blur to reduce noise
    5. high-pass filter to sharpen edges and fine detail
    6. erosion then dilation to remove small bright specks while keeping larger regions' size
"""

import cv2
import numpy as np
from PIL import Image, ImageOps

from src import config

# Identity plus a Laplacian high-pass: boosts edges while keeping the original image.
HIGH_PASS_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)


def to_grayscale(image: Image.Image) -> np.ndarray:
    return np.asarray(ImageOps.exif_transpose(image).convert("L"))


def crop_to_brain(gray: np.ndarray, threshold: int = config.CROP_THRESHOLD) -> np.ndarray:
    """Crop to the bounding box of the largest bright region (the head), removing black borders."""
    mask = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), threshold, 255, cv2.THRESH_BINARY)[1]
    # Clean the mask so stray noise pixels outside the head don't widen the crop.
    mask = cv2.dilate(cv2.erode(mask, None, iterations=2), None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray
    # The bounding box spans the contour's extreme left, right, top and bottom points.
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return gray[y : y + h, x : x + w]


def letterbox(gray: np.ndarray, size: int) -> np.ndarray:
    height, width = gray.shape
    side = max(height, width)
    canvas = np.zeros((side, side), dtype=gray.dtype)
    top, left = (side - height) // 2, (side - width) // 2
    canvas[top : top + height, left : left + width] = gray
    interpolation = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    return cv2.resize(canvas, (size, size), interpolation=interpolation)


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(gray, config.BLUR_KERNEL, 0)


def sharpen(gray: np.ndarray) -> np.ndarray:
    return cv2.filter2D(gray, -1, HIGH_PASS_KERNEL)


def erode_dilate(gray: np.ndarray) -> np.ndarray:
    kernel = np.ones((config.MORPH_KERNEL, config.MORPH_KERNEL), np.uint8)
    return cv2.dilate(cv2.erode(gray, kernel), kernel)


def preprocess_stages(image: Image.Image, size: int = config.IMAGE_SIZE) -> dict[str, np.ndarray]:
    """Run the pipeline and return the image after every stage (useful for inspecting it)."""
    stages = {"grayscale": to_grayscale(image)}
    stages["cropped"] = crop_to_brain(stages["grayscale"])
    stages["resized"] = letterbox(stages["cropped"], size)
    stages["blurred"] = denoise(stages["resized"])
    stages["sharpened"] = sharpen(stages["blurred"])
    stages["eroded_dilated"] = erode_dilate(stages["sharpened"])
    return stages


def preprocess_image(image: Image.Image, size: int = config.IMAGE_SIZE) -> Image.Image:
    """Run the full pipeline, returning a size x size grayscale image."""
    return Image.fromarray(preprocess_stages(image, size)["eroded_dilated"])
