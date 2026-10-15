import numpy as np
from PIL import Image

from src import transforms


def make_scan(width=120, height=80) -> Image.Image:
    """A fake RGB scan: a bright 'head' rectangle with a brighter 'tumor', on a black background."""
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[20:60, 30:90] = 150
    pixels[35:45, 50:60] = 250
    return Image.fromarray(pixels)


def test_crop_to_brain_removes_black_border():
    cropped = transforms.crop_to_brain(transforms.to_grayscale(make_scan()))
    # The 40x60 head, plus at most a couple of pixels of margin from the blurred mask.
    height, width = cropped.shape
    assert 40 <= height <= 44 and 60 <= width <= 64


def test_crop_to_brain_keeps_image_without_bright_region():
    blank = np.zeros((50, 50), dtype=np.uint8)
    assert transforms.crop_to_brain(blank).shape == (50, 50)


def test_letterbox_pads_to_square_without_stretching():
    wide = np.full((50, 100), 255, dtype=np.uint8)
    result = transforms.letterbox(wide, 32)

    assert result.shape == (32, 32)
    assert result[0, 0] == 0  # padded top and bottom
    assert result[16, 16] == 255


def test_erode_dilate_removes_isolated_specks():
    image = np.zeros((20, 20), dtype=np.uint8)
    image[10, 10] = 255  # single-pixel speck
    image[2:8, 2:8] = 255  # larger region
    result = transforms.erode_dilate(image)

    assert result[10, 10] == 0
    assert result[4, 4] == 255


def test_sharpen_increases_edge_contrast():
    image = np.zeros((20, 20), dtype=np.uint8)
    image[:, 10:] = 100
    sharpened = transforms.sharpen(image)
    assert sharpened[5, 10] > 100  # bright side of the edge gets brighter


def test_preprocess_image_outputs_square_grayscale():
    stages = transforms.preprocess_stages(make_scan(), size=64)
    assert list(stages) == ["grayscale", "cropped", "resized", "blurred", "sharpened", "eroded_dilated"]

    result = transforms.preprocess_image(make_scan(), size=64)
    assert result.mode == "L"
    assert result.size == (64, 64)
