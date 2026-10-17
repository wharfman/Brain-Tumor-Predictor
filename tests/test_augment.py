import numpy as np
from PIL import Image

from src.augment import AUG_MARKER, augment_image, augment_split


def make_scan(size=64):
    """A bright disk with a brighter off-center spot, so rotations and flips change the pixels."""
    yy, xx = np.mgrid[:size, :size]
    scan = np.where((yy - size / 2) ** 2 + (xx - size / 2) ** 2 < (size / 3) ** 2, 120, 0)
    scan[size // 4 : size // 3, size // 2 : size // 2 + 8] = 230
    return scan.astype(np.uint8)


def test_augment_image_keeps_shape_and_changes_pixels():
    scan = make_scan()
    copy = augment_image(scan, np.random.default_rng(0))
    assert copy.shape == scan.shape and copy.dtype == np.uint8
    assert not np.array_equal(copy, scan)


def test_copies_draw_different_settings():
    scan, rng = make_scan(), np.random.default_rng(0)
    copies = [augment_image(scan, rng) for _ in range(5)]
    assert all(not np.array_equal(a, b) for i, a in enumerate(copies) for b in copies[i + 1 :])


def test_augment_split_adds_copies_without_touching_originals(tmp_path):
    train_dir = tmp_path / "train"
    originals = {}
    for class_name in ("glioma", "notumor"):
        (train_dir / class_name).mkdir(parents=True)
        for i in range(3):
            path = train_dir / class_name / f"img_{i}.jpg"
            Image.fromarray(make_scan()).save(path)
            originals[path] = path.read_bytes()
    val_file = tmp_path / "val" / "glioma" / "img_9.jpg"
    val_file.parent.mkdir(parents=True)
    Image.fromarray(make_scan()).save(val_file)

    assert augment_split(train_dir, copies=2, seed=0) == {"glioma": 6, "notumor": 6}
    # Re-running replaces copies instead of piling up or copying the copies
    assert augment_split(train_dir, copies=2, seed=0) == {"glioma": 6, "notumor": 6}

    for path, data in originals.items():
        assert path.read_bytes() == data
    copies = sorted(p.name for p in (train_dir / "glioma").iterdir() if AUG_MARKER in p.stem)
    assert copies == [f"img_{i}{AUG_MARKER}{k}.jpg" for i in range(3) for k in (1, 2)]
    assert [p.name for p in val_file.parent.iterdir()] == ["img_9.jpg"]
