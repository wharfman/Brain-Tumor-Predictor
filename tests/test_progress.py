import torch

from src.progress import AccuracyLoss, format_line, seconds_remaining


def test_accuracy_loss_counts_training_batches_only():
    criterion = AccuracyLoss()
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 0.0]], requires_grad=True)
    batch = {"cls": torch.tensor([0, 1, 1, 1])}

    loss, items = criterion(logits, batch)
    assert set(items) == {"loss"} and loss.requires_grad
    assert (criterion.correct, criterion.seen, criterion.accuracy) == (2, 4, 0.5)

    with torch.no_grad():  # validation
        criterion(logits, batch)
    assert criterion.seen == 4

    criterion.reset()
    assert criterion.accuracy == 0.0


def test_seconds_remaining_is_whole_seconds():
    assert seconds_remaining(elapsed=10.0, done=3, total=10) == 23  # 3.33s per batch, 7 left
    assert seconds_remaining(elapsed=10.0, done=10, total=10) == 0
    assert seconds_remaining(elapsed=0.0, done=0, total=10) == 0


def test_format_line():
    line = format_line("Training", 412, 1214, elapsed=50.0, accuracy=0.884, loss=0.31214)
    assert "412/1214 batches" in line
    assert "loss 0.3121" in line
    assert "accuracy  88.4%" in line
    assert line.endswith(" 97s remaining")

    validation = format_line("Validation", 64, 64, elapsed=5.0, accuracy=0.912, ascii_bar=True)
    assert "############" in validation and "loss" not in validation
    assert validation.endswith("  0s remaining")
