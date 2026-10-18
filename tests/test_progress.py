import pytest
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


def test_validation_shows_running_mean_loss():
    import io
    from types import SimpleNamespace

    from src.progress import ProgressPrinter

    stream = io.StringIO()
    printer = ProgressPrinter(stream)
    validator = SimpleNamespace(training=True, dataloader=[None, None], pred=[], targets=[], loss={"loss": torch.tensor(0.0)})
    printer.on_val_start(validator)
    for batch_loss in (0.2, 0.4):  # Ultralytics adds each batch's loss to the running sum
        validator.loss["loss"] += batch_loss
        validator.pred.append(torch.tensor([[0], [1]]))
        validator.targets.append(torch.tensor([0, 0]))
        printer.on_val_batch_end(validator)
    printer.on_val_end(validator)
    assert printer.val_loss == pytest.approx(0.3)
    assert "loss 0.3000" in stream.getvalue()

    printer.epoch_started = 0.0
    printer.on_fit_epoch_end(SimpleNamespace(epoch=0))
    assert "validation loss 0.3000" in stream.getvalue()

    # The final check of best.pt runs outside training, where Ultralytics keeps no loss.
    final = SimpleNamespace(training=False, dataloader=[None], pred=[torch.tensor([[0]])], targets=[torch.tensor([0])])
    printer.on_val_start(final)
    printer.on_val_batch_end(final)
    printer.on_val_end(final)
    assert printer.val_loss is None
