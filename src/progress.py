"""Readable console progress for classification training.

Replaces Ultralytics' per-epoch progress bars with one line per phase, showing accuracy and the
whole seconds left in that phase:

    Epoch 3/50
      Training    ━━━━━━──────   412/1214 batches   loss 0.3121   accuracy  88.4%   97s remaining
      Validation  ━━━━━━━━━━━━    64/64 batches                   accuracy  91.2%    0s remaining
      Epoch 3 done in 214s: training accuracy 89.0%, validation accuracy 91.2% (best 91.5%)

Usage:
    with progress.attach(model):
        model.train(...)
"""

import logging
import sys
import time

import torch
from ultralytics.utils import LOGGER
from ultralytics.utils.loss import v8ClassificationLoss
from ultralytics.utils.torch_utils import unwrap_model

BAR_WIDTH = 12
REFRESH_INTERVAL = 0.1  # seconds between redraws of a live line

CALLBACKS = (
    "on_train_start",
    "on_train_epoch_start",
    "on_train_batch_end",
    "on_train_epoch_end",
    "on_val_start",
    "on_val_batch_end",
    "on_val_end",
    "on_fit_epoch_end",
)


class AccuracyLoss(v8ClassificationLoss):
    """Ultralytics' classification loss, which also counts correct training predictions.

    Validation runs without gradients, so only training batches are counted.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.correct = self.seen = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.seen if self.seen else 0.0

    def __call__(self, preds, batch):
        loss, items = super().__call__(preds, batch)
        if torch.is_grad_enabled():
            logits = preds[1] if isinstance(preds, (list, tuple)) else preds
            self.correct += int((logits.argmax(1) == batch["cls"]).sum())
            self.seen += len(batch["cls"])
        return loss, items


def seconds_remaining(elapsed: float, done: int, total: int) -> int:
    """Estimate whole seconds left from the average time per batch so far."""
    return round(elapsed / done * (total - done)) if done else 0


def progress_bar(done: int, total: int, full: str = "━", empty: str = "─") -> str:
    filled = min(BAR_WIDTH, round(BAR_WIDTH * done / total)) if total else BAR_WIDTH
    return full * filled + empty * (BAR_WIDTH - filled)


def format_line(
    phase: str, done: int, total: int, elapsed: float, accuracy: float, loss: float | None = None, ascii_bar=False
) -> str:
    bar = progress_bar(done, total, *("#", "-") if ascii_bar else ())
    loss_text = f"loss {loss:.4f}" if loss is not None else ""
    return (
        f"  {phase:<10}  {bar}  {done:>{len(str(total))}}/{total} batches   {loss_text:<11}   "
        f"accuracy {accuracy:6.1%}   {seconds_remaining(elapsed, done, total):>3}s remaining"
    )


class ProgressPrinter:
    """Ultralytics callbacks that print the progress lines. Use as a context manager around training."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.live = self.stream.isatty()  # redraw lines in place; otherwise print each finished line once
        try:
            "━─".encode(self.stream.encoding or "ascii")
            self.ascii_bar = False
        except (LookupError, UnicodeEncodeError):  # e.g. Windows cp1252 when piped to a file
            self.ascii_bar = True
        self.criterion = AccuracyLoss()
        self.saved_level = None
        self.epoch_started = None  # set while a training epoch is in progress
        self.val_correct = self.val_seen = 0
        self.val_accuracy = None
        self.best_val_accuracy = 0.0

    # Ultralytics' progress bars draw only while its logger shows INFO, so raise it during each phase.
    def _quiet(self) -> None:
        if self.saved_level is None:
            self.saved_level = LOGGER.level
            LOGGER.setLevel(logging.WARNING)

    def _restore(self) -> None:
        if self.saved_level is not None:
            LOGGER.setLevel(self.saved_level)
            self.saved_level = None

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self._restore()

    def _start_phase(self, phase: str, total: int) -> None:
        self.phase, self.total, self.done = phase, total, 0
        self.started = self.last_draw = time.time()

    def _draw(self, accuracy: float, loss: float | None = None, final: bool = False) -> None:
        now = time.time()
        if not final and (not self.live or now - self.last_draw < REFRESH_INTERVAL):
            return
        self.last_draw = now
        line = format_line(self.phase, self.done, self.total, now - self.started, accuracy, loss, self.ascii_bar)
        self.stream.write((f"\r\033[K{line}" if self.live else line) + ("\n" if final else ""))
        self.stream.flush()

    def on_train_start(self, trainer) -> None:
        # Set after the trainer copies the model for EMA, so validation keeps the default loss.
        unwrap_model(trainer.model).criterion = self.criterion

    def on_train_epoch_start(self, trainer) -> None:
        self._quiet()
        self.stream.write(f"\nEpoch {trainer.epoch + 1}/{trainer.epochs}\n")
        self.epoch_started = time.time()
        self.val_accuracy = None
        self.criterion.reset()
        self._start_phase("Training", len(trainer.train_loader))

    def _train_loss(self, trainer) -> float | None:
        return float(trainer.tloss["loss"]) if trainer.tloss else None

    def on_train_batch_end(self, trainer) -> None:
        self.done += 1
        self._draw(self.criterion.accuracy, self._train_loss(trainer))

    def on_train_epoch_end(self, trainer) -> None:
        self._draw(self.criterion.accuracy, self._train_loss(trainer), final=True)

    def on_val_start(self, validator) -> None:
        self._quiet()
        self.val_correct = self.val_seen = 0
        self._start_phase("Validation", len(validator.dataloader))

    def on_val_batch_end(self, validator) -> None:
        top1, targets = validator.pred[-1][:, 0], validator.targets[-1]
        self.val_correct += int((top1 == targets).sum())
        self.val_seen += len(targets)
        self.done += 1
        self._draw(self.val_correct / self.val_seen)

    def on_val_end(self, validator) -> None:
        self.val_accuracy = self.val_correct / self.val_seen if self.val_seen else 0.0
        self._draw(self.val_accuracy, final=True)
        if not validator.training:  # the final check of best.pt after training
            self.stream.write(f"Final validation (best.pt): accuracy {self.val_accuracy:.1%}\n")
        self._restore()

    def on_fit_epoch_end(self, trainer) -> None:
        if self.epoch_started is None:  # also fired after the final validation, which is not an epoch
            return
        summary = f"  Epoch {trainer.epoch + 1} done in {round(time.time() - self.epoch_started)}s: "
        summary += f"training accuracy {self.criterion.accuracy:.1%}"
        if self.val_accuracy is not None:
            self.best_val_accuracy = max(self.best_val_accuracy, self.val_accuracy)
            summary += f", validation accuracy {self.val_accuracy:.1%} (best {self.best_val_accuracy:.1%})"
        self.stream.write(summary + "\n")
        self.stream.flush()
        self.epoch_started = None
        self._restore()


def attach(model) -> ProgressPrinter:
    """Register the progress callbacks on a classification YOLO model."""
    printer = ProgressPrinter()
    for event in CALLBACKS:
        model.add_callback(event, getattr(printer, event))
    return printer
