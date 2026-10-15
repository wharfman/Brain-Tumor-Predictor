"""Evaluate trained weights on a held-out split.

Classification reports top-1/top-5 accuracy on the test split and saves a confusion matrix
to runs/<task>/eval/. Segmentation reports box and mask mAP on the validation split.

Usage:
    python -m src.evaluate --task classify [--split test] [--weights models/classify_best.pt]
"""

import argparse

from src import config, dataloader
from src.model import load_best_model

DEFAULT_SPLIT = {"classify": "test", "segment": "val"}


def evaluate(task: str = "classify", weights: str | None = None, split: str | None = None, **overrides) -> dict:
    model = load_best_model(task, weights)
    metrics = model.val(
        data=str(dataloader.get_data(task)),
        split=split or DEFAULT_SPLIT[task],
        project=str(config.RUNS_DIR / task),
        name="eval",
        **{"imgsz": config.TRAIN_DEFAULTS[task]["imgsz"], **overrides},
    )
    results = {key: round(float(value), 4) for key, value in metrics.results_dict.items()}
    for key, value in results.items():
        print(f"{key:<30} {value}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=config.TASKS, default="classify")
    parser.add_argument("--weights", help="checkpoint to evaluate (default: models/<task>_best.pt)")
    parser.add_argument("--split", help="dataset split to evaluate on (default: test for classify, val for segment)")
    parser.add_argument("--device", help="e.g. 0 for the first GPU, or cpu (default: auto)")
    args = parser.parse_args()

    evaluate(args.task, weights=args.weights, split=args.split, device=args.device)


if __name__ == "__main__":
    main()
