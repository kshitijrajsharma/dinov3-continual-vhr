"""Training dynamics of continued-pretraining runs: drift from ImageNet and anchor engagement.

Four panels from the per-run metrics CSVs: ImageNet-kNN (drift from the source domain), feature
std (representation spread), Gram loss (patch anchor), and CLS-anchor loss (semantic anchor).

    uv run python -m dinov3_vhr.figures.training_dynamics \
        --run gram0=out/vits_gram0/version_0/metrics.csv \
        --run full=out/vits_full/version_0/metrics.csv \
        --out out/figures/training_dynamics.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PANELS = [
    ("monitor/imagenet_knn", "Drift from ImageNet (kNN accuracy)", "ImageNet kNN"),
    ("monitor/feature_std", "Feature std (representation spread)", "feature std"),
    ("gram_loss", "Patch (Gram) anchor loss", "gram loss"),
    ("cls_anchor_loss", "CLS anchor loss", "cls anchor loss"),
]


def series(frame: pd.DataFrame, column: str) -> tuple:
    """Step and value arrays for one metric, dropping the rows where it was not logged."""
    present = frame[["step", column]].dropna()
    return present["step"].to_numpy(), present[column].to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        required=True,
        help="name=metrics.csv, repeatable",
    )
    parser.add_argument("--out", type=Path, default=Path("out/figures/training_dynamics.png"))
    args = parser.parse_args()

    runs = {}
    for spec in args.run:
        name, sep, path = spec.partition("=")
        if not sep:
            raise ValueError(f"run spec must be name=metrics.csv, got {spec!r}")
        runs[name] = pd.read_csv(path)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (column, title, ylabel) in zip(axes.flat, PANELS, strict=True):
        for name, frame in runs.items():
            if column not in frame:
                continue
            x, y = series(frame, column)
            if len(y) and (column.startswith("monitor") or y.max() > 0):
                ax.plot(x, y, label=name, lw=1.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    fig.suptitle("Continued pretraining: training dynamics", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
