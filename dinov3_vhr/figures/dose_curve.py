"""Dose curve (metric vs step) for control vs patch_only, plus a stock-normalized geometric-mean
suite score, straight from out/results/metrics.csv."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CSV = Path("out/results/metrics.csv")
OUT = Path("out/figures/dose_curve.png")
STEPS = [2000, 4000, 6000, 8000]
ARMS = {
    "sw_control": ("#888888", "control"),
    "sw_patch_only": ("#d62728", "patch anchor"),
}
# task -> (primary metric, higher-is-better, stock value); stock None -> left out of the geomean.
TASKS = {
    "flair2": ("mIoU", True, 49.51),
    "pureforest": ("mF1", True, 0.614),
    "opencanopy": ("RMSE", False, 44.13),
    "nzcattle_det": ("map_50", True, None),
}


def load() -> dict[tuple[str, str, int], list[float]]:
    values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in csv.DictReader(CSV.open()):
        task, arm = row["dataset"], row["arm"]
        if arm in ARMS and task in TASKS and row["metric"] == TASKS[task][0]:
            values[task, arm, int(row["step"])].append(float(row["value"]))
    return values


def geomean_vs_stock(values, arm: str, step: int) -> float:
    ratios = []
    for task, (_, higher, stock) in TASKS.items():
        seeds = values.get((task, arm, step))
        if stock is None or not seeds:
            continue
        mean = float(np.mean(seeds))
        ratios.append(mean / stock if higher else stock / mean)
    return float(np.exp(np.mean(np.log(ratios)))) if ratios else np.nan


def main() -> None:
    values = load()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, task in zip(axes.flat, TASKS, strict=False):
        metric, higher, _ = TASKS[task]
        for arm, (color, label) in ARMS.items():
            mean = [
                np.mean(values[task, arm, s]) if values.get((task, arm, s)) else np.nan
                for s in STEPS
            ]
            std = [
                np.std(values[task, arm, s]) if values.get((task, arm, s)) else 0.0 for s in STEPS
            ]
            ax.errorbar(STEPS, mean, yerr=std, color=color, marker="o", capsize=3, label=label)
        ax.set_title(f"{task} ({metric}, {'up' if higher else 'down'})")
        ax.set_xlabel("training step")
        ax.legend(fontsize=8)

    ax = axes.flat[len(TASKS)]
    for arm, (color, label) in ARMS.items():
        ax.plot(
            STEPS,
            [geomean_vs_stock(values, arm, s) for s in STEPS],
            color=color,
            marker="o",
            label=label,
        )
    ax.axhline(1.0, ls=":", color="k", lw=1)
    ax.set_title("geometric mean vs stock (flair2/pureforest/opencanopy)")
    ax.set_xlabel("training step")
    ax.set_ylabel("relative to stock (>1 better)")
    ax.legend(fontsize=8)
    axes.flat[-1].axis("off")

    fig.suptitle(
        "Continued pretraining dose curve: patch anchor vs control (3 seeds)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
