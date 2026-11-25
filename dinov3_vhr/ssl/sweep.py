"""Expand the ViT-S structural sweep config into training jobs and evaluation combos.

Reads a sweep YAML (base recipe, arms, learning rates, seeds, schedule, eval) and turns it
into concrete ``dinov3_vhr.ssl.train`` commands and pangaea evaluation combo lines. Config
expansion is pure and testable; SLURM submission is guarded behind an explicit flag and
otherwise prints a dry run.
"""

import argparse
import itertools
import subprocess
from pathlib import Path
from typing import cast

from omegaconf import DictConfig, OmegaConf

TRAIN_MODULE = "dinov3_vhr.ssl.train"

# dataset -> (task, decoder, preprocessing, criterion) for the pangaea eval combo lines.
EVAL_TASKS: dict[str, tuple[str, str, str, str]] = {
    "flair2": ("segmentation", "seg_upernet", "seg_default", "cross_entropy"),
    "pureforest": (
        "linear_classification",
        "cls_linear",
        "cls_resize",
        "cross_entropy",
    ),
    "opencanopy": ("regression", "reg_upernet", "reg_default", "mse"),
    "nzcattle_det": ("object_detection", "det_faster_rcnn", "det_default", "none"),
}


def load_sweep(config_path: Path) -> DictConfig:
    """Parse the sweep config, failing loud if it is missing or not a mapping."""
    if not config_path.is_file():
        raise FileNotFoundError(f"sweep config not found: {config_path}")
    config = OmegaConf.load(config_path)
    if not isinstance(config, DictConfig):
        raise TypeError(f"sweep config must be a mapping, got {type(config).__name__}")
    return config


def run_name(prefix: str, arm: str, lr: float, seed: int) -> str:
    """Filesystem-safe name identifying one training run."""
    return f"{prefix}_{arm}_lr{lr:g}_s{seed}"


def train_parameters(config: DictConfig, arm: str, lr: float, seed: int) -> dict[str, object]:
    """Merge base recipe, schedule epochs, and arm overrides into one train.py parameter set."""
    params = cast(dict[str, object], OmegaConf.to_container(config.base, resolve=True))
    params["epochs"] = config.schedule.epochs
    params.update(cast(dict[str, object], OmegaConf.to_container(config.arms[arm], resolve=True)))
    name = run_name(str(config.name_prefix), arm, lr, seed)
    params.update(lr=lr, seed=seed, run_name=name, out_ckpt=f"out/{name}.pth")
    return params


def train_command(params: dict[str, object]) -> list[str]:
    """Render the train.py argv from a parameter set (key -> --key, underscores hyphenated)."""
    command = ["python", "-m", TRAIN_MODULE]
    for key, value in params.items():
        command += [f"--{key.replace('_', '-')}", str(value)]
    return command


def training_runs(config: DictConfig) -> list[tuple[str, list[str]]]:
    """Every (run_name, train argv) in the arm x learning-rate x seed grid."""
    prefix = str(config.name_prefix)
    arms = list(config.arms.keys())
    learning_rates = [float(lr) for lr in config.learning_rates]
    seeds = [int(seed) for seed in config.seeds]
    runs = []
    for arm, lr, seed in itertools.product(arms, learning_rates, seeds):
        params = train_parameters(config, arm, lr, seed)
        runs.append((run_name(prefix, arm, lr, seed), train_command(params)))
    return runs


def eval_combos(config: DictConfig) -> list[str]:
    """One pangaea combo line per (run, checkpoint step, dataset).

    Each line names an encoder config ``<run_name>_<step>`` that the eval flow points at the
    matching checkpoint. Fails loud if a configured dataset has no known task mapping.
    """
    prefix = str(config.name_prefix)
    arms = list(config.arms.keys())
    learning_rates = [float(lr) for lr in config.learning_rates]
    seeds = [int(seed) for seed in config.seeds]
    steps = [int(step) for step in config.eval.steps]
    datasets = [str(name) for name in config.eval.datasets]
    lines = []
    for arm, lr, seed in itertools.product(arms, learning_rates, seeds):
        name = run_name(prefix, arm, lr, seed)
        for step, dataset in itertools.product(steps, datasets):
            task, decoder, preprocessing, criterion = EVAL_TASKS[dataset]
            lines.append(f"{name}_{step} {dataset} {task} {decoder} {preprocessing} {criterion}")
    return lines


def submit_training(name: str, command: list[str], slurm: DictConfig) -> str:
    """Submit one training job via sbatch, returning the job id. Raises if sbatch fails."""
    sbatch = [
        "sbatch",
        "--parsable",
        f"--job-name={name}",
        f"--partition={slurm.partition}",
        f"--gres={slurm.gres}",
        f"--cpus-per-task={slurm.cpus_per_task}",
        f"--mem={slurm.mem}",
        f"--time={slurm.time}",
        f"--output=slurm/logs/{name}-%j.out",
        str(slurm.script),
        *command,
    ]
    result = subprocess.run(sbatch, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand and launch the ViT-S structural sweep.")
    parser.add_argument("--config", type=Path, default=Path("configs/sweep_vits_confirm.yaml"))
    parser.add_argument("--submit", action="store_true", help="submit to SLURM (default: dry run)")
    parser.add_argument(
        "--eval-combos", type=Path, help="write pangaea eval combo lines to this file"
    )
    args = parser.parse_args()

    config = load_sweep(args.config)
    runs = training_runs(config)
    print(f"{len(runs)} training runs from {args.config}")
    for name, command in runs:
        if args.submit:
            print(f"  submitted {submit_training(name, command, config.slurm)}  {name}")
        else:
            print(f"  {name}")

    if args.eval_combos is not None:
        combos = eval_combos(config)
        args.eval_combos.write_text("\n".join(combos) + "\n")
        print(f"wrote {len(combos)} eval combos to {args.eval_combos}")


if __name__ == "__main__":
    main()
