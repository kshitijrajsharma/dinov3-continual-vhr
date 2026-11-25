"""Score a continued-pretrained backbone on the pangaea VHR suite.

Writes an encoder config for the checkpoint into a pangaea source tree, then runs the
benchmark's own hydra entrypoint (``pangaea/run.py``) once per (dataset, seed). The numbers
come from the benchmark's decoders, splits and test-set metrics, so they stay directly
comparable to the published LVD and SAT baselines. Nothing here reimplements a probe.

The encoder is frozen and a decoder is trained per (dataset, seed), so a delta against a
baseline is attributable to the representation rather than to the head. Collect the
per-dataset metrics into a CSV afterwards with ``dinov3_vhr.eval.collect_results``.

    uv run --group eval python -m dinov3_vhr.eval.run_suite \
        --checkpoint out/vitl_gram0_s42_epoch1.pth --name vitl_gram0

The eval group installs the pinned pangaea package and its dependencies, which is the
environment the suite runs in. The wheel does not ship the hydra ``configs/`` tree, so the
pinned source is fetched once (or taken from ``--pangaea-root``) to supply it; the benchmark
datasets download themselves on first use.
"""

import argparse
import json
import os
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import timm
import torch

from dinov3_vhr.backbone import as_vit

# Two dense tasks (does specialising buy geometric gains?) and two global tasks (does it
# cost semantic ones?). The tax and the gain are both measured inside the benchmark.
DEFAULT_DATASETS = ("flair2", "pureforest", "opencanopy", "nzcattle_det")

# dataset -> (task, decoder, preprocessing, criterion). These names resolve against the
# pangaea source's configs/{task,decoder,preprocessing,criterion} groups and are the suite's
# canonical settings (scripts/suite.yaml in the benchmark repo).
DATASET_RECIPES: dict[str, tuple[str, str, str, str]] = {
    "spacenet2": ("segmentation", "seg_upernet", "seg_default", "cross_entropy"),
    "spacenet3": ("segmentation", "seg_upernet", "seg_default", "cross_entropy"),
    "flair2": ("segmentation", "seg_upernet", "seg_default", "cross_entropy"),
    "hot_buildings": ("segmentation", "seg_upernet", "seg_default", "cross_entropy"),
    "xview2": ("change_detection", "seg_siamupernet_conc", "seg_default", "cross_entropy"),
    "pureforest": ("linear_classification", "cls_linear", "cls_resize", "cross_entropy"),
    "mpv4ger": ("linear_classification", "cls_linear", "cls_resize", "cross_entropy"),
    "opencanopy": ("regression", "reg_upernet", "reg_default", "mse"),
    "everwatch": ("object_detection", "det_faster_rcnn", "det_default", "none"),
    "nzcattle_det": ("object_detection", "det_faster_rcnn", "det_default", "none"),
}

ENCODER_CONFIG = """\
_target_: pangaea.encoders.timm_encoder.TimmEncoder
model_name: {model}
encoder_weights: {weights}
encoder_type: vit
input_size: {input_size}
input_bands:
  optical:
    - B4
    - B3
    - B2
output_layers:
{layers}
"""


def evenly_spaced_layers(depth: int, count: int = 4) -> list[int]:
    """The `count` block indices pangaea taps for its pyramid, spread across the depth."""
    return [round(depth * (i + 1) / count) - 1 for i in range(count)]


def validate_checkpoint(model_name: str, checkpoint: Path) -> int:
    """Load the checkpoint into the architecture and return its block count.

    Runs before any GPU time is spent so an architecture mismatch fails here rather than
    hours into a decoder training run.
    """
    backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    backbone.load_state_dict(state)
    return len(as_vit(backbone).blocks)


def write_encoder_config(
    pangaea_root: Path, name: str, model_name: str, checkpoint: Path, input_size: int
) -> Path:
    depth = validate_checkpoint(model_name, checkpoint)
    layers = "\n".join(f"  - {i}" for i in evenly_spaced_layers(depth))
    path = pangaea_root / "configs" / "encoder" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ENCODER_CONFIG.format(
            model=model_name,
            weights=checkpoint.resolve(),
            input_size=input_size,
            layers=layers,
        )
    )
    return path


def pinned_pangaea_source() -> tuple[str, str]:
    """Repo URL and commit of the installed pangaea, from its PEP 610 ``direct_url.json``.

    The eval group installs pangaea from the pinned git source, so the fetched tree matches
    the installed package and the commit cannot drift from pyproject.
    """
    try:
        meta = distribution("pangaea")
    except PackageNotFoundError as exc:
        raise SystemExit("pangaea is not installed; run `just setup-eval`") from exc
    direct_url = meta.read_text("direct_url.json")
    if direct_url is None:
        raise SystemExit("pangaea has no direct_url.json; pass --pangaea-root to a checkout")
    info = json.loads(direct_url)
    commit = info.get("vcs_info", {}).get("commit_id")
    if not commit:
        raise SystemExit("pangaea was not installed from git; pass --pangaea-root to a checkout")
    return info["url"], commit


def fetch_pangaea_source(repo: str, commit: str) -> Path:
    """Clone the pinned pangaea source into a cache so its hydra configs sit on disk; the
    wheel ships the code but not the configs. Reused across runs, keyed by commit."""
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "dinov3-vhr"
    source = cache / f"pangaea-{commit[:12]}"
    if (source / "pangaea" / "run.py").is_file():
        return source
    if source.exists():
        shutil.rmtree(source)
    source.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--filter=blob:none", repo, str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "checkout", commit], check=True)
    return source


def resolve_pangaea_source(pangaea_root: Path | None) -> Path:
    """The pangaea source tree to run from: an explicit checkout, an adjacent one from the
    original layout, or the pinned source fetched on demand."""
    if pangaea_root is not None:
        root = pangaea_root.resolve()
        if not (root / "pangaea" / "run.py").is_file():
            raise FileNotFoundError(f"no pangaea checkout at {root} (expected pangaea/run.py)")
        return root
    adjacent = Path("../pangaea-bench2.0").resolve()
    if (adjacent / "pangaea" / "run.py").is_file():
        return adjacent
    return fetch_pangaea_source(*pinned_pangaea_source())


def eval_command(
    name: str, dataset: str, seed: int, limited_label: float, fast_dev: bool
) -> list[str]:
    """One frozen-encoder, trained-decoder pangaea run on a single (dataset, seed)."""
    task, decoder, preprocessing, criterion = DATASET_RECIPES[dataset]
    overrides = [
        f"dataset={dataset}",
        f"encoder={name}",
        f"decoder={decoder}",
        f"preprocessing={preprocessing}",
        f"criterion={criterion}",
        f"task={task}",
        f"seed={seed}",
        "use_wandb=false",
        "finetune=false",
        f"limited_label_train={0.01 if fast_dev else limited_label}",
        f"batch_size={2 if fast_dev else 8}",
        f"test_batch_size={1 if fast_dev else 8}",
    ]
    if fast_dev:
        overrides.append("task.trainer.n_epochs=1")
    return [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=1",
        "pangaea/run.py",
        "--config-name=train",
        *overrides,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True, help="encoder name in the results table")
    parser.add_argument("--model", default="vit_large_patch16_dinov3.lvd1689m")
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument(
        "--pangaea-root",
        type=Path,
        default=None,
        help="a pangaea checkout to run from; the pinned source is fetched when omitted",
    )
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 1234, 2025])
    parser.add_argument("--limited-label", type=float, default=0.1, help="train-label fraction")
    parser.add_argument(
        "--fast-dev",
        action="store_true",
        help="one epoch, one seed, a tiny label fraction: a wiring smoke test, not a real score",
    )
    args = parser.parse_args()

    unknown = [d for d in args.datasets if d not in DATASET_RECIPES]
    if unknown:
        raise SystemExit(f"no recipe for {unknown}; known datasets: {sorted(DATASET_RECIPES)}")

    root = resolve_pangaea_source(args.pangaea_root)
    print(f"pangaea source: {root}")

    config = write_encoder_config(root, args.name, args.model, args.checkpoint, args.input_size)
    print(f"wrote {config}")

    pythonpath = os.pathsep.join(p for p in [str(root), os.environ.get("PYTHONPATH", "")] if p)
    env = {**os.environ, "PYTHONPATH": pythonpath}
    seeds = args.seeds[:1] if args.fast_dev else args.seeds
    for dataset in args.datasets:
        for seed in seeds:
            cmd = eval_command(args.name, dataset, seed, args.limited_label, args.fast_dev)
            print(f"[{dataset} seed={seed}] {' '.join(cmd)}")
            subprocess.run(cmd, cwd=root, env=env, check=True)

    print(
        f"done: {len(args.datasets)} dataset(s) x {len(seeds)} seed(s). "
        "Collect metrics with: uv run python -m dinov3_vhr.eval.collect_results"
    )


if __name__ == "__main__":
    main()
