"""Score a continued-pretrained backbone on the pangaea VHR suite.

Emits an encoder config for the checkpoint into a pangaea checkout, then defers to that
repo's run_suite driver. The numbers therefore come from the benchmark's own decoders,
splits and test-set metrics, and stay directly comparable to the published LVD and SAT
baselines. Nothing here reimplements a probe.

The suite freezes the encoder and trains a decoder per (dataset, seed), so a delta
against a baseline is attributable to the representation rather than to the head.

    uv run --group eval python -m dinov3_vhr.eval.run_suite \
        --checkpoint out/dinov3_vitl_full.pth --name vitl_full
"""

import argparse
import subprocess
from pathlib import Path

import timm
import torch

from dinov3_vhr.backbone import as_vit

# Two dense tasks (does specialising buy geometric gains?) and two global tasks (does it
# cost semantic ones?). The tax and the gain are both measured inside the benchmark.
DEFAULT_DATASETS = ("flair2", "pureforest", "opencanopy", "nzcattle_det")

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
    path.write_text(
        ENCODER_CONFIG.format(
            model=model_name,
            weights=checkpoint.resolve(),
            input_size=input_size,
            layers=layers,
        )
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True, help="encoder name in the results table")
    parser.add_argument("--model", default="vit_large_patch16_dinov3.lvd1689m")
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--pangaea-root", type=Path, default=Path("../pangaea-bench2.0"))
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    args = parser.parse_args()

    root = args.pangaea_root.resolve()
    if not (root / "scripts" / "run_suite.py").is_file():
        raise FileNotFoundError(f"no pangaea checkout at {root}")

    config = write_encoder_config(root, args.name, args.model, args.checkpoint, args.input_size)
    print(f"wrote {config}")

    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_suite.py",
            "--encoders",
            args.name,
            "--datasets",
            *args.datasets,
        ],
        cwd=root,
        check=True,
    )


if __name__ == "__main__":
    main()
