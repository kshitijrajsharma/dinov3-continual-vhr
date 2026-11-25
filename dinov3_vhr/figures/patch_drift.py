"""Dense patch features and where continued pretraining drifted them.

Per-model patch-PCA maps plus a drift heatmap (1 - cosine to a reference model per patch), so
the change to the dense representation is visible directly: a uniform bright drift map means the
whole representation moved, a structured one means the change concentrates on real edges.

    uv run python -m dinov3_vhr.figures.patch_drift \
        --checkpoint full=out/vits_full_epoch6.pth --checkpoint gram0=out/vits_gram0_epoch6.pth \
        --tiles-tar data/tiles/oam/oam-000000.tar --n-images 5 --out out/figures/patch_drift.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from dinov3_vhr.backbone import ViT
from dinov3_vhr.figures._common import (
    load_encoder,
    load_tiles_from_tar,
    parse_models,
    pca_rgb,
    select_device,
)


@torch.no_grad()
def patch_tokens(model: ViT, batch: torch.Tensor, device: str) -> torch.Tensor:
    """Patch tokens (B, P, C) for a normalized batch."""
    features = model.forward_features(batch.to(device))
    return features[:, model.num_prefix_tokens :, :].cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", default="vit_small_patch16_dinov3.lvd1689m")
    parser.add_argument(
        "--checkpoint", action="append", default=[], help="name=path.pth, repeatable"
    )
    parser.add_argument("--reference", default="stock", help="model name to measure drift against")
    parser.add_argument("--tiles-tar", type=Path, required=True)
    parser.add_argument("--n-images", type=int, default=5)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("out/figures/patch_drift.png"))
    args = parser.parse_args()

    device = select_device()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    displays, batch = load_tiles_from_tar(args.tiles_tar, args.n_images, args.input_size)

    models = parse_models("stock", args.checkpoint)
    if args.reference not in models:
        raise ValueError(f"reference {args.reference!r} is not one of {list(models)}")
    tokens = {
        name: patch_tokens(load_encoder(args.stock, weights, device), batch, device)
        for name, weights in models.items()
    }

    others = [name for name in models if name != args.reference]
    grid = args.input_size // 16
    rows, cols = len(displays), 1 + len(models) + len(others)
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.1 * rows), squeeze=False)
    titles = ["RGB", *models, *[f"drift {args.reference}->{o}" for o in others]]
    for j, title in enumerate(titles):
        axes[0][j].set_title(title, fontsize=9)

    heatmap = None
    for r in range(rows):
        axes[r][0].imshow(displays[r])
        for c, name in enumerate(models, start=1):
            axes[r][c].imshow(pca_rgb(tokens[name][r]))
        for c, other in enumerate(others, start=1 + len(models)):
            drift = 1 - F.cosine_similarity(tokens[args.reference][r], tokens[other][r], dim=-1)
            heatmap = axes[r][c].imshow(
                drift.numpy().reshape(grid, grid), cmap="magma", vmin=0, vmax=0.6
            )
        for c in range(cols):
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
    if heatmap is not None:
        fig.colorbar(heatmap, ax=axes[:, -1], fraction=0.046, label="1 - cosine (drift)")
    fig.suptitle("Dense patch features and drift from the reference model", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
