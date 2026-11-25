"""Side-by-side CLS t-SNE colored by source: stock DINOv3 vs continued checkpoints.

Shows whether the latent space reorganizes per config on one shared tile sample.

    uv run python -m dinov3_vhr.figures.embedding_compare \
        --checkpoints out/solid_winner_ld10_epoch1.pth out/solid_gram0_ld10_epoch1.pth \
        --names "winner+1.0 (anchor)" "control (no anchor)"
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from torch import Tensor

from dinov3_vhr.figures.feature_drift import ViT, load_model, sample_tiles


@torch.no_grad()
def cls_features(vit: ViT, images: Tensor, device: str, batch: int = 32) -> Tensor:
    """Final-layer CLS token per tile."""
    out = []
    for start in range(0, len(images), batch):
        features = vit.forward_features(images[start : start + batch].to(device))
        out.append(features[:, 0, :].cpu())
    return torch.cat(out)


def tsne_2d(features: Tensor) -> Tensor:
    standardized = (features - features.mean(0)) / (features.std(0) + 1e-6)
    coords = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(
        standardized.numpy()
    )
    return torch.from_numpy(coords)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles-dir", type=Path, default=Path("data/tiles"))
    parser.add_argument("--model", default="vit_large_patch16_dinov3.lvd1689m")
    parser.add_argument("--n-tiles", type=int, default=900)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("results/figs/embedding_compare.png"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    sources = sorted(p.name for p in args.tiles_dir.iterdir() if p.is_dir())
    per_source = max(1, args.n_tiles // max(1, len(sources)))
    model_batch, _, labels = sample_tiles(args.tiles_dir, per_source)
    names = sorted(set(labels))
    color_of = {n: plt.cm.tab10(i % 10) for i, n in enumerate(names)}
    colors = [color_of[label] for label in labels]
    print(f"sampled {len(labels)} tiles across {len(names)} sources")

    panels = [("stock DINOv3-LVD", "stock")] + list(zip(args.names, args.checkpoints, strict=True))
    cols = len(panels)
    fig, axes = plt.subplots(1, cols, figsize=(4.2 * cols, 4.6))
    for ax, (title, ckpt) in zip(axes, panels, strict=True):
        vit = load_model(ckpt, args.model, device)
        coords = tsne_2d(cls_features(vit, model_batch, device))
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=10, alpha=0.75)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        del vit
        torch.cuda.empty_cache()

    handles = [plt.Line2D([], [], marker="o", ls="", color=color_of[n], label=n) for n in names]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(names),
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        "CLS latent space by source: does continued pretraining reorganize it?",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
