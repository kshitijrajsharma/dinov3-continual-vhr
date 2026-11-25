"""Dense patch-feature comparison across continued-pretraining variants.

For each backbone it renders the figure the mechanism claim rests on:

- patch-feature PCA maps: top-3 principal components of the patch tokens painted back onto the
  image grid as RGB. Crisp boundaries mean well-structured local features; a muddy map means the
  patch representation has degraded.
- CLS-to-patch cosine similarity: the DINOv3 locality metric (arXiv 2508.10104 Sec 4). As patch
  tokens drift toward the global token this rises and locality dies, so lower is better.
- a companion CLS-token t-SNE, showing global structure is preserved (the anchor acts on patches).

    uv run python -m dinov3_vhr.figures.patch_pca \
        --checkpoint gram0=out/diag/vits_gram0.pth --checkpoint gram2=out/diag/vits_gram2.pth \
        --tiles-tar data/tiles/oam/oam-000000.tar --n-images 6 --out out/figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.manifold import TSNE

from dinov3_vhr.backbone import ViT
from dinov3_vhr.figures._common import (
    load_encoder,
    load_tiles_from_tar,
    parse_models,
    pca_rgb,
    select_device,
)


@torch.no_grad()
def patch_and_cls(
    model: ViT, batch: torch.Tensor, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Patch tokens (B, P, C) and CLS token (B, C) for a normalized batch."""
    features = model.forward_features(batch.to(device))
    return features[:, model.num_prefix_tokens :, :].cpu(), features[:, 0, :].cpu()


def cls_patch_similarity(patches: torch.Tensor, cls: torch.Tensor) -> float:
    """Mean cosine similarity between the CLS token and its patch tokens (locality metric)."""
    patch_norm = F.normalize(patches.float(), dim=-1)
    cls_norm = F.normalize(cls.float(), dim=-1).unsqueeze(1)
    return float((patch_norm * cls_norm).sum(-1).mean())


def render_pca_montage(
    displays: list[Image.Image], maps: dict[str, list[np.ndarray]], out: Path
) -> None:
    names = list(maps)
    rows, cols = len(displays), len(names) + 1
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows), squeeze=False)
    for r in range(rows):
        axes[r][0].imshow(displays[r])
        axes[r][0].set_ylabel(f"tile {r}", fontsize=9)
        for c, name in enumerate(names, start=1):
            axes[r][c].imshow(np.kron(maps[name][r], np.ones((16, 16, 1))))
    for c, title in enumerate(["input", *names]):
        axes[0][c].set_title(title, fontsize=11)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Patch-feature PCA (dense structure)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def render_locality_bar(similarity: dict[str, float], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    names = list(similarity)
    ax.bar(names, [similarity[n] for n in names], color="#3b6ea5")
    ax.set_ylabel("mean CLS-patch cosine\n(lower = better locality)", fontsize=9)
    ax.set_title("Patch locality (DINOv3 metric)", fontsize=11)
    for i, name in enumerate(names):
        ax.text(
            i,
            similarity[name],
            f"{similarity[name]:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def render_cls_tsne(embeddings: dict[str, torch.Tensor], out: Path) -> None:
    names = list(embeddings)
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.4), squeeze=False)
    perplexity = min(30, embeddings[names[0]].shape[0] - 1)
    for c, name in enumerate(names):
        points = TSNE(n_components=2, perplexity=perplexity, init="pca").fit_transform(
            embeddings[name].numpy()
        )
        axes[0][c].scatter(points[:, 0], points[:, 1], s=10, c="#3b6ea5")
        axes[0][c].set_title(name, fontsize=11)
        axes[0][c].set_xticks([])
        axes[0][c].set_yticks([])
    fig.suptitle("CLS-token t-SNE (global structure, companion)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", default="vit_small_patch16_dinov3.lvd1689m")
    parser.add_argument(
        "--checkpoint", action="append", default=[], help="name=path.pth, repeatable"
    )
    parser.add_argument("--tiles-tar", type=Path, required=True)
    parser.add_argument("--n-images", type=int, default=6)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path("out/figures"))
    args = parser.parse_args()

    device = select_device()
    args.out.mkdir(parents=True, exist_ok=True)
    displays, batch = load_tiles_from_tar(args.tiles_tar, args.n_images, args.input_size)

    models = parse_models("stock", args.checkpoint)
    pca_maps: dict[str, list[np.ndarray]] = {}
    locality: dict[str, float] = {}
    cls_embeddings: dict[str, torch.Tensor] = {}
    for name, weights in models.items():
        model = load_encoder(args.stock, weights, device)
        patches, cls = patch_and_cls(model, batch, device)
        pca_maps[name] = [pca_rgb(patches[i]) for i in range(patches.shape[0])]
        locality[name] = cls_patch_similarity(patches, cls)
        cls_embeddings[name] = cls
        print(f"{name}: CLS-patch similarity {locality[name]:.4f}")

    render_pca_montage(displays, pca_maps, args.out / "patch_pca.png")
    render_locality_bar(locality, args.out / "patch_locality.png")
    render_cls_tsne(cls_embeddings, args.out / "cls_tsne.png")
    print(f"wrote figures to {args.out}")


if __name__ == "__main__":
    main()
