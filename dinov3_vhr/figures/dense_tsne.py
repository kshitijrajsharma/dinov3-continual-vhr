"""Patch-embedding t-SNE on FLAIR2, coloured by ground-truth land-cover class.

The dense-feature figure: unlike a CLS t-SNE (one point per image), this projects the
per-patch tokens and colours each by the class of the ground-truth mask under that patch.
Patches that cluster by class mean the dense representation encodes real spatial categories,
which is what segmentation reads. A weighted patch-kNN accuracy gives the same signal as a number.

    uv run --group eval python -m dinov3_vhr.figures.dense_tsne \
        --tortilla ../pangaea-bench2.0/data/flair2/geobench_flair2.tortilla \
        --checkpoint gram2=out/diag/vits_gram2.pth \
        --n-tiles 40 --out out/figures/flair2_dense_tsne.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import tacoreader.v1 as tacoreader
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE

from dinov3_vhr.backbone import ViT
from dinov3_vhr.figures._common import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    balanced_knn,
    load_encoder,
    parse_models,
    select_device,
)

RGB_BANDS = [0, 1, 2]  # FLAIR2 aerial bands are R,G,B,NIR
IGNORE = -1
FLAIR2_CLASSES = 13
_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def load_tiles(tortilla: Path, n_tiles: int, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """n FLAIR2 test tiles as a normalized RGB batch and their class masks."""
    frame = tacoreader.load(str(tortilla))
    frame = frame[frame["tortilla:data_split"] == "test"].reset_index(drop=True)
    images, masks = [], []
    for index in range(min(n_tiles, len(frame))):
        row = frame.read(index)
        with rasterio.open(row.read(0)) as src:
            aerial = torch.from_numpy(src.read()[RGB_BANDS]).float() / 255.0
        with rasterio.open(row.read(1)) as src:
            mask = torch.from_numpy(src.read(1)).long()
        aerial = F.interpolate(
            aerial.unsqueeze(0), size=size, mode="bilinear", align_corners=False
        ).squeeze(0)
        images.append((aerial - _MEAN) / _STD)
        masks.append(mask.clamp(max=FLAIR2_CLASSES) - 1)
    return torch.stack(images), torch.stack(masks)


@torch.no_grad()
def patch_tokens_and_labels(
    model: ViT, images: torch.Tensor, masks: torch.Tensor, device: str
) -> tuple[torch.Tensor, np.ndarray]:
    """Per-patch tokens and their majority-class labels across all tiles."""
    side = images.shape[-1] // 16
    tokens = []
    for start in range(0, images.shape[0], 16):
        chunk = images[start : start + 16].to(device)
        features = model.forward_features(chunk)[:, model.num_prefix_tokens :, :]
        tokens.append(features.reshape(-1, features.shape[-1]).cpu())
    grid = F.interpolate(masks.unsqueeze(1).float(), size=side, mode="nearest").squeeze(1).long()
    return torch.cat(tokens), grid.reshape(-1).numpy()


def patch_knn(tokens: torch.Tensor, labels: np.ndarray) -> float:
    """Weighted patch-kNN top-1 over the labelled patches."""
    valid = labels != IGNORE
    return balanced_knn(tokens[valid], labels[valid], FLAIR2_CLASSES)


def render(
    projections: dict[str, np.ndarray],
    labels: np.ndarray,
    accuracies: dict[str, float],
    out: Path,
) -> None:
    names = list(projections)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.2), squeeze=False)
    for c, name in enumerate(names):
        points = projections[name]
        axes[0][c].scatter(points[:, 0], points[:, 1], c=labels, cmap="tab20", s=6)
        axes[0][c].set_title(f"{name}\npatch kNN {accuracies[name]:.3f}", fontsize=11)
        axes[0][c].set_xticks([])
        axes[0][c].set_yticks([])
    fig.suptitle("FLAIR2 patch t-SNE, coloured by land-cover class", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tortilla", type=Path, required=True)
    parser.add_argument("--stock", default="vit_small_patch16_dinov3.lvd1689m")
    parser.add_argument(
        "--checkpoint", action="append", default=[], help="name=path.pth, repeatable"
    )
    parser.add_argument("--n-tiles", type=int, default=40)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--max-points", type=int, default=6000)
    parser.add_argument("--out", type=Path, default=Path("out/figures/flair2_dense_tsne.png"))
    args = parser.parse_args()

    device = select_device()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    images, masks = load_tiles(args.tortilla, args.n_tiles, args.input_size)
    print(f"loaded {images.shape[0]} tiles")

    models = parse_models("stock", args.checkpoint)
    subset: np.ndarray | None = None
    plot_labels: np.ndarray | None = None
    projections: dict[str, np.ndarray] = {}
    accuracies: dict[str, float] = {}
    for name, weights in models.items():
        model = load_encoder(args.stock, weights, device)
        tokens, labels = patch_tokens_and_labels(model, images, masks, device)
        accuracies[name] = patch_knn(tokens, labels)
        if subset is None:
            valid = np.flatnonzero(labels != IGNORE)
            subset = np.random.default_rng(0).permutation(valid)[: args.max_points]
            plot_labels = labels[subset]
        projections[name] = TSNE(
            n_components=2, perplexity=min(30, len(subset) - 1), init="pca"
        ).fit_transform(tokens[subset].numpy())
        print(f"{name}: patch kNN {accuracies[name]:.4f}")

    assert plot_labels is not None
    render(projections, plot_labels, accuracies, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
