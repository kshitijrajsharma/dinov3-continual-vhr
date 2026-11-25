"""CLS-embedding t-SNE on PureForest, coloured by tree species.

The global representation-quality figure: run a balanced sample of PureForest tiles through
each backbone, take the CLS token, and project to 2D. Points that cluster by species mean the
global representation encodes real categories. A weighted kNN accuracy per model gives the
same signal as a number.

Reads the parquet mirror directly (image bytes are 4-channel PNGs ordered NIR,R,G,B).

    uv run python -m dinov3_vhr.figures.cls_tsne \
        --parquet-dir ../pangaea-bench2.0/data/pureforest/pureforest \
        --checkpoint gram2=out/diag/vits_gram2.pth \
        --per-class 40 --out out/figures/species_tsne.png
"""

import argparse
import io
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from sklearn.manifold import TSNE
from torchvision import transforms

from dinov3_vhr.backbone import ViT
from dinov3_vhr.figures._common import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    balanced_knn,
    load_encoder,
    parse_models,
    select_device,
)

RGB_CHANNELS = (1, 2, 3)  # parquet PNG bands are NIR,R,G,B; take R,G,B


def load_balanced(parquet_dir: Path, per_class: int, size: int) -> tuple[torch.Tensor, np.ndarray]:
    """Up to per_class tiles per species from the parquet shards as a normalized batch."""
    to_tensor = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    counts: dict[int, int] = defaultdict(int)
    tensors, labels = [], []
    for shard in sorted(parquet_dir.glob("*.parquet")):
        table = pq.read_table(shard, columns=["image", "label"])
        for cell, raw_label in zip(table["image"], table["label"], strict=True):
            label = int(raw_label.as_py())
            if counts[label] >= per_class:
                continue
            payload = cell.as_py()
            data = payload["bytes"] if isinstance(payload, dict) else payload
            channels = np.array(Image.open(io.BytesIO(data)))
            tensors.append(to_tensor(Image.fromarray(channels[:, :, RGB_CHANNELS])))
            labels.append(label)
            counts[label] += 1
    if not tensors:
        raise FileNotFoundError(f"no tiles read from {parquet_dir}")
    return torch.stack(tensors), np.array(labels)


@torch.no_grad()
def cls_embeddings(model: ViT, batch: torch.Tensor, device: str) -> torch.Tensor:
    """CLS token for each image, extracted in chunks to bound memory."""
    out = []
    for start in range(0, batch.shape[0], 64):
        chunk = batch[start : start + 64].to(device)
        out.append(model.forward_features(chunk)[:, 0, :].cpu())
    return torch.cat(out)


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
        axes[0][c].scatter(points[:, 0], points[:, 1], c=labels, cmap="tab20", s=14)
        axes[0][c].set_title(f"{name}\nspecies kNN {accuracies[name]:.3f}", fontsize=11)
        axes[0][c].set_xticks([])
        axes[0][c].set_yticks([])
    fig.suptitle("PureForest CLS t-SNE, coloured by tree species", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--stock", default="vit_small_patch16_dinov3.lvd1689m")
    parser.add_argument(
        "--checkpoint", action="append", default=[], help="name=path.pth, repeatable"
    )
    parser.add_argument("--per-class", type=int, default=40)
    parser.add_argument("--classes", type=int, default=13)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("out/figures/species_tsne.png"))
    args = parser.parse_args()

    device = select_device()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    batch, labels = load_balanced(args.parquet_dir, args.per_class, args.input_size)
    print(f"loaded {len(labels)} tiles across {len(np.unique(labels))} species")

    models = parse_models("Released DINOv3", args.checkpoint)
    perplexity = min(30, len(labels) - 1)
    projections: dict[str, np.ndarray] = {}
    accuracies: dict[str, float] = {}
    for name, weights in models.items():
        model = load_encoder(args.stock, weights, device)
        embeddings = cls_embeddings(model, batch, device)
        accuracies[name] = balanced_knn(embeddings, labels, args.classes)
        projections[name] = TSNE(
            n_components=2, perplexity=perplexity, init="pca", random_state=0
        ).fit_transform(embeddings.numpy())
        print(f"{name}: species kNN {accuracies[name]:.4f}")

    render(projections, labels, accuracies, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
