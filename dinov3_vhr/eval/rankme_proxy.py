"""RankMe (Garrido et al., ICML 2023, arXiv:2210.02885): unsupervised proxy for representation
quality, needing no labels, so an in-domain stand-in for downstream score during an LR search.

    uv run python -m dinov3_vhr.eval.rankme_proxy --tiles-dir data/tiles --n-tiles 2000 \
        --checkpoints stock out/vitl_gram0_s42_epoch1.pth
"""

import argparse
from pathlib import Path

import torch
from torch import Tensor
from torchvision import transforms
from webdataset.compat import WebDataset
from webdataset.mix import RandomMix

from dinov3_vhr.backbone import ViT, as_vit, build_backbone
from dinov3_vhr.data.tile_loader import (
    _keep_all_shards,
    source_shards,
    source_tile_counts,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def rankme(features: Tensor, eps: float = 1e-7) -> float:
    """Effective rank: exp-entropy of the normalized singular value spectrum (Garrido et al.)."""
    singular_values = torch.linalg.svdvals(features.float())
    p = singular_values / (singular_values.sum() + eps) + eps
    return float(torch.exp(-(p * p.log()).sum()))


def load_tiles(tiles_dir: Path, n_tiles: int, image_size: int = 256, seed: int = 0) -> Tensor:
    """A stratified tile sample, resized and ImageNet-normalized: RandomMix across sources
    weighted by each source's true tile count, the training loader's stratification. The sample
    is deterministic (fixed seed, no resampling) so the same tiles score every checkpoint."""
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    sources = source_shards(tiles_dir)
    weights = source_tile_counts(tiles_dir, list(sources))
    streams = [
        WebDataset(shards, shardshuffle=False, nodesplitter=_keep_all_shards)
        .shuffle(2000, seed=seed + index)
        .decode("pil")
        .to_tuple("png")
        .map_tuple(transform)
        for index, shards in enumerate(sources.values())
    ]
    mixed = RandomMix(streams, weights)  # ty: ignore[invalid-argument-type]
    images = []
    for (image,) in mixed:
        images.append(image)
        if len(images) >= n_tiles:
            break
    return torch.stack(images)


@torch.no_grad()
def extract_features(
    vit: ViT,
    images: Tensor,
    device: str,
    batch_size: int = 32,
    max_patch_rows: int = 20000,
) -> tuple[Tensor, Tensor]:
    """Class token (global, semantic tasks) and patch tokens (dense tasks) as separate signals;
    patch tokens are strided down to keep the singular value decomposition tractable."""
    cls_features, patch_features = [], []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)
        features = vit.forward_features(batch)
        cls_features.append(features[:, 0, :].cpu())
        patches = features[:, vit.num_prefix_tokens :, :]
        patch_features.append(patches.reshape(-1, patches.shape[-1]).cpu())
    cls = torch.cat(cls_features)
    patches = torch.cat(patch_features)
    stride = max(1, len(patches) // max_patch_rows)
    return cls, patches[::stride][:max_patch_rows]


def load_backbone(name: str, model_name: str, device: str) -> ViT:
    """Backbone for a checkpoint path, or the released weights when name == 'stock'."""
    if name == "stock":
        backbone = build_backbone(model_name, pretrained=True)
    else:
        backbone = build_backbone(model_name, pretrained=False)
        state = torch.load(name, map_location="cpu", weights_only=True)
        backbone.load_state_dict(state)
    backbone.eval().to(device)
    return as_vit(backbone)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles-dir", type=Path, default=Path("data/tiles"))
    parser.add_argument("--n-tiles", type=int, default=2000)
    parser.add_argument("--model", default="vit_large_patch16_dinov3.lvd1689m")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    images = load_tiles(args.tiles_dir, args.n_tiles)
    print(f"loaded {len(images)} tiles from {args.tiles_dir}")
    print(f"{'checkpoint':<48} {'rankme_cls':>12} {'rankme_patch':>14}")
    for name in args.checkpoints:
        vit = load_backbone(name, args.model, device)
        cls_features, patch_features = extract_features(vit, images, device)
        print(
            f"{Path(name).name:<48} {rankme(cls_features):>12.2f} {rankme(patch_features):>14.2f}"
        )


if __name__ == "__main__":
    main()
