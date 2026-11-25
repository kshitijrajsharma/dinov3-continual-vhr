"""Feature-space diagnostics: does continued pretraining actually move the backbone?

Compares a continued checkpoint against stock DINOv3-LVD on a fixed multi-source VHR
tile sample, at the four layers the pangaea probes read (output_layers [5,11,17,23]).
Emits three figures: per-layer CKA drift and effective rank, a 2D CLS embedding by
source, and dense patch-PCA maps.

    uv run python -m dinov3_vhr.figures.feature_drift \
        --continued out/vitl_gram0_s42_epoch1.pth --n-tiles 1200
"""

import argparse
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import Tensor
from torchvision import transforms
from webdataset.compat import WebDataset

from dinov3_vhr.backbone import ViT, as_vit, build_backbone
from dinov3_vhr.eval.rankme_proxy import rankme
from dinov3_vhr.monitor.drift import linear_cka

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PROBE_LAYERS = [5, 11, 17, 23]


def sample_tiles(
    tiles_dir: Path, per_source: int, image_size: int = 256, seed: int = 0
) -> tuple[Tensor, Tensor, list[str]]:
    """Equal tiles per source. Returns model input (normalized), display tensor (0..1),
    and the per-tile source label, in a fixed order so both models see identical inputs."""
    to_display = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    sources = sorted(p.name for p in tiles_dir.iterdir() if p.is_dir())
    display, labels = [], []
    for index, source in enumerate(sources):
        shards = sorted(str(p) for p in (tiles_dir / source).glob("*.tar"))
        if not shards:
            continue
        stream = (
            WebDataset(shards, shardshuffle=False)
            .shuffle(1000, seed=seed + index)
            .decode("pil")
            .to_tuple("png")
            .map_tuple(to_display)
        )
        taken = 0
        for (image,) in stream:  # ty: ignore[not-iterable]
            display.append(image)
            labels.append(source)
            taken += 1
            if taken >= per_source:
                break
    display_batch = torch.stack(display)
    model_batch = torch.stack([normalize(image) for image in display])
    return model_batch, display_batch, labels


def load_model(name: str, model_name: str, device: str) -> ViT:
    if name == "stock":
        backbone = build_backbone(model_name, pretrained=True)
    else:
        backbone = build_backbone(model_name, pretrained=False)
        backbone.load_state_dict(torch.load(name, map_location="cpu", weights_only=True))
    return as_vit(backbone.eval().to(device))


@torch.no_grad()
def layer_features(
    vit: ViT, images: Tensor, device: str, layers: list[int], batch: int = 16
) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
    """CLS token and mean-pooled patch tokens per requested block, one row per tile."""
    store: dict[int, Tensor] = {}
    handles = [
        cast(torch.nn.Module, vit.blocks[i]).register_forward_hook(
            lambda _m, _inp, out, i=i: store.__setitem__(i, out.detach())
        )
        for i in layers
    ]
    prefix = vit.num_prefix_tokens
    cls: dict[int, list[Tensor]] = {i: [] for i in layers}
    patch: dict[int, list[Tensor]] = {i: [] for i in layers}
    for start in range(0, len(images), batch):
        vit.forward_features(images[start : start + batch].to(device))
        for i in layers:
            tokens = store[i]
            cls[i].append(tokens[:, 0, :].cpu())
            patch[i].append(tokens[:, prefix:, :].mean(dim=1).cpu())
    for handle in handles:
        handle.remove()
    return (
        {i: torch.cat(cls[i]) for i in layers},
        {i: torch.cat(patch[i]) for i in layers},
    )


@torch.no_grad()
def patch_grid(vit: ViT, images: Tensor, device: str, layer: int) -> Tensor:
    """Full patch-token grid at one block for a small set of tiles: (n, tokens, dim)."""
    store: dict[int, Tensor] = {}
    handle = cast(torch.nn.Module, vit.blocks[layer]).register_forward_hook(
        lambda _m, _inp, out: store.__setitem__(layer, out.detach())
    )
    vit.forward_features(images.to(device))
    handle.remove()
    return store[layer][:, vit.num_prefix_tokens :, :].cpu()


def figure_cka_rank(
    cls_a: dict[int, Tensor],
    patch_a: dict[int, Tensor],
    cls_b: dict[int, Tensor],
    patch_b: dict[int, Tensor],
    layers: list[int],
    out_path: Path,
    title: str,
) -> tuple[list[float], list[float]]:
    cka_cls = [linear_cka(cls_a[i], cls_b[i]) for i in layers]
    cka_patch = [linear_cka(patch_a[i], patch_b[i]) for i in layers]
    rank_cls = ([rankme(cls_a[i]) for i in layers], [rankme(cls_b[i]) for i in layers])
    rank_patch = (
        [rankme(patch_a[i]) for i in layers],
        [rankme(patch_b[i]) for i in layers],
    )
    xs = list(range(len(layers)))
    ticks = [f"L{i}" for i in layers]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    axes[0].plot(xs, cka_cls, "o-", label="CLS", color="#1C857D")
    axes[0].plot(xs, cka_patch, "s--", label="patch (mean)", color="#B0771C")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Representation drift vs stock\n(linear CKA, 1.0 = unchanged)")
    axes[0].set_xlabel("probe layer")
    axes[0].legend()
    for x, value in zip(xs, cka_cls, strict=True):
        axes[0].annotate(f"{value:.3f}", (x, value), fontsize=8, ha="center", va="bottom")

    for ax, (title_r, (ra, rb)) in zip(
        axes[1:],
        [("CLS effective rank", rank_cls), ("Patch effective rank", rank_patch)],
        strict=True,
    ):
        width = 0.38
        ax.bar([x - width / 2 for x in xs], ra, width, label="stock", color="#9CA9A6")
        ax.bar([x + width / 2 for x in xs], rb, width, label="continued", color="#1C857D")
        ax.set_title(title_r)
        ax.set_xlabel("probe layer")
        ax.legend()

    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(ticks)
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return cka_cls, cka_patch


def figure_embedding(
    cls_stock: Tensor, cls_cont: Tensor, labels: list[str], out_path: Path, layer: int
) -> None:
    names = sorted(set(labels))
    color_of = {n: plt.cm.tab10(index % 10) for index, n in enumerate(names)}
    colors = [color_of[label] for label in labels]
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for row, (name, feats) in enumerate([("stock", cls_stock), ("continued", cls_cont)]):
        standardized = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
        pca = PCA(n_components=2).fit_transform(standardized.numpy())
        tsne = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(
            standardized.numpy()
        )
        for ax, coords, kind in [
            (axes[row, 0], pca, "PCA"),
            (axes[row, 1], tsne, "t-SNE"),
        ]:
            ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=8, alpha=0.7)
            ax.set_title(f"{name} · {kind} · CLS L{layer}")
            ax.set_xticks([])
            ax.set_yticks([])
    handles = [plt.Line2D([], [], marker="o", ls="", color=color_of[n], label=n) for n in names]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(names),
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("CLS embedding geometry, colored by source", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def patch_pca_rgb(patches: Tensor, grid: int) -> Tensor:
    """3-component PCA of a tile's patch tokens, min-max scaled to an RGB grid image."""
    reduced = PCA(n_components=3).fit_transform(patches.float().numpy())
    reduced = torch.from_numpy(reduced)
    low = reduced.amin(0)
    reduced = (reduced - low) / (reduced.amax(0) - low + 1e-6)
    return reduced.reshape(grid, grid, 3)


def figure_patch_pca(
    display: Tensor, grids_stock: Tensor, grids_cont: Tensor, out_path: Path, layer: int
) -> None:
    count = len(display)
    grid = int(grids_stock.shape[1] ** 0.5)
    fig, axes = plt.subplots(3, count, figsize=(2.5 * count, 7.6))
    for col in range(count):
        axes[0, col].imshow(display[col].permute(1, 2, 0).numpy())
        axes[1, col].imshow(patch_pca_rgb(grids_stock[col], grid).numpy())
        axes[2, col].imshow(patch_pca_rgb(grids_cont[col], grid).numpy())
        for row in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    for row, name in enumerate(["tile", f"stock patch-PCA L{layer}", f"continued L{layer}"]):
        axes[row, 0].set_ylabel(name, fontsize=11)
    fig.suptitle("Dense patch structure (3-comp PCA to RGB)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles-dir", type=Path, default=Path("data/tiles"))
    parser.add_argument("--continued", required=True)
    parser.add_argument("--model", default="vit_large_patch16_dinov3.lvd1689m")
    parser.add_argument("--n-tiles", type=int, default=1200)
    parser.add_argument("--out-dir", type=Path, default=Path("results/figs"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(p.name for p in args.tiles_dir.iterdir() if p.is_dir())
    per_source = max(1, args.n_tiles // max(1, len(sources)))
    model_batch, display_batch, labels = sample_tiles(args.tiles_dir, per_source)
    print(f"sampled {len(labels)} tiles across {len(sources)} sources: {sources}")

    stock = load_model("stock", args.model, device)
    cls_stock, patch_stock = layer_features(stock, model_batch, device, PROBE_LAYERS)
    grids_stock = patch_grid(stock, model_batch[:6], device, PROBE_LAYERS[-1])
    del stock
    torch.cuda.empty_cache()

    cont = load_model(args.continued, args.model, device)
    cls_cont, patch_cont = layer_features(cont, model_batch, device, PROBE_LAYERS)
    grids_cont = patch_grid(cont, model_batch[:6], device, PROBE_LAYERS[-1])
    del cont
    torch.cuda.empty_cache()

    tag = Path(args.continued).stem
    cka_cls, cka_patch = figure_cka_rank(
        cls_stock,
        patch_stock,
        cls_cont,
        patch_cont,
        PROBE_LAYERS,
        args.out_dir / f"drift_cka_rank_{tag}.png",
        f"stock vs {tag}",
    )
    figure_embedding(
        cls_stock[PROBE_LAYERS[-1]],
        cls_cont[PROBE_LAYERS[-1]],
        labels,
        args.out_dir / f"drift_embedding_{tag}.png",
        PROBE_LAYERS[-1],
    )
    figure_patch_pca(
        display_batch[:6],
        grids_stock,
        grids_cont,
        args.out_dir / f"drift_patchpca_{tag}.png",
        PROBE_LAYERS[-1],
    )
    print("CKA vs stock per layer (1.0 = unchanged):")
    for layer, cvalue, pvalue in zip(PROBE_LAYERS, cka_cls, cka_patch, strict=True):
        print(f"  L{layer}: CLS {cvalue:.4f}   patch {pvalue:.4f}")
    print(f"figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
