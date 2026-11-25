"""Weighted multi-source WebDataset loader for the VHR tile shards.

Each source is an independent, resampled (infinite) stream of decoded RGB tiles run
through the DINOv3 multi-crop transform. RandomMix draws across sources with per-source
weights that default to each source's true tile count, so sampling is stratified: every
source is covered at the same fraction of its own tiles per epoch instead of the small
sources being recycled while the large ones are starved.
"""

import json
import os
from collections.abc import Callable
from pathlib import Path

from PIL.Image import Image
from torch import Tensor
from webdataset.compat import WebDataset, WebLoader
from webdataset.mix import RandomMix

from dinov3_vhr.dino_ssl import multicrop_collate

Transform = Callable[[Image], list[Tensor] | list[Image] | tuple[list[Tensor], Tensor]]


def distributed_rank_world() -> tuple[int, int]:
    """Global rank and world size from the launcher env, defaulting to single process."""
    for rank_key, world_key in (
        ("SLURM_PROCID", "SLURM_NTASKS"),
        ("RANK", "WORLD_SIZE"),
    ):
        if rank_key in os.environ and world_key in os.environ:
            return int(os.environ[rank_key]), int(os.environ[world_key])
    return 0, 1


def source_shards(tiles_dir: Path) -> dict[str, list[str]]:
    """Map each source subfolder to its shard paths, partitioned across DDP ranks.

    Each rank keeps a disjoint stride of a source's shards so two GPUs never stream the
    same tiles. Sources with fewer shards than the world size are left whole rather than
    starving a rank.
    """
    rank, world = distributed_rank_world()
    sources: dict[str, list[str]] = {}
    for folder in sorted(
        p for p in tiles_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    ):
        shards = sorted(str(p) for p in folder.glob("*.tar"))
        if len(shards) >= world:
            shards = shards[rank::world]
        if shards:
            sources[folder.name] = shards
    if not sources:
        raise FileNotFoundError(f"no source/*.tar shards under {tiles_dir}")
    return sources


def _keep_all_shards(urls: list[str]) -> list[str]:
    """No-op node splitter: source_shards already gives each DDP rank disjoint shards,
    so webdataset must not split again. This also satisfies its distributed check."""
    return urls


def source_tile_counts(tiles_dir: Path, names: list[str]) -> list[float]:
    """True tile count per source, read from each source's manifest.json.

    Used as the default RandomMix weights so draws are proportional to source size
    (stratified). A source is weighted by its full tile count, independent of the
    per-rank shard split, so the mix stays proportional under DDP.
    """
    counts = []
    for name in names:
        manifest = json.loads((tiles_dir / name / "manifest.json").read_text())
        counts.append(float(manifest["n_tiles"]))
    return counts


def build_tile_loader(
    tiles_dir: Path,
    transform: Transform,
    batch_size: int,
    steps_per_epoch: int,
    weights: dict[str, float] | None = None,
    num_workers: int = 8,
    shuffle_buffer: int = 2000,
    seed: int = 0,
    full_coverage: bool = False,
) -> tuple[WebLoader, list[str]]:
    """WebLoader over all sources, mixed by per-source weight.

    Weights default to each source's true tile count (stratified sampling); pass an
    explicit ``weights`` mapping to override (e.g. equal weights for an ablation). Each
    tile carries its source index as the sample label so a coverage callback can verify
    every sensor is drawn. The per-stream shuffle is seeded for reproducibility. Returns
    the loader and the ordered source names the labels index into.

    ``full_coverage`` switches the streams from resampled (infinite, with replacement) to
    finite; RandomMix drains every source once (``longest``), so one epoch sees each unique
    tile about once instead of ~63% under resampling.
    """
    sources = source_shards(tiles_dir)
    names = list(sources)
    if weights is None:
        probabilities = source_tile_counts(tiles_dir, names)
    else:
        probabilities = [weights[name] for name in names]
    streams = [
        WebDataset(
            shards,
            shardshuffle=True,
            resampled=not full_coverage,
            nodesplitter=_keep_all_shards,
        )
        .shuffle(shuffle_buffer, seed=seed + index)
        .decode("pil")
        .to_tuple("png", "json")
        .map_tuple(transform, lambda _meta, source=index: source)
        for index, shards in enumerate(sources.values())
    ]
    mixed = RandomMix(streams, probabilities, longest=full_coverage)  # ty: ignore[invalid-argument-type]
    loader = WebLoader(
        mixed,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=multicrop_collate,
        drop_last=True,
    )
    return loader.with_epoch(steps_per_epoch), names  # ty: ignore[invalid-return-type]
