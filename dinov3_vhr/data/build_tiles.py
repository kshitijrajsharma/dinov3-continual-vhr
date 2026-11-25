"""Build standard 256 x 256 WebDataset tile shards from a pretraining source.

Each source has an adapter that yields SourceImage records; this drives the shared
tiler and shard writer and records a manifest with the tile and shard counts.
"""

import argparse
import json
from collections.abc import Callable, Iterator
from pathlib import Path

from dinov3_vhr.data.sources import loveda, ramp, worldstrat
from dinov3_vhr.data.tiles import (
    SHARD_SIZE,
    TILE_PX,
    SourceImage,
    iter_tiles,
    write_shards,
)

SOURCES: dict[str, Callable[[Path], Iterator[SourceImage]]] = {
    "loveda": loveda.iter_images,
    "ramp": ramp.iter_images,
    "worldstrat": worldstrat.iter_images,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=sorted(SOURCES))
    parser.add_argument("--in-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tile-px", type=int, default=TILE_PX)
    parser.add_argument("--min-valid", type=float, default=0.7)
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    args = parser.parse_args()

    images = SOURCES[args.source](args.in_root)
    tiles = (tile for image in images for tile in iter_tiles(image, args.tile_px, args.min_valid))
    manifest = write_shards(tiles, args.out, args.source, args.shard_size)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"DONE: {manifest['n_tiles']} tiles in {manifest['n_shards']} shards -> {args.out}")


if __name__ == "__main__":
    main()
