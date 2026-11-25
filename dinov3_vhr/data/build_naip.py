"""Stream SatlasPretrain NAIP (allenai/s2-naip) into standard tile shards, capped.

The NAIP imagery is stored as WebDataset tars of 512 x 512 RGBA PNG chips, one set per
UTM zone across the continental US. This samples tars across zones for geographic spread,
splits each chip into 256 tiles, and writes shards. Tars download to a temp dir and are
deleted after tiling so raw never accumulates.
"""

import argparse
import io
import itertools
import json
import tarfile
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

from dinov3_vhr.data.tiles import (
    SHARD_SIZE,
    TILE_PX,
    SourceImage,
    Tile,
    iter_tiles,
    write_shards,
)

REPO = "allenai/s2-naip"
LICENSE = "NAIP-public-domain"
GSD_M = 1.25


def list_zone_tars(api: HfApi) -> dict[str, list[str]]:
    """Group the NAIP tar keys by UTM zone."""
    by_zone: dict[str, list[str]] = {}
    for path in api.list_repo_files(REPO, repo_type="dataset"):
        if path.startswith("naip_") and path.endswith(".tar"):
            by_zone.setdefault(path.split("/")[0], []).append(path)
    return {zone: sorted(tars) for zone, tars in sorted(by_zone.items())}


def sample_evenly(items: list[str], cap: int) -> list[str]:
    """Take up to cap items spread evenly across the list."""
    if len(items) <= cap:
        return items
    step = len(items) / cap
    return [items[int(i * step)] for i in range(cap)]


def iter_tar_images(tar_path: Path, zone: str) -> Iterator[SourceImage]:
    """Yield each 512 px NAIP chip in a tar as an RGB SourceImage."""
    with tarfile.open(tar_path) as tar:
        for member in tar:
            if not (member.isfile() and member.name.endswith(".png")):
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            array = np.asarray(Image.open(io.BytesIO(extracted.read())).convert("RGB"))
            yield SourceImage(
                array=array,
                source="naip",
                license=LICENSE,
                gsd_m=GSD_M,
                original_id=Path(member.name).stem,
                extra={"utm_zone": zone},
            )


def generate_tiles(
    api: HfApi,
    workdir: Path,
    target_tiles: int,
    per_zone: int,
    tile_px: int,
    min_valid: float,
) -> Iterator[Tile]:
    """Interleave UTM zones and yield tiles up to target_tiles."""
    zones = list_zone_tars(api)
    sampled = {zone: sample_evenly(tars, per_zone) for zone, tars in zones.items()}
    interleaved = [t for t in itertools.chain(*itertools.zip_longest(*sampled.values())) if t]
    print(f"zones: {len(zones)} | sampled tars: {len(interleaved)}")

    total = 0
    for key in interleaved:
        if total >= target_tiles:
            return
        zone = key.split("/")[0]
        local = Path(hf_hub_download(REPO, key, repo_type="dataset", local_dir=str(workdir)))
        try:
            for image in iter_tar_images(local, zone):
                for tile in iter_tiles(image, tile_px, min_valid):
                    if total >= target_tiles:
                        return
                    yield tile
                    total += 1
        finally:
            local.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/tiles/naip"))
    parser.add_argument("--target-tiles", type=int, default=100000)
    parser.add_argument("--per-zone", type=int, default=1000)
    parser.add_argument("--tile-px", type=int, default=TILE_PX)
    parser.add_argument("--min-valid", type=float, default=0.7)
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    args = parser.parse_args()

    api = HfApi()
    with tempfile.TemporaryDirectory(prefix="naip_") as tmp:
        tiles = generate_tiles(
            api,
            Path(tmp),
            args.target_tiles,
            args.per_zone,
            args.tile_px,
            args.min_valid,
        )
        manifest = write_shards(tiles, args.out, "naip", args.shard_size)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"DONE: {manifest['n_tiles']} tiles in {manifest['n_shards']} shards -> {args.out}")


if __name__ == "__main__":
    main()
