"""Stream fMoW-RGB from public S3 into standard tile shards, sampled for balance.

Lists the fMoW-rgb train split on the public spacenet-dataset bucket, samples up to a
cap of instances per category, streams each _rgb.jpg in memory, tiles it with a
per-image cap so large scenes do not dominate, and round-robins across categories so
no single category floods the mix. Raw imagery is never stored on disk.
"""

import argparse
import io
import itertools
import json
from collections.abc import Iterator
from pathlib import Path

import boto3
import numpy as np
from botocore import UNSIGNED
from botocore.client import BaseClient
from botocore.config import Config
from PIL import Image

from dinov3_vhr.data.tiles import (
    SHARD_SIZE,
    TILE_PX,
    SourceImage,
    Tile,
    iter_tiles,
    write_shards,
)

# fMoW scenes routinely exceed PIL's default bomb threshold; the source is trusted.
Image.MAX_IMAGE_PIXELS = None

BUCKET = "spacenet-dataset"
PREFIX = "Hosted-Datasets/fmow/fmow-rgb"
LICENSE = "fMoW-Challenge-Public-License-NC"


def client() -> BaseClient:
    """An unsigned S3 client for the public bucket."""
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_categories(s3: BaseClient, split: str) -> list[str]:
    """List the category folder names under a split."""
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{PREFIX}/{split}/", Delimiter="/")
    return sorted(p["Prefix"].rstrip("/").split("/")[-1] for p in resp.get("CommonPrefixes", []))


def list_instances(s3: BaseClient, split: str, category: str) -> list[str]:
    """List the instance-folder prefixes for one category."""
    prefixes: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=BUCKET, Prefix=f"{PREFIX}/{split}/{category}/", Delimiter="/"
    ):
        prefixes.extend(cp["Prefix"] for cp in page.get("CommonPrefixes", []))
    return prefixes


def sample_evenly(items: list[str], cap: int) -> list[str]:
    """Take up to cap items spread evenly across the list."""
    if len(items) <= cap:
        return items
    step = len(items) / cap
    return [items[int(i * step)] for i in range(cap)]


def rgb_keys(s3: BaseClient, instance_prefix: str) -> list[str]:
    """List the _rgb.jpg keys inside one instance folder."""
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=instance_prefix)
    return sorted(o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith("_rgb.jpg"))


def _get(s3: BaseClient, key: str) -> bytes:
    return s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()


def _source_image(s3: BaseClient, key: str, category: str) -> SourceImage | None:
    """Stream one fMoW image and its metadata; None if the JPEG cannot be decoded."""
    try:
        array = np.asarray(Image.open(io.BytesIO(_get(s3, key))).convert("RGB"))
    except (OSError, Image.UnidentifiedImageError) as error:
        print(f"  skip {key}: {error}")
        return None
    meta = json.loads(_get(s3, f"{key[:-4]}.json"))
    return SourceImage(
        array=array,
        source="fmow",
        license=LICENSE,
        gsd_m=meta.get("gsd"),
        original_id=key.split("/")[-1][:-4],
        extra={"category": category, "country": meta.get("country_code")},
    )


def generate_tiles(
    s3: BaseClient,
    split: str,
    per_category: int,
    target_tiles: int,
    max_tiles_per_image: int,
    tile_px: int,
    min_valid: float,
) -> Iterator[Tile]:
    """Round-robin categories, yielding tiles up to target_tiles."""
    categories = list_categories(s3, split)
    queues = {
        category: iter(
            key
            for instance in sample_evenly(list_instances(s3, split, category), per_category)
            for key in rgb_keys(s3, instance)
        )
        for category in categories
    }
    print(f"categories: {len(categories)}")

    total = 0
    active = list(categories)
    while active and total < target_tiles:
        for category in list(active):
            if total >= target_tiles:
                break
            key = next(queues[category], None)
            if key is None:
                active.remove(category)
                continue
            image = _source_image(s3, key, category)
            if image is None:
                continue
            for tile in itertools.islice(
                iter_tiles(image, tile_px, min_valid), max_tiles_per_image
            ):
                yield tile
                total += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/tiles/fmow"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--target-tiles", type=int, default=500000)
    parser.add_argument("--per-category", type=int, default=3000)
    parser.add_argument("--max-tiles-per-image", type=int, default=30)
    parser.add_argument("--tile-px", type=int, default=TILE_PX)
    parser.add_argument("--min-valid", type=float, default=0.7)
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    args = parser.parse_args()

    s3 = client()
    tiles = generate_tiles(
        s3,
        args.split,
        args.per_category,
        args.target_tiles,
        args.max_tiles_per_image,
        args.tile_px,
        args.min_valid,
    )
    manifest = write_shards(tiles, args.out, "fmow", args.shard_size)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"DONE: {manifest['n_tiles']} tiles in {manifest['n_shards']} shards -> {args.out}")


if __name__ == "__main__":
    main()
