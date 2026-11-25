"""Stream Maxar/Vantor Open Data visual COGs into standard tile shards, capped.

Crawls the public maxar-opendata bucket's disaster events, windowed-tiles each 30 cm
8-bit visual RGB COG at the target GSD, and writes shards. Events that predate 2020 (the
xView2/xBD window) or match an xBD disaster name are held out so training never touches
the source imagery of the xView2 evaluation. Events are interleaved for geographic spread.
"""

import argparse
import itertools
import json
import os
from collections.abc import Iterator
from pathlib import Path

import boto3
import rasterio
from botocore import UNSIGNED
from botocore.client import BaseClient
from botocore.config import Config
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.io import DatasetReader
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

from dinov3_vhr.data.tiles import SHARD_SIZE, TILE_PX, Tile, write_shards

BUCKET = "maxar-opendata"
HTTP_BASE = f"https://{BUCKET}.s3.amazonaws.com"
LICENSE = "CC-BY-NC-4.0"

# xView2/xBD is built from Maxar Open Data (2011-2019 events) and is in the eval suite.
# Hold out anything in that window or matching an xBD disaster name.
HELDOUT_EVENTS = frozenset(
    {
        "Nepal-Earthquake-Apr-2015",
        "Hurricane-Maria-PuertoRico-Oct-2017",
        "Vanuatu-Earthquake-Dec17",
    }
)
XBD_TOKENS = (
    "guatemala-volcano",
    "palu",
    "sunda",
    "woolsey",
    "socal",
    "santa-rosa",
    "pinery",
    "joplin",
    "moore-tornado",
    "tuscaloosa",
    "lower-puna",
    "portugal-wildfire",
    "midwest",
    "hurricane-matthew",
    "hurricane-michael",
    "hurricane-florence",
    "hurricane-harvey",
    "mexico-earthquake",
    "nepal-flood",
)

os.environ.update(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    GDAL_HTTP_MULTIRANGE="YES",
    VSI_CACHE="TRUE",
)


def client() -> BaseClient:
    """An unsigned S3 client for the public maxar-opendata bucket."""
    return boto3.client("s3", region_name="us-west-2", config=Config(signature_version=UNSIGNED))


def _is_heldout(event: str) -> bool:
    lowered = event.lower()
    return event in HELDOUT_EVENTS or any(token in lowered for token in XBD_TOKENS)


def list_events(s3: BaseClient) -> list[str]:
    """List event names, dropping the xBD-window and xBD-named ones."""
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix="events/", Delimiter="/")
    events = [p["Prefix"].split("/")[1] for p in resp.get("CommonPrefixes", [])]
    return sorted(e for e in events if not _is_heldout(e))


def list_visual_cogs(s3: BaseClient, event: str, limit: int) -> list[str]:
    """List up to limit -visual.tif COG keys under one event, stopping early."""
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"events/{event}/"):
        keys.extend(o["Key"] for o in page.get("Contents", []) if o["Key"].endswith("-visual.tif"))
        if len(keys) >= limit:
            break
    return sorted(keys)[:limit]


def iter_cog_tiles(
    src: DatasetReader,
    key: str,
    event: str,
    target_gsd: float,
    tile_px: int,
    max_tiles: int,
    min_valid: float,
    native: bool,
) -> Iterator[Tile]:
    """Windowed-tile one visual COG into standard RGB tiles at native or target GSD."""
    if src.count < 3:
        return
    native_res = src.res[0]
    if native:
        factor = 1
        out_gsd = round(native_res, 4)
    else:
        factor = max(1, round(target_gsd / native_res)) if native_res else 1
        out_gsd = target_gsd
    win = tile_px * factor
    has_alpha = src.count >= 4
    centre_x = (src.bounds.left + src.bounds.right) / 2
    centre_y = (src.bounds.top + src.bounds.bottom) / 2
    lons, lats = warp_transform(src.crs, "EPSG:4326", [centre_x], [centre_y])
    lon, lat = round(lons[0], 6), round(lats[0], 6)
    catalog_id = Path(key).stem
    saved = 0
    for row in range(0, src.height - win, win):
        for col in range(0, src.width - win, win):
            if saved >= max_tiles:
                return
            # ty cannot see rasterio's attrs-generated Window constructor
            window = Window(col, row, win, win)  # ty: ignore[too-many-positional-arguments]
            rgb = src.read(
                [1, 2, 3],
                window=window,
                out_shape=(3, tile_px, tile_px),
                resampling=Resampling.bilinear,
            )
            if has_alpha:
                alpha = src.read(
                    4,
                    window=window,
                    out_shape=(tile_px, tile_px),
                    resampling=Resampling.nearest,
                )
                valid = float((alpha > 0).mean())
            else:
                valid = float((rgb.sum(axis=0) > 0).mean())
            if valid < min_valid:
                continue
            meta = {
                "source": "maxar",
                "license": LICENSE,
                "gsd_m": out_gsd,
                "original_id": f"{catalog_id}_{row}_{col}",
                "row": row,
                "col": col,
                "tile_px": tile_px,
                "sensor": "worldview",
                "event": event,
                "lon": lon,
                "lat": lat,
            }
            yield Tile(
                array=rgb.transpose(1, 2, 0),
                key=f"maxar_{event}_{catalog_id}_{row}_{col}",
                meta=meta,
            )
            saved += 1


def generate_tiles(
    s3: BaseClient,
    target_tiles: int,
    target_gsd: float,
    tile_px: int,
    max_tiles_per_cog: int,
    min_valid: float,
    cogs_per_event: int,
    native: bool,
) -> Iterator[Tile]:
    """Interleave events and yield tiles up to target_tiles."""
    events = list_events(s3)
    cogs_by_event = [[(e, k) for k in list_visual_cogs(s3, e, cogs_per_event)] for e in events]
    interleaved = [u for u in itertools.chain(*itertools.zip_longest(*cogs_by_event)) if u]
    print(f"events: {len(events)} | visual COGs: {len(interleaved)}")

    total = 0
    for event, key in interleaved:
        if total >= target_tiles:
            return
        before = total
        try:
            with rasterio.open(f"/vsicurl/{HTTP_BASE}/{key}") as src:
                for tile in iter_cog_tiles(
                    src,
                    key,
                    event,
                    target_gsd,
                    tile_px,
                    min(max_tiles_per_cog, target_tiles - total),
                    min_valid,
                    native,
                ):
                    yield tile
                    total += 1
        except RasterioIOError as error:
            print(f"  [{event}] {Path(key).name} skipped: {error}")
            continue
        print(f"  [{event}] {Path(key).name} tiles={total - before} total={total}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/tiles/maxar"))
    parser.add_argument("--target-tiles", type=int, default=120000)
    parser.add_argument("--target-gsd", type=float, default=0.5)
    parser.add_argument("--tile-px", type=int, default=TILE_PX)
    parser.add_argument("--max-tiles-per-cog", type=int, default=400)
    parser.add_argument("--cogs-per-event", type=int, default=60)
    parser.add_argument("--min-valid", type=float, default=0.7)
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    parser.add_argument("--native", action="store_true", help="tile at each COG's native GSD")
    args = parser.parse_args()

    s3 = client()
    tiles = generate_tiles(
        s3,
        args.target_tiles,
        args.target_gsd,
        args.tile_px,
        args.max_tiles_per_cog,
        args.min_valid,
        args.cogs_per_event,
        args.native,
    )
    manifest = write_shards(tiles, args.out, "maxar", args.shard_size)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"DONE: {manifest['n_tiles']} tiles in {manifest['n_shards']} shards -> {args.out}")


if __name__ == "__main__":
    main()
