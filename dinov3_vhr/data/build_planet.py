"""Stream Planet SkySat/Pelican disaster imagery into standard tile shards.

Pulls the open CC-BY-NC 50 cm "visual" RGB products Planet releases per disaster event
on Source Cooperative and windowed-tiles them at the target GSD. Some events ship the
visual product as a zipped GeoTIFF (downloaded to a temp dir and deleted after tiling),
others as a direct COG read over /vsicurl. Events are interleaved for geographic spread.
"""

import argparse
import itertools
import json
import os
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path

import boto3
import rasterio
from botocore import UNSIGNED
from botocore.client import BaseClient
from botocore.config import Config
from rasterio.enums import Resampling
from rasterio.io import DatasetReader
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

from dinov3_vhr.data.tiles import SHARD_SIZE, TILE_PX, Tile, write_shards

ENDPOINT = "https://data.source.coop"
BUCKET = "planet"
HTTP_BASE = f"{ENDPOINT}/{BUCKET}"
LICENSE = "CC-BY-NC-4.0"
EVENTS = [
    "venezuela-earthquake-2026-06-24",
    "philippines-earthquake-2026-06-08",
    "disasterdata/hurricane-melissa-2025",
]

os.environ.update(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    GDAL_HTTP_MULTIRANGE="YES",
    VSI_CACHE="TRUE",
)


def client() -> BaseClient:
    """An unsigned S3 client for the Source Cooperative endpoint."""
    return boto3.client("s3", endpoint_url=ENDPOINT, config=Config(signature_version=UNSIGNED))


def _is_visual(key: str) -> bool:
    return key.endswith(("_visual.zip", "_visual.tif", "_visual_file_format.tif"))


def list_visual_units(s3: BaseClient, event: str) -> list[str]:
    """List the visual RGB products (zip or direct tif) under one event."""
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{event}/"):
        keys.extend(o["Key"] for o in page.get("Contents", []) if _is_visual(o["Key"]))
    return sorted(keys)


def _sensor_of(key: str) -> str:
    return "pelican" if "pelican" in key.lower() else "skysat"


def _tile_dataset(
    src: DatasetReader,
    aoi: str,
    event: str,
    sensor: str,
    target_gsd: float,
    tile_px: int,
    max_tiles: int,
    min_valid: float,
) -> Iterator[Tile]:
    """Windowed-tile an open visual dataset into standard RGB tiles."""
    native = src.res[0]
    factor = max(1, round(target_gsd / native)) if native else 1
    win = tile_px * factor
    has_alpha = src.count >= 4
    centre_x = (src.bounds.left + src.bounds.right) / 2
    centre_y = (src.bounds.top + src.bounds.bottom) / 2
    lons, lats = warp_transform(src.crs, "EPSG:4326", [centre_x], [centre_y])
    lon, lat = round(lons[0], 6), round(lats[0], 6)
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
                "source": "planet",
                "license": LICENSE,
                "gsd_m": target_gsd,
                "original_id": f"{aoi}_{row}_{col}",
                "row": row,
                "col": col,
                "tile_px": tile_px,
                "sensor": sensor,
                "event": event,
                "lon": lon,
                "lat": lat,
            }
            yield Tile(array=rgb.transpose(1, 2, 0), key=f"planet_{aoi}_{row}_{col}", meta=meta)
            saved += 1


def iter_unit_tiles(
    s3: BaseClient,
    key: str,
    event: str,
    workdir: Path,
    target_gsd: float,
    tile_px: int,
    max_tiles: int,
    min_valid: float,
) -> Iterator[Tile]:
    """Yield tiles for one visual unit, handling both zipped and direct-tif layouts."""
    aoi = Path(key).stem
    sensor = _sensor_of(key)
    if key.endswith(".zip"):
        zip_path = workdir / Path(key).name
        s3.download_file(BUCKET, key, str(zip_path))
        with zipfile.ZipFile(zip_path) as archive:
            tif_name = next((n for n in archive.namelist() if n.lower().endswith(".tif")), None)
            if tif_name is None:
                zip_path.unlink()
                return
            archive.extract(tif_name, workdir)
        tif_path = workdir / tif_name
        try:
            with rasterio.open(tif_path) as src:
                yield from _tile_dataset(
                    src, aoi, event, sensor, target_gsd, tile_px, max_tiles, min_valid
                )
        finally:
            tif_path.unlink(missing_ok=True)
            zip_path.unlink(missing_ok=True)
    else:
        with rasterio.open(f"/vsicurl/{HTTP_BASE}/{key}") as src:
            yield from _tile_dataset(
                src, aoi, event, sensor, target_gsd, tile_px, max_tiles, min_valid
            )


def generate_tiles(
    s3: BaseClient,
    events: list[str],
    workdir: Path,
    target_tiles: int,
    target_gsd: float,
    tile_px: int,
    max_tiles_per_unit: int,
    min_valid: float,
) -> Iterator[Tile]:
    """Interleave events and yield tiles up to target_tiles."""
    units_by_event = [[(event, k) for k in list_visual_units(s3, event)] for event in events]
    interleaved = [u for u in itertools.chain(*itertools.zip_longest(*units_by_event)) if u]
    print(f"events: {len(events)} | visual units: {len(interleaved)}")

    total = 0
    for event, key in interleaved:
        if total >= target_tiles:
            return
        before = total
        for tile in iter_unit_tiles(
            s3,
            key,
            event,
            workdir,
            target_gsd,
            tile_px,
            min(max_tiles_per_unit, target_tiles - total),
            min_valid,
        ):
            yield tile
            total += 1
        print(f"  [{event}] {key.split('/')[-1]} tiles={total - before} total={total}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/tiles/planet"))
    parser.add_argument("--target-tiles", type=int, default=40000)
    parser.add_argument("--target-gsd", type=float, default=0.5)
    parser.add_argument("--tile-px", type=int, default=TILE_PX)
    parser.add_argument("--max-tiles-per-unit", type=int, default=2000)
    parser.add_argument("--min-valid", type=float, default=0.7)
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    args = parser.parse_args()

    s3 = client()
    with tempfile.TemporaryDirectory(prefix="planet_") as tmp:
        tiles = generate_tiles(
            s3,
            EVENTS,
            Path(tmp),
            args.target_tiles,
            args.target_gsd,
            args.tile_px,
            args.max_tiles_per_unit,
            args.min_valid,
        )
        manifest = write_shards(tiles, args.out, "planet", args.shard_size)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"DONE: {manifest['n_tiles']} tiles in {manifest['n_shards']} shards -> {args.out}")


if __name__ == "__main__":
    main()
