"""Build a deduplicated CC-BY VHR image-tile corpus from OpenAerialMap.

Queries the OAM STAC API for sub-meter CC-BY imagery, removes overlapping/re-uploaded
orthomosaics by footprint IoU, then tiles the survivor COGs at a target GSD into fixed
patches for self-supervised pretraining. Reads are windowed over /vsicurl so whole
orthomosaics are never downloaded.
"""

import argparse
import json
import os
from collections.abc import Iterator
from pathlib import Path

import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.windows import Window
from shapely.geometry import shape
from shapely.strtree import STRtree

from dinov3_vhr.data.tiles import SHARD_SIZE, TILE_PX, Tile, write_shards

STAC_SEARCH = "https://api.imagery.hotosm.org/stac/search"
LICENSE = "CC-BY-4.0"

# hot_buildings (evaluation suite) is built from this OAM mosaic; hold it out of training.
HELDOUT_MOSAICS = frozenset({"60d35697eade0f00079b58c6"})

os.environ.update(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    GDAL_HTTP_MULTIRANGE="YES",
    VSI_CACHE="TRUE",
)


def fetch_candidates(max_items: int, gsd_max: float, page_size: int = 250) -> list[dict]:
    """Page the OAM STAC search for CC-BY items at or below gsd_max."""
    items: list[dict] = []
    params = {
        "collections": "openaerialmap",
        "limit": page_size,
        "filter-lang": "cql2-text",
        "filter": f"license='CC-BY-4.0' AND gsd<={gsd_max}",
    }
    url = STAC_SEARCH
    while url and len(items) < max_items:
        response = requests.get(url, params=params if url == STAC_SEARCH else None, timeout=60)
        response.raise_for_status()
        page = response.json()
        for feature in page["features"]:
            items.append(
                {
                    "id": feature["id"],
                    "gsd": feature["properties"]["gsd"],
                    "geom": shape(feature["geometry"]),
                    "href": feature["assets"]["visual"]["href"],
                }
            )
        url = next((link["href"] for link in page["links"] if link["rel"] == "next"), None)
    return items[:max_items]


def dedup_by_footprint(items: list[dict], iou_threshold: float) -> list[dict]:
    """Drop overlapping orthomosaics, keeping the finest-GSD item per overlap cluster."""
    geoms = [item["geom"] for item in items]
    tree = STRtree(geoms)
    parent = list(range(len(items)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, geom in enumerate(geoms):
        for j in tree.query(geom):
            if j <= i:
                continue
            inter = geom.intersection(geoms[j]).area
            if inter == 0:
                continue
            union = geom.union(geoms[j]).area
            if union > 0 and inter / union >= iou_threshold:
                parent[find(i)] = find(j)

    clusters: dict[int, dict] = {}
    for idx, item in enumerate(items):
        root = find(idx)
        if root not in clusters or item["gsd"] < clusters[root]["gsd"]:
            clusters[root] = item
    return list(clusters.values())


def iter_cog_tiles(
    item: dict,
    target_gsd: float,
    tile_px: int,
    max_tiles: int,
    min_valid: float,
    native: bool,
) -> Iterator[Tile]:
    """Window-read a COG and yield valid tile_px RGB tiles at native or target GSD."""
    if native:
        factor = 1
        out_gsd = round(item["gsd"], 4)
    else:
        factor = max(1, round(target_gsd / item["gsd"]))
        out_gsd = target_gsd
    win = tile_px * factor
    lon, lat = item["geom"].centroid.x, item["geom"].centroid.y
    saved = 0
    with rasterio.open(f"/vsicurl/{item['href']}") as src:
        if src.count < 3:
            print(f"  skip {item['id']}: {src.count}-band (not RGB)")
            return
        if src.dtypes[0] != "uint8":
            print(f"  skip {item['id']}: {src.dtypes[0]} (not 8-bit)")
            return
        has_alpha = src.count >= 4
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
                    "source": "oam",
                    "license": LICENSE,
                    "gsd_m": out_gsd,
                    "original_id": item["id"],
                    "row": row,
                    "col": col,
                    "tile_px": tile_px,
                    "orig_gsd_m": round(item["gsd"], 4),
                    "lon": round(lon, 6),
                    "lat": round(lat, 6),
                }
                yield Tile(
                    array=rgb.transpose(1, 2, 0),
                    key=f"oam_{item['id']}_{row}_{col}",
                    meta=meta,
                )
                saved += 1


def generate_tiles(
    survivors: list[dict],
    target_tiles: int,
    target_gsd: float,
    tile_px: int,
    max_tiles_per_image: int,
    min_valid: float,
    native: bool,
) -> Iterator[Tile]:
    """Yield tiles across survivors up to target_tiles, skipping held-out mosaics."""
    total = 0
    for n, item in enumerate(survivors):
        if total >= target_tiles:
            return
        if item["id"] in HELDOUT_MOSAICS:
            print(f"[{n + 1}/{len(survivors)}] {item['id']} held out (evaluation overlap)")
            continue
        before = total
        for tile in iter_cog_tiles(
            item,
            target_gsd,
            tile_px,
            min(max_tiles_per_image, target_tiles - total),
            min_valid,
            native,
        ):
            yield tile
            total += 1
        print(
            f"[{n + 1}/{len(survivors)}] {item['id']} gsd={item['gsd']:.3f} "
            f"tiles={total - before} total={total}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/tiles/oam"))
    parser.add_argument("--max-items", type=int, default=2000)
    parser.add_argument("--target-tiles", type=int, default=60000)
    parser.add_argument("--gsd-max", type=float, default=1.0)
    parser.add_argument("--target-gsd", type=float, default=0.5)
    parser.add_argument("--tile-px", type=int, default=TILE_PX)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-tiles-per-image", type=int, default=80)
    parser.add_argument("--min-valid", type=float, default=0.7)
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    parser.add_argument("--native", action="store_true", help="tile at each item's native GSD")
    args = parser.parse_args()

    candidates = fetch_candidates(args.max_items, args.gsd_max)
    survivors = dedup_by_footprint(candidates, args.iou)
    print(f"candidates: {len(candidates)} | after dedup: {len(survivors)}")

    tiles = generate_tiles(
        survivors,
        args.target_tiles,
        args.target_gsd,
        args.tile_px,
        args.max_tiles_per_image,
        args.min_valid,
        args.native,
    )
    manifest = write_shards(tiles, args.out, "oam", args.shard_size)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"DONE: {manifest['n_tiles']} tiles in {manifest['n_shards']} shards -> {args.out}")


if __name__ == "__main__":
    main()
