"""Build an H3 coverage GeoJSON of the standardized tile shards.

Scans every shard's per-tile metadata, resolves a longitude/latitude for each tile
(precise per-COG coordinates where the source carries them, representative coordinates
otherwise), aggregates counts into H3 cells per source, and writes a GeoJSON of hexagon
polygons plus a printed summary. Approximate placements are flagged so the map does not
overclaim precision.
"""

import argparse
import collections
import csv
import functools
import gzip
import io
import json
import tarfile
from pathlib import Path

import h3
import pyproj
import requests

FMOW_CENTROIDS_URL = (
    "https://raw.githubusercontent.com/eesur/country-codes-lat-long/master/"
    "country-codes-lat-long-alpha3.json"
)
WORLDSTRAT_METADATA_URL = "https://zenodo.org/records/6810792/files/metadata.csv?download=1"

# s2-naip tiles are 512 px at 1.25 m/pixel, so one grid step is 640 m in its UTM CRS.
NAIP_TILE_METRES = 640
# Plausible VHR ground sample distance bounds in metres. Out-of-range values are contributor
# metadata errors (some OpenAerialMap gsd tags are sub-millimetre), dropped from the resolution
# average but still counted in coverage.
GSD_MIN_VALID = 0.01
GSD_MAX_VALID = 2.0
# RAMP subset -> approximate city centroid (lon, lat).
RAMP_CITY = {
    "accra_ghana": (-0.19, 5.60),
    "barishal_bangladesh": (90.37, 22.70),
    "bentiu_south_sudan": (29.79, 9.23),
    "chittagong_bangladesh": (91.83, 22.36),
    "coxs_bazar_bangladesh": (91.98, 21.43),
    "daressalaam_tanzania": (39.28, -6.82),
    "dhaka_bangladesh": (90.41, 23.81),
    "hpa_an_myanmar": (97.63, 16.89),
    "jashore_bangladesh": (89.21, 23.17),
    "karnataka_india": (75.71, 15.32),
    "les_cayes_haiti": (-73.75, 18.20),
    "lubumbashi_drc": (27.48, -11.66),
    "manjama_sierra_leone": (-11.74, 7.88),
    "mesopotamia_st_vincent": (-61.20, 13.20),
    "muscat_oman": (58.41, 23.59),
    "mzuzu_malawi": (34.02, -11.46),
    "nairobi_kenya": (36.82, -1.29),
    "ndjamena_chad": (15.04, 12.13),
    "sylhet_bangladesh": (91.87, 24.90),
    "wa_ghana": (-2.50, 10.06),
}
# LoveDA covers three eastern Chinese cities; one representative point.
LOVEDA_POINT = (118.80, 32.06)


@functools.lru_cache(maxsize=16)
def _utm_transformer(epsg: int) -> pyproj.Transformer:
    return pyproj.Transformer.from_crs(epsg, 4326, always_xy=True)


def naip_lonlat(original_id: str) -> tuple[float, float]:
    """Decode a NAIP tile id (epsg_col_row) to a real (lon, lat)."""
    epsg, col, row = (int(part) for part in original_id.split("_"))
    lon, lat = _utm_transformer(epsg).transform(col * NAIP_TILE_METRES, -row * NAIP_TILE_METRES)
    return lon, lat


def fmow_centroids() -> dict[str, tuple[float, float]]:
    """Fetch ISO alpha-3 country centroids as {alpha3: (lon, lat)}."""
    rows = requests.get(FMOW_CENTROIDS_URL, timeout=60).json()["ref_country_codes"]
    return {r["alpha3"]: (float(r["longitude"]), float(r["latitude"])) for r in rows}


def worldstrat_aois() -> dict[str, tuple[float, float]]:
    """Fetch WorldStrat AOI centroids as {aoi: (lon, lat)}, empty on any failure."""
    raw = requests.get(WORLDSTRAT_METADATA_URL, timeout=120).content
    text = (
        gzip.decompress(raw).decode("utf-8", "ignore")
        if raw[:2] == b"\x1f\x8b"
        else raw.decode("utf-8", "ignore")
    )
    reader = csv.DictReader(io.StringIO(text))
    names = reader.fieldnames or []
    fields = {f.lower(): f for f in names}
    # The AOI id is the first column, which ships with an empty header.
    key = names[0] if names else None
    lon_c = next((fields[c] for c in ("lon", "longitude", "lng") if c in fields), None)
    lat_c = next((fields[c] for c in ("lat", "latitude") if c in fields), None)
    if key is None or lon_c is None or lat_c is None:
        return {}
    out: dict[str, tuple[float, float]] = {}
    for row in reader:
        try:
            out[row[key]] = (float(row[lon_c]), float(row[lat_c]))
        except (ValueError, KeyError):
            continue
    return out


def resolve_lonlat(meta: dict, lookups: dict) -> tuple[float, float, bool] | None:
    """Return (lon, lat, precise) for one tile, or None if it cannot be placed."""
    lon, lat = meta.get("lon"), meta.get("lat")
    if lon is not None and lat is not None:
        return float(lon), float(lat), True
    source = meta["source"]
    if source == "naip":
        lon, lat = naip_lonlat(meta["original_id"])
        return lon, lat, True
    if source == "ramp":
        point = RAMP_CITY.get(meta.get("country", ""))
        return (*point, False) if point else None
    if source == "loveda":
        return (*LOVEDA_POINT, False)
    if source == "fmow":
        point = lookups["fmow"].get(meta.get("country") or "")
        return (*point, False) if point else None
    if source == "worldstrat":
        point = lookups["worldstrat"].get(meta.get("aoi", ""))
        return (*point, True) if point else None
    return None


def gsd_stats(hist: collections.Counter) -> list[float] | None:
    """Return [min, weighted-median, max] ground sample distance from a value histogram."""
    if not hist:
        return None
    values = sorted(hist)
    half = sum(hist.values()) / 2
    cumulative = 0
    median = values[-1]
    for value in values:
        cumulative += hist[value]
        if cumulative >= half:
            median = value
            break
    return [round(values[0], 3), round(median, 3), round(values[-1], 3)]


def scan(tiles_dir: Path, resolution: int, lookups: dict) -> tuple[dict, dict, int]:
    """Aggregate tiles into H3 cells; return per-cell counts and GSD, per-source stats, unplaced."""
    cells: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    cell_gsd: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    source_gsd: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    precise_cells: set[str] = set()
    stats: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    unplaced = 0
    for shard in sorted(tiles_dir.glob("*/*.tar")):
        with tarfile.open(shard) as tar:
            for member in tar:
                if not member.name.endswith(".json"):
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                meta = json.loads(extracted.read())
                placed = resolve_lonlat(meta, lookups)
                if placed is None:
                    unplaced += 1
                    stats[meta["source"]]["unplaced"] += 1
                    continue
                lon, lat, precise = placed
                cell = h3.latlng_to_cell(lat, lon, resolution)
                cells[cell][meta["source"]] += 1
                stats[meta["source"]]["placed"] += 1
                gsd = meta.get("gsd_m")
                if gsd is not None and GSD_MIN_VALID <= float(gsd) <= GSD_MAX_VALID:
                    bucket = round(float(gsd), 3)
                    cell_gsd[cell][bucket] += 1
                    source_gsd[meta["source"]][bucket] += 1
                elif gsd is not None:
                    stats[meta["source"]]["gsd_dropped"] += 1
                if precise:
                    precise_cells.add(cell)
                    stats[meta["source"]]["precise"] += 1
    return (
        {"cells": cells, "precise": precise_cells, "gsd": cell_gsd},
        {"placement": stats, "source_gsd": source_gsd},
        unplaced,
    )


def to_geojson(cells: dict) -> dict:
    """Turn per-cell source counts into a GeoJSON FeatureCollection of hexagons."""
    features = []
    for cell, counts in cells["cells"].items():
        boundary = h3.cell_to_boundary(cell)
        ring = [[lon, lat] for lat, lon in boundary]
        ring.append(ring[0])
        gsd = gsd_stats(cells["gsd"].get(cell, collections.Counter()))
        properties = {
            "h3": cell,
            "total": sum(counts.values()),
            "precise": cell in cells["precise"],
            **{f"n_{source}": n for source, n in counts.items()},
        }
        if gsd is not None:
            properties["gsd_min"], properties["gsd_med"], properties["gsd_max"] = gsd
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": properties,
            }
        )
    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles-dir", type=Path, default=Path("data/tiles"))
    parser.add_argument("--out", type=Path, default=Path("data/footprint_h3.geojson"))
    parser.add_argument("--resolution", type=int, default=3)
    args = parser.parse_args()

    lookups = {"fmow": fmow_centroids(), "worldstrat": worldstrat_aois()}
    print(
        f"loaded {len(lookups['fmow'])} country centroids, "
        f"{len(lookups['worldstrat'])} worldstrat AOIs"
    )

    cells, stats, unplaced = scan(args.tiles_dir, args.resolution, lookups)
    geojson = to_geojson(cells)
    args.out.write_text(json.dumps(geojson))

    placement, source_gsd = stats["placement"], stats["source_gsd"]
    print(f"\nH3 res {args.resolution}: {len(geojson['features'])} cells, {unplaced} unplaced")
    print(f"{'source':<12}{'placed':>10}{'precise':>10}{'gsd_drop':>10}{'gsd min/med/max':>22}")
    for source in sorted(placement):
        s = placement[source]
        gsd = gsd_stats(source_gsd.get(source, collections.Counter()))
        span = f"{gsd[0]}/{gsd[1]}/{gsd[2]}" if gsd else "n/a"
        print(f"{source:<12}{s['placed']:>10}{s['precise']:>10}{s['gsd_dropped']:>10}{span:>22}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
