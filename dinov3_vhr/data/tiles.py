"""Standard 256x256 RGB tile format shared by every pretraining source.

A source adapter yields SourceImage records; iter_tiles cuts each into fixed
non-overlapping uint8 RGB tiles, and write_shards packs them into WebDataset tar
shards (one lossless `.png` and one `.json` per tile) so any source lands in one layout.
"""

import io
import json
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

TILE_PX = 256
PNG_COMPRESS = 6
SHARD_SIZE = 10000


@dataclass(frozen=True)
class SourceImage:
    """One image from a source, ready to tile. array is (H, W, 3) uint8 RGB."""

    array: np.ndarray
    source: str
    license: str
    gsd_m: float | None
    original_id: str
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Tile:
    """One standardized tile plus the metadata written alongside it."""

    array: np.ndarray
    key: str
    meta: dict


def iter_tiles(
    image: SourceImage, tile_px: int = TILE_PX, min_valid: float = 0.7
) -> Iterator[Tile]:
    """Cut non-overlapping tile_px RGB tiles, skipping ones mostly filled with black."""
    arr = image.array
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ValueError(
            f"{image.source}/{image.original_id}: expected (H, W, 3) uint8, "
            f"got shape {arr.shape} dtype {arr.dtype}"
        )
    height, width, _ = arr.shape
    for row in range(0, height - tile_px + 1, tile_px):
        for col in range(0, width - tile_px + 1, tile_px):
            tile = arr[row : row + tile_px, col : col + tile_px, :]
            if float((tile.sum(axis=2) > 0).mean()) < min_valid:
                continue
            meta = {
                "source": image.source,
                "license": image.license,
                "gsd_m": image.gsd_m,
                "original_id": image.original_id,
                "row": row,
                "col": col,
                "tile_px": tile_px,
                **image.extra,
            }
            yield Tile(
                array=tile,
                key=f"{image.source}_{image.original_id}_{row}_{col}",
                meta=meta,
            )


def encode_png(tile: np.ndarray, compress_level: int = PNG_COMPRESS) -> bytes:
    """Encode an RGB uint8 tile as lossless PNG bytes."""
    buffer = io.BytesIO()
    Image.fromarray(tile).save(buffer, format="PNG", compress_level=compress_level)
    return buffer.getvalue()


def _add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    """Append a byte payload to an open tar with a fixed mtime for reproducible shards."""
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = 0
    tar.addfile(info, io.BytesIO(payload))


def write_shards(
    tiles: Iterator[Tile], out_dir: Path, source: str, shard_size: int = SHARD_SIZE
) -> dict:
    """Write tiles into WebDataset tar shards of shard_size samples each."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    shard_index = -1
    tar: tarfile.TarFile | None = None
    try:
        for tile in tiles:
            if total % shard_size == 0:
                if tar is not None:
                    tar.close()
                shard_index += 1
                tar = tarfile.open(out_dir / f"{source}-{shard_index:06d}.tar", "w")
            assert tar is not None
            _add_bytes(tar, f"{tile.key}.png", encode_png(tile.array))
            _add_bytes(tar, f"{tile.key}.json", json.dumps(tile.meta).encode())
            total += 1
    finally:
        if tar is not None:
            tar.close()
    return {"source": source, "n_tiles": total, "n_shards": shard_index + 1}
