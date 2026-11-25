import json
import tarfile

import numpy as np
from dinov3_vhr.data.tiles import SourceImage, iter_tiles, write_shards


def _image(height: int, width: int, fill: int = 200) -> SourceImage:
    array = np.full((height, width, 3), fill, dtype=np.uint8)
    return SourceImage(
        array=array,
        source="test",
        license="CC-BY-4.0",
        gsd_m=0.5,
        original_id="img0",
        extra={"split": "train"},
    )


def test_iter_tiles_counts_non_overlapping_grid():
    tiles = list(iter_tiles(_image(512, 512), tile_px=256))
    assert len(tiles) == 4
    assert all(tile.array.shape == (256, 256, 3) for tile in tiles)


def test_iter_tiles_drops_partial_edges():
    tiles = list(iter_tiles(_image(700, 512), tile_px=256))
    assert len(tiles) == 4


def test_iter_tiles_skips_mostly_black_tiles():
    image = _image(256, 256)
    image.array[:, :200, :] = 0
    assert list(iter_tiles(image, tile_px=256, min_valid=0.7)) == []


def test_iter_tiles_rejects_non_uint8():
    bad = SourceImage(
        array=np.zeros((256, 256, 3), dtype=np.float32),
        source="test",
        license="x",
        gsd_m=None,
        original_id="img0",
    )
    try:
        list(iter_tiles(bad))
    except ValueError:
        return
    raise AssertionError("expected ValueError on non-uint8 input")


def test_iter_tiles_carries_metadata():
    tile = next(iter_tiles(_image(256, 256)))
    assert tile.meta["source"] == "test"
    assert tile.meta["gsd_m"] == 0.5
    assert tile.meta["split"] == "train"
    assert tile.meta["tile_px"] == 256


def test_write_shards_packs_jpg_and_json(tmp_path):
    tiles = iter_tiles(_image(512, 512), tile_px=256)
    manifest = write_shards(tiles, tmp_path, source="test", shard_size=3)
    assert manifest["n_tiles"] == 4
    assert manifest["n_shards"] == 2

    members = []
    for shard in sorted(tmp_path.glob("test-*.tar")):
        with tarfile.open(shard) as tar:
            members.extend(tar.getnames())
    assert sum(name.endswith(".png") for name in members) == 4
    assert sum(name.endswith(".json") for name in members) == 4

    with tarfile.open(sorted(tmp_path.glob("test-*.tar"))[0]) as tar:
        json_name = next(name for name in tar.getnames() if name.endswith(".json"))
        entry = tar.extractfile(json_name)
        assert entry is not None
        meta = json.loads(entry.read())
    assert meta["source"] == "test"
