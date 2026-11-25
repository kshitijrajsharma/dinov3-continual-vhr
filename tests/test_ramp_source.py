import numpy as np
import rasterio
from dinov3_vhr.data.sources import ramp
from rasterio.transform import from_origin


def _write_chip(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((3, 256, 256), value, dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=256,
        width=256,
        count=3,
        dtype="uint8",
        crs="EPSG:32630",
        transform=from_origin(0, 0, 0.3, 0.3),
    ) as dst:
        dst.write(array)


def test_iter_images_skips_heldout_subsets(tmp_path):
    _write_chip(tmp_path / "ramp_accra_ghana" / "source" / "a.tif", 100)
    _write_chip(tmp_path / "ramp_paris_france" / "source" / "b.tif", 100)
    _write_chip(tmp_path / "ramp_shanghai_china" / "source" / "c.tif", 100)

    images = list(ramp.iter_images(tmp_path))

    assert len(images) == 1
    assert images[0].extra["subset"] == "ramp_accra_ghana"
    assert images[0].extra["country"] == "accra_ghana"
    assert images[0].array.shape == (256, 256, 3)
    assert images[0].array.dtype == np.uint8
