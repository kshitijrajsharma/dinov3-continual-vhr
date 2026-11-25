import io
import tarfile

import numpy as np
from dinov3_vhr.data.sources import worldstrat
from PIL import Image


def _png_bytes(size, value):
    array = np.full((size, size, 3), value, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _add(tar, name, payload):
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def test_iter_images_reads_rgb_png_and_skips_tiffs(tmp_path):
    archive = tmp_path / "hr_dataset.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        _add(tar, "AOI-1/AOI-1_rgb.png", _png_bytes(300, 120))
        _add(tar, "AOI-1/AOI-1_pan.tiff", b"not an image")
        _add(tar, "AOI-2/AOI-2_rgb.png", _png_bytes(300, 90))

    images = list(worldstrat.iter_images(archive))

    assert len(images) == 2
    assert {img.original_id for img in images} == {"AOI-1", "AOI-2"}
    assert images[0].array.shape == (300, 300, 3)
    assert images[0].array.dtype == np.uint8
    assert images[0].gsd_m == 1.5
