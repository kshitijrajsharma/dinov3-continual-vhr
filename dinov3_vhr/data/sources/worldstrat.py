"""WorldStrat source adapter.

Reads the shipped pansharpened RGB scene (_rgb.png, 1054 x 1054 at 1.5 m) for each AOI
from the WorldStrat hr_dataset, given either the downloaded hr_dataset.tar.gz or an
extracted directory. The pan, ps, and rgbn tiffs are ignored.
"""

import io
import tarfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image

from dinov3_vhr.data.tiles import SourceImage

LICENSE = "CC-BY-NC-4.0"
GSD_M = 1.5


def _source_image(aoi: str, data: bytes) -> SourceImage:
    array = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    return SourceImage(
        array=array,
        source="worldstrat",
        license=LICENSE,
        gsd_m=GSD_M,
        original_id=aoi,
        extra={"aoi": aoi},
    )


def iter_images(root: Path) -> Iterator[SourceImage]:
    """Yield each WorldStrat AOI's pansharpened RGB scene as a SourceImage."""
    if root.is_file() and root.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(root, "r:gz") as tar:
            for member in tar:
                if not (member.isfile() and member.name.endswith("_rgb.png")):
                    continue
                extracted = tar.extractfile(member)
                if extracted is not None:
                    yield _source_image(member.name.split("/")[0], extracted.read())
    else:
        for path in sorted(root.rglob("*_rgb.png")):
            yield _source_image(path.parent.name, path.read_bytes())
