"""LoveDA source adapter.

Reads the local LoveDA extract (Train/Val, Rural/Urban, images_png) and yields the
1024 x 1024 RGB scenes. Masks are ignored since pretraining is self-supervised.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image

from dinov3_vhr.data.tiles import SourceImage

LICENSE = "CC-BY-NC-SA-4.0"
GSD_M = 0.3


def iter_images(root: Path) -> Iterator[SourceImage]:
    """Yield every LoveDA scene present under root as a SourceImage."""
    for split in ("Train", "Val"):
        for domain in ("Rural", "Urban"):
            image_dir = root / split / domain / "images_png"
            if not image_dir.is_dir():
                continue
            for path in sorted(image_dir.glob("*.png"), key=lambda p: int(p.stem)):
                array = np.asarray(Image.open(path).convert("RGB"))
                yield SourceImage(
                    array=array,
                    source="loveda",
                    license=LICENSE,
                    gsd_m=GSD_M,
                    original_id=f"{split}_{domain}_{path.stem}",
                    extra={"split": split.lower(), "domain": domain.lower()},
                )
