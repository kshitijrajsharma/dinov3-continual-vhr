"""RAMP source adapter.

Reads the local RAMP extract (one directory per city, each with a source/ folder of
256 x 256 RGB GeoTIFF chips) and yields each chip as a SourceImage. The Paris and
Shanghai subsets are SpaceNet-2 imagery that overlaps the evaluation suite and are
skipped so they never enter pretraining.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import rasterio

from dinov3_vhr.data.tiles import SourceImage

LICENSE = "CC-BY-NC-4.0"
GSD_M = 0.3
HELDOUT_SUBSETS = frozenset({"ramp_paris_france", "ramp_shanghai_china"})


def iter_images(root: Path) -> Iterator[SourceImage]:
    """Yield every RAMP chip under root, skipping the evaluation-overlap subsets."""
    for subset_dir in sorted(root.glob("ramp_*")):
        if not subset_dir.is_dir() or subset_dir.name in HELDOUT_SUBSETS:
            continue
        country = subset_dir.name.split("_", 1)[1]
        for path in sorted((subset_dir / "source").glob("*.tif")):
            with rasterio.open(path) as src:
                array = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8)
            yield SourceImage(
                array=array,
                source="ramp",
                license=LICENSE,
                gsd_m=GSD_M,
                original_id=f"{subset_dir.name}_{path.stem}",
                extra={"subset": subset_dir.name, "country": country},
            )
