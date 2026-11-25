from dinov3_vhr.data.build_oam import dedup_by_footprint
from shapely.geometry import box


def test_dedup_keeps_finest_gsd_among_overlapping():
    coarse = {"id": "coarse", "gsd": 0.5, "geom": box(0, 0, 1, 1)}
    fine = {"id": "fine", "gsd": 0.1, "geom": box(0, 0, 1, 1)}
    survivors = dedup_by_footprint([coarse, fine], iou_threshold=0.7)
    assert len(survivors) == 1
    assert survivors[0]["id"] == "fine"


def test_dedup_keeps_disjoint_mosaics_separate():
    here = {"id": "here", "gsd": 0.5, "geom": box(0, 0, 1, 1)}
    far = {"id": "far", "gsd": 0.5, "geom": box(10, 10, 11, 11)}
    survivors = dedup_by_footprint([here, far], iou_threshold=0.7)
    assert len(survivors) == 2


def test_dedup_keeps_low_overlap_pair_separate():
    left = {"id": "left", "gsd": 0.5, "geom": box(0, 0, 1, 1)}
    shifted = {"id": "shifted", "gsd": 0.5, "geom": box(0.9, 0, 1.9, 1)}
    survivors = dedup_by_footprint([left, shifted], iou_threshold=0.7)
    assert len(survivors) == 2
