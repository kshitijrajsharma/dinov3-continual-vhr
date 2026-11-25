"""Unit tests for eval metric collection (JSON-first, log fallback) into the tidy CSV."""

import json
from pathlib import Path

from dinov3_vhr.eval.collect_results import (
    collect,
    extract_value,
    parse_name,
    read_json,
)

FLAIR2_LOG = """
INFO - [test] ------- IoU --------
[test]-------------------
[test] building     	 12.3
[test] Mean          	 49.998
INFO - [test] ------- F1-score --------
[test] Mean          	 64.863
INFO - Mean Accuracy: 69.392
"""

NZCATTLE_LOG = """
INFO - [val] map_50: 0.4782
INFO - [test] map: 0.1276
INFO - [test] map_50: 0.4275
INFO - [test] map_75: 0.0166
"""

OPENCANOPY_LOG = """
INFO - [val] ------- MSE and RMSE --------
RMSE 	 45.163
INFO - [test] ------- MSE and RMSE --------
[test]-------------------
RMSE 	 44.074
"""


def test_flair2_miou_is_mean_under_iou_not_per_class_or_later_section():
    val = extract_value(FLAIR2_LOG.splitlines(), r"\[test\] ------- IoU", r"\[test\] Mean")
    assert val == 49.998


def test_nzcattle_map50_is_test_not_val():
    val = extract_value(NZCATTLE_LOG.splitlines(), "", r"\[test\] map_50:")
    assert val == 0.4275


def test_opencanopy_rmse_is_test_section_not_val():
    val = extract_value(OPENCANOPY_LOG.splitlines(), r"\[test\] ------- MSE and RMSE", r"RMSE")
    assert val == 44.074


def test_missing_metric_returns_none():
    assert extract_value(["nothing here"], r"\[test\] ------- IoU", r"\[test\] Mean") is None


def test_parse_name_decomposes_known_dataset():
    assert parse_name(Path("e1_sw_control_s42_2000_flair2.log")) == (
        "sw_control_s42_2000",
        "flair2",
    )
    assert parse_name(Path("e1_sw_patch_only_s1234_8000_nzcattle_det.log")) == (
        "sw_patch_only_s1234_8000",
        "nzcattle_det",
    )
    assert parse_name(Path("e1_something_unknowndataset.log")) is None


def test_collect_rows_and_arm_seed_step(tmp_path):
    (tmp_path / "e1_sw_control_s42_2000_flair2.log").write_text(FLAIR2_LOG)
    (tmp_path / "e1_sw_patch_only_s1234_8000_nzcattle_det.log").write_text(NZCATTLE_LOG)
    rows = collect(tmp_path)
    flair = next(r for r in rows if r["dataset"] == "flair2")
    assert flair["metric"] == "mIoU" and flair["value"] == 49.998
    assert flair["arm"] == "sw_control" and flair["seed"] == 42 and flair["step"] == 2000
    nz = [r for r in rows if r["dataset"] == "nzcattle_det"]
    assert {r["metric"] for r in nz} == {"map", "map_50", "map_75"}


def test_read_json_extracts_primary_metrics(tmp_path):
    seg = tmp_path / "e1_vitl_gram0_s42_2000_flair2.json"
    seg.write_text(json.dumps({"IoU": [12.3, 88.0], "mIoU": 50.15}))
    assert read_json(seg, "flair2") == [("mIoU", 50.15)]
    det = tmp_path / "e1_x_nzcattle_det.json"
    det.write_text(json.dumps({"map": 0.13, "map_50": 0.43, "map_75": 0.02, "mar_100": 0.5}))
    assert dict(read_json(det, "nzcattle_det")) == {
        "map": 0.13,
        "map_50": 0.43,
        "map_75": 0.02,
    }


def test_collect_prefers_json_over_log(tmp_path):
    (tmp_path / "e1_vitl_gram0_s42_2000_flair2.log").write_text(FLAIR2_LOG)
    (tmp_path / "e1_vitl_gram0_s42_2000_flair2.json").write_text(json.dumps({"mIoU": 54.321}))
    flair = [r for r in collect(tmp_path) if r["dataset"] == "flair2"]
    assert len(flair) == 1
    assert flair[0]["value"] == 54.321 and flair[0]["metric"] == "mIoU"
