"""Collect pangaea eval metrics into one tidy CSV, preferring the structured test_metrics.json
each eval writes (e1_<encoder>_<dataset>.json) and falling back to log scraping when it is absent.

    uv run python -m dinov3_vhr.eval.collect_results \
        --log-dir ../pangaea-bench2.0/slurm/logs --out out/results/metrics.csv
"""

import argparse
import csv
import json
import re
from collections.abc import Sequence
from pathlib import Path

# dataset -> (metric, json_key) in the evaluator's metrics dict.
_SEG_J = [("mIoU", "mIoU")]
_CLS_J = [("mF1", "F1")]
_DET_J = [("map", "map"), ("map_50", "map_50"), ("map_75", "map_75")]
JSON_KEYS: dict[str, list[tuple[str, str]]] = {
    "flair2": _SEG_J,
    "spacenet2": _SEG_J,
    "spacenet3": _SEG_J,
    "hot_buildings": _SEG_J,
    "xview2": _SEG_J,
    "pureforest": _CLS_J,
    "mpv4ger": _CLS_J,
    "opencanopy": [("RMSE", "RMSE")],
    "nzcattle_det": _DET_J,
    "everwatch": _DET_J,
}

# dataset -> (metric, section_header, value_line) for the log fallback: first float on the first
# value_line at or after the header ("" = from the top). [test] avoids [val]; seg/cls match Mean.
_SEG = [("mIoU", r"\[test\] ------- IoU", r"\[test\] Mean")]
_CLS = [("mF1", r"\[test\] ------- F1-score", r"\[test\] Mean")]
_DET = [
    ("map", "", r"\[test\] map:"),
    ("map_50", "", r"\[test\] map_50:"),
    ("map_75", "", r"\[test\] map_75:"),
]
METRICS: dict[str, list[tuple[str, str, str]]] = {
    "flair2": _SEG,
    "spacenet2": _SEG,
    "spacenet3": _SEG,
    "hot_buildings": _SEG,
    "xview2": _SEG,
    "pureforest": _CLS,
    "mpv4ger": _CLS,
    "opencanopy": [("RMSE", r"\[test\] ------- MSE and RMSE", r"RMSE")],
    "nzcattle_det": _DET,
    "everwatch": _DET,
}
FLOAT = re.compile(r"[-+]?\d+\.\d+")
ENCODER = re.compile(r"^(?P<arm>.+)_s(?P<seed>\d+)_(?P<step>\d+)$")
FIELDS = ["encoder", "arm", "seed", "step", "dataset", "metric", "value"]


def extract_value(lines: Sequence[str], section: str, value_line: str) -> float | None:
    """First float on the first value_line at or after the section header."""
    section_re = re.compile(section) if section else None
    value_re = re.compile(value_line)
    in_section = section_re is None
    for line in lines:
        if section_re is not None and section_re.search(line):
            in_section = True
            continue
        if in_section and value_re.search(line):
            match = FLOAT.search(line)
            return float(match.group()) if match else None
    return None


def parse_name(path: Path) -> tuple[str, str] | None:
    """(encoder, dataset) from e1_<encoder>_<dataset>.log for a known dataset, else None."""
    body = path.stem.removeprefix("e1_")
    for dataset in METRICS:
        if body.endswith(f"_{dataset}"):
            return body[: -(len(dataset) + 1)], dataset
    return None


def make_row(encoder: str, dataset: str, metric: str, value: float) -> dict[str, object]:
    """One CSV row, splitting <arm>_s<seed>_<step> encoder names into columns when they match."""
    row: dict[str, object] = {
        "encoder": encoder,
        "dataset": dataset,
        "metric": metric,
        "value": value,
    }
    name = ENCODER.match(encoder)
    if name:
        row |= {
            "arm": name["arm"],
            "seed": int(name["seed"]),
            "step": int(name["step"]),
        }
    return row


def read_json(path: Path, dataset: str) -> list[tuple[str, float]]:
    """(metric, value) pairs from a test_metrics.json for a known dataset."""
    data = json.loads(path.read_text())
    return [(metric, float(data[key])) for metric, key in JSON_KEYS[dataset] if key in data]


def read_log(path: Path, dataset: str) -> list[tuple[str, float]]:
    """(metric, value) pairs scraped from an eval log for a known dataset."""
    lines = path.read_text(errors="ignore").splitlines()
    pairs = []
    for metric, section, value_line in METRICS[dataset]:
        value = extract_value(lines, section, value_line)
        if value is not None:
            pairs.append((metric, value))
    return pairs


def collect(log_dir: Path) -> list[dict[str, object]]:
    """One row per (encoder, dataset, metric), from the JSON where present else the log."""
    rows: list[dict[str, object]] = []
    from_json: set[tuple[str, str]] = set()
    for path in sorted(log_dir.glob("e1_*.json")):
        parsed = parse_name(path)
        if parsed is None:
            continue
        encoder, dataset = parsed
        rows.extend(make_row(encoder, dataset, m, v) for m, v in read_json(path, dataset))
        from_json.add((encoder, dataset))
    for path in sorted(log_dir.glob("e1_*.log")):
        parsed = parse_name(path)
        if parsed is None or parsed in from_json:
            continue
        encoder, dataset = parsed
        rows.extend(make_row(encoder, dataset, m, v) for m, v in read_log(path, dataset))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("out/results/metrics.csv"))
    args = parser.parse_args()

    rows = collect(args.log_dir)
    if not rows:
        raise FileNotFoundError(f"no eval logs with metrics under {args.log_dir}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
