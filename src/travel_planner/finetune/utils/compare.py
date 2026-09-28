"""여러 run의 metrics.json → 비교표 (터미널 표 + CSV)."""

import csv
from pathlib import Path

from .tracking import read_json

# (열 이름, metrics.json 안의 위치, 표시 형식)
COLUMNS: list[tuple[str, tuple[str, ...], str]] = [
    ("experiment", ("run_name",), "s"),
    ("method", ("method",), "s"),
    ("data", ("data", "dataset_version"), "s"),
    ("split", ("data", "split_version"), "s"),
    ("n_test", ("evaluation", "num_samples"), "s"),
    ("Aspect F1", ("evaluation", "overall", "aspect_f1"), ".4f"),
    ("Aspect macro F1", ("evaluation", "overall", "aspect_macro_f1"), ".4f"),
    ("Attribute Acc", ("evaluation", "overall", "attribute_accuracy"), ".4f"),
    ("Sentiment Acc", ("evaluation", "overall", "sentiment_accuracy"), ".4f"),
    ("Evidence in-source", ("evaluation", "overall", "evidence_in_source_rate"), ".4f"),
    ("Evidence exact", ("evaluation", "overall", "evidence_exact_match"), ".4f"),
    ("Evidence overlap F1", ("evaluation", "overall", "evidence_overlap_f1"), ".4f"),
    ("JSON Valid", ("evaluation", "overall", "json_valid_rate"), ".4f"),
    ("Train VRAM (GB)", ("training", "peak_vram_reserved_gb"), ".2f"),
    ("Train Time (min)", ("training", "duration_sec"), "min"),
    ("Adapter Size (MB)", ("training", "adapter_size_mb"), ".1f"),
    ("Final Train Loss", ("training", "final_train_loss"), ".4f"),
    ("Eval Loss", ("training", "eval_loss"), ".4f"),
]


def find_metrics(paths: list[Path]) -> list[Path]:
    """파일은 그대로, 폴더는 그 안(한 단계 아래 포함)의 metrics.json을 찾는다."""
    found: list[Path] = []
    for path in paths:
        if path.is_file():
            found.append(path)
        elif (path / "metrics.json").exists():
            found.append(path / "metrics.json")
        elif path.is_dir():
            found += sorted(path.glob("*/metrics.json"))
    return found


def _lookup(data: dict, keys: tuple[str, ...]):
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def rows_from(metrics_paths: list[Path]) -> list[dict]:
    rows = []
    for path in metrics_paths:
        metrics = read_json(path)
        rows.append({name: _lookup(metrics, keys) for name, keys, _ in COLUMNS})
    return rows


def _format(value, fmt: str) -> str:
    if value is None:
        return "-"
    if fmt == "min":
        return f"{value / 60:.1f}"
    if fmt == "s":
        return str(value)
    return format(value, fmt)


def render_table(rows: list[dict], columns: list[str] | None = None) -> str:
    names = columns or [name for name, _, _ in COLUMNS]
    fmts = {name: fmt for name, _, fmt in COLUMNS}
    cells = [[_format(row[n], fmts[n]) for n in names] for row in rows]
    widths = [max(len(n), *(len(c[i]) for c in cells)) if cells else len(n) for i, n in enumerate(names)]

    def line(values: list[str]) -> str:
        return "| " + " | ".join(v.ljust(w) for v, w in zip(values, widths)) + " |"

    return "\n".join([line(names), "|" + "|".join("-" * (w + 2) for w in widths) + "|", *map(line, cells)])


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig: 엑셀에서 한글이 깨지지 않게
        writer = csv.DictWriter(f, fieldnames=[name for name, _, _ in COLUMNS])
        writer.writeheader()
        writer.writerows(rows)
