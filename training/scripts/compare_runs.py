"""여러 run의 metrics.json을 한 표로 비교한다.

사용법:
  uv run python training/scripts/compare_runs.py                                   # training/experiments/* 전부
  uv run python training/scripts/compare_runs.py training/experiments/base_v001 training/experiments/qlora_gold_v001
  uv run python training/scripts/compare_runs.py --csv training/experiments/comparison.csv
"""

import argparse
import sys
from pathlib import Path

from travel_planner.finetune.config import REPO_ROOT
from travel_planner.finetune.utils.compare import COLUMNS, find_metrics, render_table, rows_from, write_csv

SHORT_COLUMNS = [
    "experiment", "method", "4bit", "Task Success", "Aspect F1", "Attribute Acc", "Sentiment Acc", "Evidence in-source", "Evidence exact",
    "JSON Valid", "Train VRAM (GB)", "Train Time (min)", "Adapter Size (MB)",
]  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path, help="metrics.json 파일 또는 run 폴더 (기본: training/experiments)")
    parser.add_argument("--csv", type=Path, help="모든 열을 CSV로 저장할 경로")
    parser.add_argument("--all-columns", action="store_true", help="터미널 표에도 모든 열을 보여준다")
    args = parser.parse_args()

    metrics_paths = find_metrics(args.paths or [REPO_ROOT / "training" / "experiments"])
    if not metrics_paths:
        sys.exit("metrics.json을 찾지 못했습니다.")
    rows = rows_from(metrics_paths)
    print(render_table(rows, None if args.all_columns else SHORT_COLUMNS))
    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nCSV 저장 ({len(COLUMNS)}열): {args.csv}")


if __name__ == "__main__":
    main()
