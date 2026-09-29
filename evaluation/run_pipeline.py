"""Run inference, automatic metrics, and human-review report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import (
    build_human_review_html,
    build_human_rows,
    build_report_html,
    evaluate_prediction_files,
    load_config,
    read_jsonl,
    run_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TripFit 5-model evaluation pipeline")
    parser.add_argument("--config", type=Path, required=True, help="모델 5개와 Gold 경로를 담은 JSON 설정")
    parser.add_argument("--out", type=Path, required=True, help="결과 디렉터리")
    parser.add_argument("--skip-inference", action="store_true", help="기존 predictions/*.jsonl 재사용")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = args.out
    prediction_dir = output_dir / "predictions"
    prediction_paths: dict[str, Path] = {}
    gold_path = Path(config["gold"])
    gold_rows = read_jsonl(gold_path)

    inference_log = []
    for model in config["models"]:
        path = prediction_dir / f"{model['name']}.jsonl"
        prediction_paths[model["name"]] = path
        if args.skip_inference and path.exists():
            inference_log.append({"model": model["name"], "status": "reused", "prediction_path": str(path)})
            continue
        inference_log.append(run_model(model, gold_path, path))

    metrics = evaluate_prediction_files(gold_rows, prediction_paths)
    metrics["config"] = str(args.config)
    metrics_path = output_dir / "automatic_metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    human_rows = build_human_rows(gold_rows, prediction_paths)
    human_path = output_dir / "human_review.html"
    human_path.write_text(build_human_review_html(human_rows, list(prediction_paths)), encoding="utf-8")

    report_data = dict(metrics)
    report_data["inference"] = inference_log
    report_path = output_dir / "report.html"
    report_path.write_text(build_report_html(report_data, human_path.name), encoding="utf-8")
    (output_dir / "run_manifest.json").write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"자동 평가 결과: {metrics_path}")
    print(f"최종 리포트: {report_path}")
    print(f"인간 평가: {human_path}")


if __name__ == "__main__":
    main()
