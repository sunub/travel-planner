"""두 run의 저장된 예측을 같은 split에서 리뷰 단위로 비교한다 (모델 · GPU 불필요).

과업 성공률, 개선 · 퇴행 리뷰 수, paired bootstrap 95% 신뢰구간을 낸다.
  - 요약(리뷰 원문 없음, 커밋용): <experiments_root>/comparisons/<A>__vs__<B>_<split>.json
  - 리뷰별 사례(원문 포함, Git 제외): <artifacts_root>/comparisons/<A>__vs__<B>_<split>/cases.jsonl

사용법:
  uv run python training/scripts/compare_predictions.py --config training/configs/base_eval.yaml base_v001 qlora_gold_v001
  uv run python training/scripts/compare_predictions.py --config training/configs/base_eval_busan_v2.yaml base4bit_e2b_busan_v001 qlora_e2b_busan_silver_v001

--config는 데이터를 읽는 데만 쓴다. 두 run이 이 설정과 같은 split으로 평가됐는지 metrics.json에서 확인한다.
"""

import argparse
import sys

from travel_planner.finetune.config import load_config, resolve_path
from travel_planner.finetune.data.dataset import write_jsonl
from travel_planner.finetune.data.pipeline import load_splits
from travel_planner.finetune.evaluation.metrics import TASK_SUCCESS_MIN_OVERLAP
from travel_planner.finetune.evaluation.paired import align_outputs, compare, load_outputs
from travel_planner.finetune.inference.run_eval import evaluation_key
from travel_planner.finetune.utils.tracking import git_info, now_iso, read_json, write_json

SHOWN = [
    ("과업 성공률", "task_success_rate"), ("리뷰 완전 일치", "exact_review_match"), ("Aspect F1", "aspect_f1"),
    ("Aspect 정밀도", "aspect_precision"), ("Aspect 재현율", "aspect_recall"), ("Attribute 정확도", "attribute_accuracy"),
    ("Sentiment 정확도", "sentiment_accuracy"), ("Evidence 정확 일치", "evidence_exact_match"),
    ("Evidence 겹침 F1", "evidence_overlap_f1"), ("JSON 준수율", "json_valid_rate"),
]  # fmt: skip


def check_run(run: str, split: str, split_version: str | None, experiments_root) -> None:
    metrics_path = experiments_root / run / "metrics.json"
    if not metrics_path.exists():
        sys.exit(f"{metrics_path}가 없습니다.")
    metrics = read_json(metrics_path)
    if evaluation_key(split) not in metrics:
        sys.exit(f"{run}에는 {split} 평가 결과가 없습니다.")
    if metrics["data"].get("split_version") != split_version:
        sys.exit(f"{run}의 split({metrics['data'].get('split_version')})이 --config의 split({split_version})과 다릅니다.")


def fmt(value) -> str:
    return "-" if value is None else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="데이터를 읽을 설정 (예: base_eval.yaml)")
    parser.add_argument("run_a", help="기준 run (예: base_v001)")
    parser.add_argument("run_b", help="비교 run (예: qlora_gold_v001)")
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--resamples", type=int, default=2000, help="bootstrap 반복 수")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap 난수 seed")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    splits, data_info = load_splits(config)
    records = splits[args.split]
    experiments_root = resolve_path(config["output"]["experiments_root"])
    artifacts_root = resolve_path(config["output"]["artifacts_root"])
    for run in (args.run_a, args.run_b):
        check_run(run, args.split, data_info["split_version"], experiments_root)

    outputs = [
        align_outputs(records, load_outputs(artifacts_root / run / "predictions" / f"{args.split}.jsonl"), run)
        for run in (args.run_a, args.run_b)
    ]
    result = compare(records, *outputs, n_resamples=args.resamples, seed=args.seed)
    report = {
        "created_at": now_iso(),
        "git": git_info(),
        "run_a": args.run_a,
        "run_b": args.run_b,
        "split": args.split,
        "split_version": data_info["split_version"],
        "dataset_version": data_info["dataset_version"],
        "task_success_min_overlap": TASK_SUCCESS_MIN_OVERLAP,
        "bootstrap": {"resamples": args.resamples, "seed": args.seed},
        **result["report"],
    }

    name = f"{args.run_a}__vs__{args.run_b}_{args.split}"
    report_path = experiments_root / "comparisons" / f"{name}.json"
    cases_path = artifacts_root / "comparisons" / name / "cases.jsonl"
    write_json(report_path, report)
    write_jsonl(cases_path, result["cases"])

    a, b, boot = report["overall"]["a"], report["overall"]["b"], report["bootstrap"]
    print(f"{args.split} {len(records)}건 · A={args.run_a} · B={args.run_b} · evidence 겹침 기준 {TASK_SUCCESS_MIN_OVERLAP}\n")
    print(f"{'지표':<18}{'A':>8}{'B':>8}{'B-A':>9}   95% CI (bootstrap {args.resamples})")
    for label, key in SHOWN:
        ci = boot.get(key)
        ci_text = "-" if not ci or ci["ci_low"] is None else f"[{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]" + (" *" if ci["excludes_zero"] else "")
        diff = None if a[key] is None or b[key] is None else b[key] - a[key]
        print(f"{label:<18}{fmt(a[key]):>8}{fmt(b[key]):>8}{'-' if diff is None else f'{diff:+.3f}':>9}   {ci_text}")
    print("  * 신뢰구간이 0을 포함하지 않음\n")
    t = report["transitions"]
    print(f"리뷰별 과업 성공: 개선 {t['improved']['count']} · 퇴행 {t['regressed']['count']} · 둘 다 성공 {t['both_success']['count']} · 둘 다 실패 {t['both_fail']['count']}")
    print(f"\n요약: {report_path}\n사례: {cases_path} (리뷰 원문 포함, Git 제외)")


if __name__ == "__main__":
    main()
