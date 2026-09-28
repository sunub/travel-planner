"""Base 모델 또는 학습한 어댑터를 Gold Test(또는 --split)에서 평가하고 metrics.json을 쓴다.

사용법:
  # Base (학습 전 기준선)
  uv run python training/scripts/evaluate.py --config training/configs/base_eval.yaml

  # 학습한 어댑터를 그 run에 평가 결과로 더하기
  uv run python training/scripts/evaluate.py --config training/configs/qlora_gold_v1.yaml \\
      --adapter training/artifacts/qlora_gold_v001/adapter --run-name qlora_gold_v001

  # 어디서 받은 어댑터든 새 run으로 평가 (base 모델 id는 어댑터의 tripfit_adapter.json에서 읽는다)
  uv run python training/scripts/evaluate.py --config training/configs/lora_gold_v1.yaml --adapter D:/adapters/x
"""

import argparse
from pathlib import Path

from travel_planner.finetune.config import load_config
from travel_planner.finetune.inference.run_eval import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", type=Path, help="LoRA/QLoRA 어댑터 폴더. 없으면 Base 모델을 평가한다")
    parser.add_argument("--run-name", help="결과를 더할 기존 run 이름 (없으면 새 run)")
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--model", help="model.name_or_path 덮어쓰기")
    parser.add_argument("--artifacts-root", help="예측 결과를 저장할 곳 (Git 제외)")
    parser.add_argument("--max-samples", type=int, help="split마다 앞에서 N건만 (스모크 실행)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    overrides = list(args.set)
    if args.model:
        overrides.append(f"model.name_or_path={args.model}")
    if args.artifacts_root:
        overrides.append(f"output.artifacts_root={args.artifacts_root}")
    if args.max_samples:
        overrides.append(f"data.max_samples={args.max_samples}")
    path = run_evaluation(load_config(args.config, overrides), adapter_path=args.adapter, run_name=args.run_name, split=args.split)
    print(f"완료: {path}")


if __name__ == "__main__":
    main()
