"""LoRA / QLoRA 학습. 설정 파일만 바꿔서 방식과 하이퍼파라미터를 고른다.

사용법:
  uv run python training/scripts/train.py --config training/configs/qlora_gold_v1.yaml --dry-run   # 모델 없이 데이터 확인
  uv run python training/scripts/train.py --config training/configs/qlora_gold_v1.yaml
  uv run python training/scripts/train.py --config training/configs/lora_gold_v1.yaml \\
      --model google/gemma-4-E4B-it --artifacts-root D:/tripfit_artifacts --set training.num_train_epochs=1
"""

import argparse

from travel_planner.finetune.config import load_config
from travel_planner.finetune.training.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", help="model.name_or_path 덮어쓰기 (Hugging Face model_id 또는 로컬 경로)")
    parser.add_argument("--artifacts-root", help="어댑터 · 체크포인트를 저장할 곳 (Git 제외)")
    parser.add_argument("--run-name", help="run 이름을 직접 정한다 (기본: <experiment_name>_v00N)")
    parser.add_argument("--max-samples", type=int, help="split마다 앞에서 N건만 (스모크 실행)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="설정 값 바꾸기 (여러 번 가능)")
    parser.add_argument("--dry-run", action="store_true", help="데이터 · prompt만 확인하고 모델은 불러오지 않는다")
    args = parser.parse_args()

    overrides = list(args.set)
    if args.model:
        overrides.append(f"model.name_or_path={args.model}")
    if args.artifacts_root:
        overrides.append(f"output.artifacts_root={args.artifacts_root}")
    if args.max_samples:
        overrides.append(f"data.max_samples={args.max_samples}")
    train(load_config(args.config, overrides), run_name=args.run_name, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
