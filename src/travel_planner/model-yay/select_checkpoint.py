"""여러 QLoRA checkpoint를 validation으로 평가해서 best checkpoint를 고른다.

train.py는 epoch마다 checkpoint를 남긴다 (config.yaml의 training.save_total_limit: 3, 모든 epoch를
남긴다). 이 스크립트는 그 checkpoint들을 전부 validation으로 평가해서:
  - checkpoint별 eval_loss(학습 중 기록된 로그, checkpoint-N/trainer_state.json에서 읽는다)와
    validation Aspect F1 · 리뷰 완전 일치율 · JSON 준수율을 표로 보여준다.
  - validation Aspect F1이 가장 높은 checkpoint를 best로 골라 <output.artifacts_root>/best.json에 남긴다.

Test는 이 스크립트에서 절대 평가하지 않는다. best가 정해진 뒤에는 evaluate.py를 test로 딱 한 번만
따로 돌린다 (README.md의 "checkpoint 선택은 validation으로만, test는 best로 한 번만" 참고) — 그래야
test 점수를 보고 checkpoint를 고르는 일이 생기지 않는다.

채점(generate_outputs, evaluate_predictions)과 모델 로드(load_model_for_eval)는 evaluate.py의 함수를
그대로 가져다 쓴다 — checkpoint 평가와 단일 어댑터 평가가 서로 다른 코드로 갈라지면 결과가 미묘하게
달라질 수 있다.

사용법:
  uv run python src/travel_planner/model-yay/select_checkpoint.py --config src/travel_planner/model-yay/config.yaml
  uv run python src/travel_planner/model-yay/select_checkpoint.py --config src/travel_planner/model-yay/config.yaml \\
      --max-samples 8   # 스모크 실행
"""

import argparse
import gc
import json
import re
from pathlib import Path

from dotenv import load_dotenv

from data import load_records
from evaluate import evaluate_predictions, generate_outputs, load_config, load_model_for_eval, resolve_output_dir

MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODEL_DIR.parents[2]

METRIC_COLUMNS = ("eval_loss", "aspect_f1", "exact_review_match", "json_valid_rate")
COLUMN_LABELS = {
    "eval_loss": "eval_loss",
    "aspect_f1": "Aspect F1",
    "exact_review_match": "리뷰 완전 일치",
    "json_valid_rate": "JSON 준수율",
}


def find_checkpoints(checkpoints_dir: Path) -> list[Path]:
    """checkpoint-<N> 폴더를 step 번호 순으로 정렬해 돌려준다."""
    pattern = re.compile(r"^checkpoint-(\d+)$")
    found = [(int(m.group(1)), p) for p in checkpoints_dir.glob("checkpoint-*") if p.is_dir() and (m := pattern.match(p.name))]
    return [p for _, p in sorted(found)]


def read_checkpoint_eval_loss(checkpoint_dir: Path) -> float | None:
    """checkpoint 안의 trainer_state.json에서, 그 checkpoint 시점까지 기록된 마지막 eval_loss를 읽는다.

    HF Trainer는 checkpoint-N을 저장할 때 그 시점까지의 log_history를 담은 trainer_state.json을 같이
    남긴다. eval_strategy와 save_strategy가 둘 다 epoch라서, 이 checkpoint를 만든 epoch에서 막 평가한
    eval_loss가 log_history의 가장 마지막 "eval_loss" 항목이다.
    """
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.exists():
        return None
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for entry in reversed(state.get("log_history", [])):
        if "eval_loss" in entry:
            return entry["eval_loss"]
    return None


def evaluate_checkpoint(config: dict, checkpoint_dir: Path, records: list[dict], hf_token: str | None) -> dict:
    """checkpoint 하나를 validation records로 평가해 evaluate_predictions()의 "overall" 지표를 돌려준다."""
    import torch

    tokenizer, model = load_model_for_eval(config, hf_token, checkpoint_dir)
    inference = config["inference"]
    outputs, _stats = generate_outputs(
        model, tokenizer, records, max_new_tokens=inference["max_new_tokens"], batch_size=inference["batch_size"]
    )
    result = evaluate_predictions(records, outputs)

    del model  # 다음 checkpoint를 불러오기 전에 VRAM을 비운다
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result["overall"]


def render_table(rows: list[dict]) -> str:
    headers = ["checkpoint", *(COLUMN_LABELS[c] for c in METRIC_COLUMNS)]

    def fmt(value) -> str:
        return "-" if value is None else f"{value:.4f}"

    cells = [[row["checkpoint"], *(fmt(row[c]) for c in METRIC_COLUMNS)] for row in rows]
    widths = [max(len(h), *(len(c[i]) for c in cells)) if cells else len(h) for i, h in enumerate(headers)]

    def line(values: list[str]) -> str:
        return "| " + " | ".join(v.ljust(w) for v, w in zip(values, widths)) + " |"

    return "\n".join([line(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|", *map(line, cells)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints-dir", help="checkpoint-N 폴더들이 있는 위치. 기본값은 <output.artifacts_root>/checkpoints")
    parser.add_argument("--max-samples", type=int, help="validation 앞에서 N건만 (스모크 실행)")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    config = load_config(args.config)
    output_dir = resolve_output_dir(config)
    checkpoints_dir = Path(args.checkpoints_dir) if args.checkpoints_dir else output_dir / "checkpoints"
    checkpoints = find_checkpoints(checkpoints_dir)
    if not checkpoints:
        raise SystemExit(f"checkpoint를 찾지 못했습니다: {checkpoints_dir} (아직 학습이 끝나지 않았을 수 있습니다)")

    validation_path = config["data"].get("validation_path")
    if not validation_path:
        raise SystemExit("config.yaml의 data.validation_path가 비어 있습니다.")
    records = load_records(validation_path)
    if args.max_samples:
        records = records[: args.max_samples]
    print(f"checkpoint {len(checkpoints)}개, validation {len(records)}건으로 평가합니다.")

    import os

    hf_token = os.environ.get("HF_TOKEN") or None

    rows = []
    for checkpoint_dir in checkpoints:
        print(f"\n[{checkpoint_dir.name}] validation 평가 시작")
        eval_loss = read_checkpoint_eval_loss(checkpoint_dir)
        overall = evaluate_checkpoint(config, checkpoint_dir, records, hf_token)
        rows.append(
            {
                "checkpoint": checkpoint_dir.name,
                "checkpoint_path": str(checkpoint_dir),
                "eval_loss": eval_loss,
                "aspect_f1": overall.get("aspect_f1"),
                "exact_review_match": overall.get("exact_review_match"),
                "json_valid_rate": overall.get("json_valid_rate"),
            }
        )

    print("\n" + render_table(rows))

    scored = [row for row in rows if row["aspect_f1"] is not None]
    if not scored:
        raise SystemExit("validation Aspect F1을 계산한 checkpoint가 없습니다 (aspect가 하나도 없는 데이터일 수 있습니다).")
    best = max(scored, key=lambda row: row["aspect_f1"])

    best_record = {
        "selected_by": "validation_aspect_f1",
        "best_checkpoint": best["checkpoint"],
        "best_checkpoint_path": best["checkpoint_path"],
        "validation_num_samples": len(records),
        "candidates": rows,
        # test는 여기서 평가하지 않는다. best가 정해진 뒤 아래 명령으로 test를 한 번만 평가한다:
        #   uv run python evaluate.py --config <config> --adapter <best_checkpoint_path> --split test
        "note": "test는 이 스크립트에서 평가하지 않았다. best_checkpoint_path로 evaluate.py --split test를 한 번만 따로 돌린다.",
    }
    best_path = output_dir / "best.json"
    best_path.write_text(json.dumps(best_record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nbest checkpoint: {best['checkpoint']} (validation Aspect F1 {best['aspect_f1']:.4f})")
    print(f"기록: {best_path}")


if __name__ == "__main__":
    main()
