"""Base 모델 또는 LoRA/QLoRA 어댑터를 같은 조건(그리디 디코딩, 같은 max_new_tokens)으로 평가한다.

  --adapter 없음: config.yaml의 원본 모델을 그대로 불러와 Base(bf16, 양자화 없음)로 평가한다.
  --adapter 있음: --method(qlora|lora)로 학습 때와 같은 정밀도로 원본 모델을 올린 뒤 어댑터를 얹어
    평가한다 (qlora=4bit NF4, lora=bf16). train.py가 학습 끝에 저장하는 adapter/ 폴더뿐 아니라,
    epoch마다 자동 저장되는 checkpoints/checkpoint-N 폴더도 그대로 받는다.

predictions/<run>_<split>.jsonl과 metrics_<run>_<split>.json을 남긴다 — --adapter가 있으면
config.yaml의 output.artifacts_root/<method> 아래(train.py가 그 방식의 checkpoint·adapter를 남기는
곳과 같은 폴더), --adapter가 없으면(Base) output.artifacts_root 바로 아래(Base는 method와 무관하므로).
(run은 "base" 또는 어댑터/checkpoint 폴더 이름 — "adapter", "checkpoint-2" 등. 서로 다른
checkpoint·split 평가 결과가 덮어쓰지 않게 파일명에 넣는다.)

채점 로직(라벨 검사·JSON 파싱·지표 계산)은 팀원 브랜치(origin/lkh_train)의
`src/travel_planner/finetune/{labels.py,evaluation/parser.py,evaluation/metrics.py,evaluation/evaluator.py}`를
그대로 옮긴 것이다(`git show origin/lkh_train:<경로>`로 읽기만 했다. merge·checkout 없음). 이 함수들은
원래도 모델과 무관한 순수 함수라 Gemma용 코드를 그대로 재사용할 수 있었다. 지표 정의는 그 파일들의
docstring과 동일하다 (팀원 docs/evaluation-plan.md 2절 기준). 같은 지표를 써야 Base/LoRA/QLoRA뿐 아니라
팀원의 Gemma·Qwen 결과와도 비교할 수 있다.

사용법:
  # Base 모델 평가
  uv run python src/travel_planner/model-yay/evaluate.py --config src/travel_planner/model-yay/config.yaml

  # QLoRA 어댑터 평가
  uv run python src/travel_planner/model-yay/evaluate.py --config src/travel_planner/model-yay/config.yaml \\
      --method qlora --adapter src/travel_planner/model-yay/artifacts/qlora/adapter

  # LoRA 어댑터 평가
  uv run python src/travel_planner/model-yay/evaluate.py --config src/travel_planner/model-yay/config.yaml \\
      --method lora --adapter src/travel_planner/model-yay/artifacts/lora/adapter
"""

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv

from data import ATTRIBUTES, SENTIMENTS, TRAVELER_CONTEXTS, format_hm, load_records, prompt_text

MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODEL_DIR.parents[2]
ASPECT_FIELDS = ("category", "attribute", "sentiment", "evidence")


# =====================================================================================
# 라벨 검사 (팀원 labels.py의 aspect_issues/label_issues를 그대로 옮김. schema.py 허용값 기준)
# =====================================================================================

IssueKind = Literal["schema", "value", "evidence"]


@dataclass(frozen=True)
class Issue:
    kind: IssueKind
    message: str


def aspect_issues(aspect: object, category: str, review: str) -> list[Issue]:
    if not isinstance(aspect, dict):
        return [Issue("schema", "aspect가 객체가 아님")]
    missing = [f for f in ASPECT_FIELDS if f not in aspect]
    if missing:
        return [Issue("schema", f"aspect 필드 누락: {missing}")]
    if not all(isinstance(aspect[f], str) for f in ASPECT_FIELDS):
        return [Issue("schema", "aspect 필드 값은 모두 문자열이어야 함")]

    name, issues = aspect["category"], []
    allowed = ATTRIBUTES.get(category, {})
    if name not in allowed:
        issues.append(Issue("value", f"{category}에 없는 aspect: {name!r}"))
    elif aspect["attribute"] not in allowed[name]:
        issues.append(Issue("value", f"{name}: 허용되지 않은 attribute {aspect['attribute']!r}"))
    if aspect["sentiment"] not in SENTIMENTS:
        issues.append(Issue("value", f"{name}: 허용되지 않은 sentiment {aspect['sentiment']!r}"))
    evidence = aspect["evidence"]
    if not evidence.strip():
        issues.append(Issue("evidence", f"{name}: evidence가 비어 있음"))
    elif evidence not in review:
        issues.append(Issue("evidence", f"{name}: 리뷰 원문에 없는 evidence {evidence!r}"))
    return issues


def label_issues(label: object, category: str, review: str) -> list[Issue]:
    if category not in ATTRIBUTES:
        return [Issue("schema", f"알 수 없는 카테고리: {category!r}")]
    if not isinstance(label, dict):
        return [Issue("schema", "label이 JSON 객체가 아님")]
    contexts, aspects = label.get("traveler_context"), label.get("aspects")
    if not isinstance(contexts, list) or not isinstance(aspects, list):
        return [Issue("schema", "traveler_context와 aspects는 리스트여야 함")]

    issues = [
        Issue("value", f"허용되지 않은 traveler_context: {c!r}")
        for c in contexts
        if not isinstance(c, str) or c not in TRAVELER_CONTEXTS
    ]
    for aspect in aspects:
        issues += aspect_issues(aspect, category, review)
    return issues


# =====================================================================================
# 출력 문자열 -> 예측 라벨 (팀원 evaluation/parser.py를 그대로 옮김)
# =====================================================================================


@dataclass
class ParsedPrediction:
    raw: str
    parsed: bool
    data: dict | None
    parse_error: str | None = None
    issues: list[Issue] = field(default_factory=list)

    @property
    def schema_valid(self) -> bool:
        return self.parsed and not any(issue.kind == "schema" for issue in self.issues)

    @property
    def strict_valid(self) -> bool:
        return self.schema_valid and not any(issue.kind == "value" for issue in self.issues)

    def label_for_scoring(self) -> dict:
        empty = {"traveler_context": [], "aspects": []}
        if not self.parsed or not isinstance(self.data, dict):
            return empty
        contexts = self.data.get("traveler_context")
        aspects = self.data.get("aspects")
        return {
            "traveler_context": [c for c in contexts if isinstance(c, str)] if isinstance(contexts, list) else [],
            "aspects": [
                {f: a[f] for f in ASPECT_FIELDS}
                for a in (aspects if isinstance(aspects, list) else [])
                if isinstance(a, dict) and all(isinstance(a.get(f), str) for f in ASPECT_FIELDS)
            ],
        }


def extract_json_object(text: str) -> tuple[dict | None, str | None]:
    decoder = json.JSONDecoder()
    error = "출력에 JSON 객체가 없음"
    start = text.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError as e:
            error = f"JSON 파싱 실패: {e.msg}"
        else:
            if isinstance(value, dict):
                return value, None
            error = "JSON 객체가 아님"
        start = text.find("{", start + 1)
    return None, error


def parse_prediction(raw: str, category: str, review: str) -> ParsedPrediction:
    data, error = extract_json_object(raw)
    if data is None:
        return ParsedPrediction(raw=raw, parsed=False, data=None, parse_error=error)
    return ParsedPrediction(raw=raw, parsed=True, data=data, issues=label_issues(data, category, review))


# =====================================================================================
# 채점 지표 (팀원 evaluation/metrics.py를 그대로 옮김)
# =====================================================================================


def safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def f1_score(tp: float, n_pred: float, n_gold: float) -> dict[str, float | None]:
    precision, recall = safe_div(tp, n_pred), safe_div(tp, n_gold)
    if precision is None or recall is None:
        f1 = None if not n_pred and not n_gold else 0.0
    else:
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _occurrences(text: str, span: str) -> list[int]:
    starts, start = [], text.find(span)
    while span and start != -1:
        starts.append(start)
        start = text.find(span, start + 1)
    return starts


def evidence_overlap_f1(pred: str, gold: str, review: str) -> float:
    pred_starts, gold_starts = _occurrences(review, pred), _occurrences(review, gold)
    if not pred_starts or not gold_starts:
        return 1.0 if pred and pred == gold else 0.0
    best = 0
    for p in pred_starts:
        for g in gold_starts:
            best = max(best, min(p + len(pred), g + len(gold)) - max(p, g))
    return 2 * best / (len(pred) + len(gold))


def match_aspects(pred: list[dict], gold: list[dict], review: str) -> list[tuple[int, int]]:
    candidates = [
        (-evidence_overlap_f1(p["evidence"], g["evidence"], review), i, j)
        for i, p in enumerate(pred)
        for j, g in enumerate(gold)
        if p["category"] == g["category"]
    ]
    used_pred, used_gold, pairs = set(), set(), []
    for _, i, j in sorted(candidates):
        if i not in used_pred and j not in used_gold:
            used_pred.add(i)
            used_gold.add(j)
            pairs.append((i, j))
    return sorted(pairs)


def _aspect_key(aspect: dict) -> tuple[str, str, str, str]:
    return aspect["category"], aspect["attribute"], aspect["sentiment"], aspect["evidence"]


@dataclass
class ReviewScore:
    category: str
    n_pred: int
    n_gold: int
    pairs: list[dict]
    pred_in_source: int
    set_tp: int
    set_pred: int
    set_gold: int
    context_tp: int
    context_pred: int
    context_gold: int
    exact_match: bool
    gold_empty: bool
    pred_empty: bool
    per_aspect: Counter = field(default_factory=Counter)


def score_review(pred_label: dict, gold_label: dict, review: str, category: str) -> ReviewScore:
    pred, gold = pred_label["aspects"], gold_label["aspects"]
    pairs = []
    for i, j in match_aspects(pred, gold, review):
        p, g = pred[i], gold[j]
        pairs.append(
            {
                "aspect": g["category"],
                "attribute_ok": p["attribute"] == g["attribute"],
                "sentiment_ok": p["sentiment"] == g["sentiment"],
                "gold_sentiment": g["sentiment"],
                "pred_sentiment": p["sentiment"],
                "exact": p["evidence"] == g["evidence"],
                "overlap": evidence_overlap_f1(p["evidence"], g["evidence"], review),
            }
        )

    per_aspect: Counter = Counter()
    for a in pred:
        per_aspect[f"{category}/{a['category']}", "pred"] += 1
    for a in gold:
        per_aspect[f"{category}/{a['category']}", "gold"] += 1
    for pair in pairs:
        per_aspect[f"{category}/{pair['aspect']}", "tp"] += 1

    pred_set, gold_set = {a["category"] for a in pred}, {a["category"] for a in gold}
    pred_ctx, gold_ctx = set(pred_label["traveler_context"]), set(gold_label["traveler_context"])
    return ReviewScore(
        category=category,
        n_pred=len(pred),
        n_gold=len(gold),
        pairs=pairs,
        pred_in_source=sum(1 for a in pred if a["evidence"] and a["evidence"] in review),
        set_tp=len(pred_set & gold_set),
        set_pred=len(pred_set),
        set_gold=len(gold_set),
        context_tp=len(pred_ctx & gold_ctx),
        context_pred=len(pred_ctx),
        context_gold=len(gold_ctx),
        exact_match=Counter(map(_aspect_key, pred)) == Counter(map(_aspect_key, gold)) and pred_ctx == gold_ctx,
        gold_empty=not gold,
        pred_empty=not pred,
        per_aspect=per_aspect,
    )


def _macro_f1(tp: Counter, n_pred: Counter, n_gold: Counter) -> float | None:
    classes = sorted(set(n_pred) | set(n_gold))
    scores = [f1_score(tp[c], n_pred[c], n_gold[c])["f1"] or 0.0 for c in classes]
    return safe_div(sum(scores), len(scores))


def aggregate(scores: list[ReviewScore]) -> dict:
    pairs = [pair for s in scores for pair in s.pairs]
    tp, n_pred, n_gold = len(pairs), sum(s.n_pred for s in scores), sum(s.n_gold for s in scores)
    aspect = f1_score(tp, n_pred, n_gold)

    per_aspect: Counter = sum((s.per_aspect for s in scores), Counter())
    kinds = {key for key, _ in per_aspect}
    aspect_macro = _macro_f1(
        Counter({k: per_aspect[k, "tp"] for k in kinds}),
        Counter({k: per_aspect[k, "pred"] for k in kinds if per_aspect[k, "pred"]}),
        Counter({k: per_aspect[k, "gold"] for k in kinds if per_aspect[k, "gold"]}),
    )
    sentiment_macro = _macro_f1(
        Counter(p["gold_sentiment"] for p in pairs if p["sentiment_ok"]),
        Counter(p["pred_sentiment"] for p in pairs),
        Counter(p["gold_sentiment"] for p in pairs),
    )
    gold_empty = [s for s in scores if s.gold_empty]
    return {
        "reviews": len(scores),
        "gold_aspects": n_gold,
        "pred_aspects": n_pred,
        "matched_aspects": tp,
        "aspect_precision": aspect["precision"],
        "aspect_recall": aspect["recall"],
        "aspect_f1": aspect["f1"],
        "aspect_macro_f1": aspect_macro,
        "aspect_set_f1": f1_score(
            sum(s.set_tp for s in scores), sum(s.set_pred for s in scores), sum(s.set_gold for s in scores)
        )["f1"],
        "attribute_accuracy": safe_div(sum(p["attribute_ok"] for p in pairs), tp),
        "sentiment_accuracy": safe_div(sum(p["sentiment_ok"] for p in pairs), tp),
        "sentiment_macro_f1": sentiment_macro,
        "evidence_in_source_rate": safe_div(sum(s.pred_in_source for s in scores), n_pred),
        "evidence_exact_match": safe_div(sum(p["exact"] for p in pairs), tp),
        "evidence_overlap_f1": safe_div(sum(p["overlap"] for p in pairs), tp),
        "traveler_context_f1": f1_score(
            sum(s.context_tp for s in scores), sum(s.context_pred for s in scores), sum(s.context_gold for s in scores)
        )["f1"],
        "exact_review_match": safe_div(sum(s.exact_match for s in scores), len(scores)),
        "empty_review_accuracy": safe_div(sum(s.pred_empty for s in gold_empty), len(gold_empty)),
    }


# =====================================================================================
# 정답 + 출력 -> 지표 (팀원 evaluation/evaluator.py를 그대로 옮김)
# =====================================================================================


def format_metrics(parsed: list[ParsedPrediction]) -> dict:
    n = len(parsed)
    pred_aspects = sum(
        len(p.data.get("aspects", [])) for p in parsed if p.parsed and isinstance(p.data.get("aspects"), list)
    )
    value_issues = sum(1 for p in parsed for issue in p.issues if issue.kind == "value")
    return {
        "json_parse_rate": safe_div(sum(p.parsed for p in parsed), n),
        "schema_valid_rate": safe_div(sum(p.schema_valid for p in parsed), n),
        "json_valid_rate": safe_div(sum(p.strict_valid for p in parsed), n),
        "invalid_value_count": value_issues,
        "invalid_value_rate": safe_div(value_issues, pred_aspects),
    }


def evaluate_predictions(records: list[dict], outputs: list[str]) -> dict:
    """records[i]의 정답과 outputs[i]를 비교한다. {"overall", "by_category", "samples"}를 돌려준다."""
    if len(records) != len(outputs):
        raise ValueError(f"레코드 {len(records)}건과 출력 {len(outputs)}건의 수가 다릅니다")

    parsed = [parse_prediction(out, r["category"], r["review"]) for r, out in zip(records, outputs)]
    scores = [
        score_review(p.label_for_scoring(), r["label"], r["review"], r["category"]) for r, p in zip(records, parsed)
    ]

    def summary(indices: list[int]) -> dict:
        return {**format_metrics([parsed[i] for i in indices]), **aggregate([scores[i] for i in indices])}

    categories = sorted({r["category"] for r in records})
    samples = [
        {
            "review_id": r["review_id"],
            "category": r["category"],
            "raw_output": p.raw,
            "parsed": p.parsed,
            "parse_error": p.parse_error,
            "issues": [f"{issue.kind}: {issue.message}" for issue in p.issues],
            "prediction": p.label_for_scoring(),
            "gold": r["label"],
            "exact_match": s.exact_match,
        }
        for r, p, s in zip(records, parsed, scores)
    ]
    return {
        "overall": summary(list(range(len(records)))),
        "by_category": {c: summary([i for i, r in enumerate(records) if r["category"] == c]) for c in categories},
        "samples": samples,
    }


# =====================================================================================
# 추론 (팀원 inference/generate.py를 그대로 옮김. Gemma 전용 코드가 아니라 그대로 재사용 가능)
# =====================================================================================


def generate_outputs(
    model,
    tokenizer,
    records: list[dict],
    *,
    max_new_tokens: int,
    batch_size: int,
    estimate_for: dict[str, int] | None = None,
) -> tuple[list[str], dict]:
    """greedy decoding (do_sample=False)이라 같은 입력에는 매번 같은 답이 나온다. (출력, 속도 통계).

    estimate_for={"validation": 371, "test": 172}처럼 넘기면, 첫 배치를 처리한 뒤 그때까지의
    리뷰 1건당 평균 시간으로 각 split을 통째로 평가하면 얼마나 걸릴지 한 번만 출력한다.

    prompt 문자열은 data.prompt_text()로 만든다 — train.py의 encode_example()이 학습 prompt를 만들 때
    쓰는 것과 같은 함수다. 학습 때 본 prompt와 추론 때 넣는 prompt가 토큰 단위로 어긋나면 안 되므로,
    "비슷하게 두 번 짜는" 대신 함수 하나를 같이 쓴다.
    """
    import torch

    model.eval()
    if getattr(model, "is_gradient_checkpointing", False):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    tokenizer.padding_side = "left"

    prompts = [prompt_text(r, tokenizer) for r in records]
    outputs, generated_tokens, reviews_done = [], 0, 0
    estimate_printed = False
    start = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(prompts), batch_size):
            batch = tokenizer(prompts[i : i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
            batch = batch.to(model.device)
            generated = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = generated[:, batch["input_ids"].shape[1] :]
            generated_tokens += int((new_tokens != tokenizer.pad_token_id).sum())
            outputs += tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            reviews_done += batch["input_ids"].shape[0]
            print(f"  생성 {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
            if estimate_for and not estimate_printed:
                sec_per_review = (time.perf_counter() - start) / reviews_done
                estimate_printed = True
                print(f"  --- 예상 소요 시간 (첫 {reviews_done}건 기준 리뷰당 {sec_per_review:.2f}초) ---")
                for name, count in estimate_for.items():
                    print(f"    {name} {count}건 평가: 약 {format_hm(sec_per_review * count)}")
    duration = time.perf_counter() - start
    return outputs, {
        "duration_sec": round(duration, 2),
        "sec_per_review": round(duration / max(len(prompts), 1), 3),
        "generated_tokens": generated_tokens,
        "tokens_per_sec": round(generated_tokens / duration, 2) if duration else None,
    }


# =====================================================================================
# 모델 로드 및 실행
# =====================================================================================


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def resolve_output_dir(config: dict, method: str | None = None) -> Path:
    """method가 있으면 artifacts_root/<method>(train.py가 그 방식의 checkpoint·adapter를 남기는 곳과
    같은 폴더)를, 없으면(Base 평가) artifacts_root를 그대로 쓴다."""
    path = Path(config["output"]["artifacts_root"])
    if method:
        path = path / method
    return path if path.is_absolute() else REPO_ROOT / path


def load_model_for_eval(config: dict, hf_token: str | None, adapter_path: Path | None, method: str | None):
    """--adapter 없음: Base(bf16, 양자화 없음)로 로드. --adapter 있음: method로 학습 때와 같은 정밀도로
    올리고(qlora=4bit NF4, lora=bf16) 어댑터를 얹는다.

    EXAONE 4.0은 transformers에 내장돼 있어(train.py의 load_tokenizer_and_model 참고)
    trust_remote_code나 로컬 패치 없이 AutoModelForCausalLM/AutoTokenizer로 바로 불러온다.

    adapter_path는 train.py가 학습 끝에 저장하는 최종 adapter/ 폴더뿐 아니라, HF Trainer가 epoch마다
    자동으로 남기는 checkpoints/checkpoint-N 폴더도 그대로 받는다 (adapter_config.json +
    adapter_model.safetensors만 있으면 PeftModel.from_pretrained가 그걸로 충분하다). tokenizer는
    checkpoint-N 폴더에 있을 수도, 없을 수도 있어서 항상 원본 모델 id에서 불러온다 — LoRA는 tokenizer를
    바꾸지 않으므로 이래도 결과가 같다.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_cfg = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if adapter_path is not None and method == "qlora":
        quant = config["quantization"]
        compute_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[quant.get("bnb_4bit_compute_dtype", "bf16")]
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=quant["load_in_4bit"],
            bnb_4bit_quant_type=quant.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        revision=model_cfg.get("revision"),
        dtype=torch.bfloat16 if quantization_config is None else None,
        quantization_config=quantization_config,
        attn_implementation=model_cfg.get("attn_implementation"),
        device_map={"": 0},
        token=hf_token,
    )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
    return tokenizer, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--adapter",
        type=Path,
        help="LoRA/QLoRA 어댑터 폴더. 없으면 Base 모델을 평가한다. 최종 adapter/ 폴더뿐 아니라 "
        "checkpoints/checkpoint-N(epoch마다 자동 저장된 것)도 그대로 넣을 수 있다",
    )
    parser.add_argument(
        "--method",
        choices=["qlora", "lora"],
        help="--adapter를 평가할 때 필수. 그 어댑터를 학습할 때와 같은 정밀도로 base를 올린다 "
        "(qlora=4bit NF4, lora=bf16). --adapter 없이 Base만 평가할 때는 쓰지 않는다(Base는 항상 bf16)",
    )
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--max-samples", type=int, help="앞에서 N건만 (스모크 실행)")
    args = parser.parse_args()

    if args.adapter and not args.method:
        raise SystemExit("--adapter를 평가하려면 --method(qlora 또는 lora)를 지정하세요 — 학습 때와 같은 정밀도로 base를 올려야 합니다.")

    load_dotenv(REPO_ROOT / ".env")

    config = load_config(args.config)
    path_key = f"{args.split}_path" if args.split == "validation" else "test_path"
    dataset_path = config["data"].get(path_key)
    if not dataset_path:
        raise SystemExit(f"config.yaml의 data.{path_key}가 비어 있습니다. 데이터셋 경로를 채우세요.")

    records = load_records(dataset_path)
    if args.max_samples:
        records = records[: args.max_samples]
    run_label = args.method.upper() if args.adapter else "Base"
    print(f"{args.split} {len(records)}건 · 모델 {config['model']['name_or_path']} · {run_label}")

    # validation·test 전체 건수 (--split, --max-samples와 무관하게). 첫 배치 처리 후 이 두 split을
    # 통째로 평가하면 걸릴 예상 시간을 보여주는 데 쓴다.
    estimate_for = {
        "validation": len(load_records(config["data"]["validation_path"])),
        "test": len(load_records(config["data"]["test_path"])),
    }

    import os

    hf_token = os.environ.get("HF_TOKEN") or None

    tokenizer, model = load_model_for_eval(config, hf_token, args.adapter, args.method)
    inference = config["inference"]
    outputs, stats = generate_outputs(
        model,
        tokenizer,
        records,
        max_new_tokens=inference["max_new_tokens"],
        batch_size=inference["batch_size"],
        estimate_for=estimate_for,
    )
    result = evaluate_predictions(records, outputs)

    method = args.method if args.adapter else "base"
    # 어댑터/checkpoint 폴더 이름(예: "adapter", "checkpoint-2")까지 파일명에 넣는다. method만으로는
    # checkpoint-1과 checkpoint-2 평가 결과가 서로 덮어써서, 여러 checkpoint를 돌아가며 평가하는
    # select_checkpoint.py 같은 용도에서 결과가 섞인다.
    run_name = args.adapter.name if args.adapter else "base"
    # --adapter가 있으면 그 method의 artifacts 폴더(train.py가 쓰는 곳)에, 없으면(Base) artifacts_root
    # 바로 아래에 남긴다.
    output_dir = resolve_output_dir(config, args.method if args.adapter else None)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = predictions_dir / f"{run_name}_{args.split}.jsonl"
    with open(predictions_path, "w", encoding="utf-8") as f:
        for sample in result["samples"]:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    metrics = {
        "model_id": config["model"]["name_or_path"],
        "method": method,
        "adapter_path": str(args.adapter) if args.adapter else None,
        "split": args.split,
        "num_samples": len(records),
        "decoding": {"do_sample": False, "max_new_tokens": inference["max_new_tokens"], "batch_size": inference["batch_size"]},
        "overall": result["overall"],
        "by_category": result["by_category"],
        "inference": stats,
        "predictions_path": str(predictions_path),
    }
    metrics_path = output_dir / f"metrics_{run_name}_{args.split}.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"완료: {metrics_path}")


if __name__ == "__main__":
    main()
