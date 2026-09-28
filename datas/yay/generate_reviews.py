"""로컬 Ollama로 합성 리뷰를 만들어 datas/yay/out/generated/reviews.jsonl에 저장한다.

리뷰마다 "생성 계획"(장소, traveler_context, aspects, 표현 방식, 말투, 길이)을
먼저 코드로 무작위 결정한 뒤, 그 계획대로 쓰라고 Ollama에 요청한다.
라벨은 이 단계에서 달지 않는다 (docs/annotation-guideline.md 기준 라벨링은 다음 단계).

사용법 (레포 루트에서):
    uv run python datas/yay/generate_reviews.py --count 20
"""

import argparse
import itertools
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from random import Random
from typing import Any

from schema import (
    ASPECT_NAMES_KO,
    ASPECTS,
    ATTRIBUTES,
    TRAVELER_CONTEXTS,
    TRAVELER_NAMES_KO,
    VALUE_NAMES_KO,
)

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "out"
DEFAULT_OUT_PATH = OUT_DIR / "generated" / "reviews.jsonl"

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "gemma4:e4b"
MAX_GENERATION_ATTEMPTS = 3  # 영어 라벨 이름이 섞이면 다시 시도 (첫 시도 + 최대 2회 재시도)

CATEGORY_RATIOS = {"hotel": 0.3, "restaurant": 0.4, "attraction": 0.3}

EXPRESSION_STYLES = {
    "direct": 0.5,
    "concession": 0.15,
    "negation": 0.10,
    "comparison": 0.10,
    "expectation_gap": 0.10,
    "irony": 0.05,
}
EXPRESSION_HINTS_KO = {
    "direct": "생각을 있는 그대로 직접 말합니다.",
    "concession": "\"~긴 한데\"처럼 한 번 인정하고 나서 자기 의견을 말합니다.",
    "negation": "부정문으로 표현합니다. 예: \"별로 안 시끄러웠어요\"",
    "comparison": "다른 곳이나 기대와 비교하는 표현을 씁니다. 예: \"옆집보다 양이 많아요\"",
    "expectation_gap": "기대했던 것과 실제가 어떻게 달랐는지를 표현합니다. 예: \"생각보다 방이 넓었어요\"",
    "irony": "반어법을 씁니다. 실제로는 불만이면서 겉으로는 칭찬하는 척 말합니다. 예: \"참 친절하시더라고요\"",
}

TONE_STYLES = ("polite_yo", "banmal", "short_memo")
TONE_HINTS_KO = {
    "polite_yo": "해요체로 씁니다. 예: \"좋았어요\", \"괜찮더라고요\"",
    "banmal": "반말로 씁니다. 예: \"좋았어\", \"괜찮더라\"",
    "short_memo": "짧은 메모체로 씁니다. 조사나 어미를 생략한 키워드 위주 메모처럼 씁니다. 예: \"청결 굿, 위치 최고\"",
}

TRAVELER_CONTEXT_LIST = sorted(TRAVELER_CONTEXTS)

FORBIDDEN_TOKENS = (
    list(ASPECT_NAMES_KO.keys())
    + [attr for cat_attrs in ATTRIBUTES.values() for attrs in cat_attrs.values() for attr in attrs]
    + list(TRAVELER_CONTEXTS)
    + ["positive", "negative", "neutral"]
)
# 한글 사이에 낀 영어 토큰은 \b 경계가 한글을 \w로 취급해 깨질 수 있어 경계 없이 검사한다.
FORBIDDEN_RE = re.compile("(" + "|".join(re.escape(t) for t in FORBIDDEN_TOKENS) + ")", re.IGNORECASE)

REVIEW_FORMAT_SCHEMA = {
    "type": "object",
    "properties": {"review": {"type": "string"}},
    "required": ["review"],
}


def weighted_choice(rng: Random, weights: dict[str, float]) -> str:
    keys = list(weights.keys())
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def load_places() -> dict[str, list[dict]]:
    places = {}
    for category in CATEGORY_RATIOS:
        path = OUT_DIR / f"places_{category}.json"
        with path.open(encoding="utf-8") as f:
            places[category] = json.load(f)
    return places


def load_split() -> dict[str, str]:
    with (OUT_DIR / "split.json").open(encoding="utf-8") as f:
        return json.load(f)


def load_existing_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    ids = set()
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.add(json.loads(line)["review_id"])
    return ids


def pick_places(rng: Random, places: dict[str, list[dict]], count: int) -> list[dict]:
    """카테고리 비율(hotel 30 / restaurant 40 / attraction 30)을 지키고,
    장소마다 리뷰 수가 고르게 가도록 카테고리별 장소 목록을 라운드로빈으로 순회한다."""
    per_category = largest_remainder_counts(count, CATEGORY_RATIOS)

    chosen: list[dict] = []
    for category, n in per_category.items():
        pool = places[category]
        order = list(range(len(pool)))
        rng.shuffle(order)
        cycle = itertools.cycle(order)
        for _ in range(n):
            chosen.append(pool[next(cycle)])
    rng.shuffle(chosen)
    return chosen


def largest_remainder_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    keys = list(ratios.keys())
    raw = [total * ratios[k] for k in keys]
    floors = [int(x) for x in raw]
    remainder = total - sum(floors)
    order = sorted(range(len(keys)), key=lambda i: (-(raw[i] - floors[i]), i))
    for i in range(remainder):
        floors[order[i]] += 1
    return dict(zip(keys, floors))


def plan_traveler_context(rng: Random) -> list[str]:
    if rng.random() < 0.4:
        return []
    n = rng.randint(1, 2)
    if n == 1:
        return [rng.choice(TRAVELER_CONTEXT_LIST)]
    # solo는 정의상 다른 동행과 함께 나올 수 없다.
    pool = [t for t in TRAVELER_CONTEXT_LIST if t != "solo"]
    return sorted(rng.sample(pool, 2))


def plan_aspects(rng: Random, category: str) -> list[dict]:
    aspects_pool = sorted(ASPECTS[category])
    n = rng.randint(1, min(4, len(aspects_pool)))
    chosen = rng.sample(aspects_pool, n)

    plan = []
    for aspect in chosen:
        values = ATTRIBUTES[category][aspect]
        attribute = rng.choice(values)
        mismatched = rng.random() < 0.10
        plan.append({"aspect": aspect, "attribute": attribute, "mismatched_tone": mismatched})
    return plan


def plan_review(rng: Random, place: dict, split: dict[str, str]) -> dict[str, Any]:
    category = place["category"]
    return {
        "place_id": place["place_id"],
        "category": category,
        "district": place["district"],
        "split": split[place["place_id"]],
        "traveler_context": plan_traveler_context(rng),
        "aspects": plan_aspects(rng, category),
        "expression": weighted_choice(rng, EXPRESSION_STYLES),
        "tone": rng.choice(TONE_STYLES),
        "sentence_count": rng.randint(1, 4),
    }


def aspect_line_ko(category: str, item: dict) -> str:
    aspect_ko = ASPECT_NAMES_KO[item["aspect"]]
    value_ko = VALUE_NAMES_KO[item["attribute"]]
    if item["mismatched_tone"]:
        return (
            f"- {aspect_ko}: 상태는 '{value_ko}'이지만, 그 상태에 대한 화자의 평가는 상태와 어긋나게 표현합니다"
            f"(예: 웨이팅이 길었지만 기다릴 만했다처럼, 상태를 부정적으로 말하면서 평가는 긍정적으로, 또는 그 반대로)."
        )
    return f"- {aspect_ko}: '{value_ko}'"


def build_prompt(place: dict, plan: dict[str, Any]) -> str:
    category_ko = {"hotel": "숙소", "restaurant": "식당", "attraction": "관광지"}[plan["category"]]

    lines = [
        f"당신은 부산 여행 리뷰 사이트에 실제 후기를 남기는 사용자입니다.",
        f"아래 {category_ko}에 대한 한국어 리뷰를 한 편 써 주세요.",
        "",
        f"장소 이름(참고용, 리뷰에 이름을 그대로 쓸 필요는 없습니다): {place['name']}",
        f"참고 정보: {json.dumps(place.get('facts', {}), ensure_ascii=False)}",
        "",
        "리뷰에 반드시 담아야 할 내용:",
    ]

    if plan["traveler_context"]:
        traveler_ko = ", ".join(TRAVELER_NAMES_KO[t] for t in plan["traveler_context"])
        lines.append(f"- 동행: {traveler_ko}와(과) 함께 간 것이 드러나야 합니다.")
    else:
        lines.append("- 동행이 누구인지는 언급하지 않습니다.")

    for item in plan["aspects"]:
        lines.append(aspect_line_ko(plan["category"], item))

    lines += [
        "",
        f"표현 방식: {EXPRESSION_HINTS_KO[plan['expression']]}",
        f"말투: {TONE_HINTS_KO[plan['tone']]}",
        f"길이: {plan['sentence_count']}문장.",
        "",
        "규칙:",
        "- 실제 사람이 쓴 후기처럼 자연스럽게 씁니다. 항목을 나열하듯 쓰지 않습니다.",
        "- 위에 나온 상태나 동행을 나타내는 한국어 표현은 자연스러운 문장 속에 녹여 씁니다.",
        "- 영어 단어나 라벨 이름(예: cleanliness, positive, solo 같은 영어 코드)은 절대 쓰지 않습니다.",
        "- 장소 이름을 리뷰 본문에 그대로 반복하지 않아도 됩니다.",
        "- 다른 설명 없이 리뷰 본문만 씁니다.",
    ]
    return "\n".join(lines)


def call_ollama(model: str, prompt: str, seed: int) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "format": REVIEW_FORMAT_SCHEMA,
        "options": {"temperature": 0.9, "seed": seed},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as res:
            body = json.loads(res.read().decode("utf-8"))
    except urllib.error.URLError as e:
        sys.exit(f"Ollama 서버에 연결할 수 없습니다 ({e}). 'ollama serve'가 실행 중인지 확인하세요.")

    content = body["message"]["content"]
    return json.loads(content)["review"]


def generate_review_text(model: str, place: dict, plan: dict[str, Any], rng: Random) -> str:
    prompt = build_prompt(place, plan)
    last_review = ""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        seed = rng.randrange(2**31)
        last_review = call_ollama(model, prompt, seed)
        if not FORBIDDEN_RE.search(last_review):
            return last_review
    return last_review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100, help="생성할 리뷰 수 (기본 100)")
    parser.add_argument("--seed", type=int, default=42, help="생성 계획 난수 시드 (기본 42)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama 모델 이름 (기본 {DEFAULT_MODEL})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help="출력 JSONL 경로")
    args = parser.parse_args()

    places = load_places()
    split = load_split()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = load_existing_ids(args.out)
    plan_rng = Random(args.seed)
    chosen_places = pick_places(plan_rng, places, args.count)

    made = 0
    skipped = 0
    with args.out.open("a", encoding="utf-8") as f:
        for i, place in enumerate(chosen_places, start=1):
            review_id = f"yay-{i:06d}"
            # 재실행 시에도 이후 review_id의 생성 계획이 그대로 재현되도록,
            # 건너뛰는 경우에도 plan_rng는 반드시 소비한다.
            plan = plan_review(plan_rng, place, split)
            if review_id in existing_ids:
                skipped += 1
                continue

            gen_rng = Random(f"{args.seed}:{review_id}")
            review_text = generate_review_text(args.model, place, plan, gen_rng)

            record = {
                "review_id": review_id,
                "place_id": place["place_id"],
                "category": place["category"],
                "district": place["district"],
                "split": plan["split"],
                "synthetic": True,
                "review": review_text,
                "generation": plan,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            made += 1
            print(f"[{i}/{len(chosen_places)}] {review_id} ({place['category']}) 생성 완료")

    print(f"완료: 새로 생성 {made}건, 이미 있어서 건너뜀 {skipped}건 → {args.out}")


if __name__ == "__main__":
    main()
