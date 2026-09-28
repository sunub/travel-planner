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
import time
import urllib.error
import urllib.request
from collections import Counter
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
# 영어 라벨 이름이 섞이거나 추측 표현이 나오면 다시 시도한다 (첫 시도 + 최대 2회 재시도, 두 검사 합산).
MAX_GENERATION_ATTEMPTS = 3

CATEGORY_RATIOS = {"hotel": 0.3, "restaurant": 0.4, "attraction": 0.3}

EXPRESSION_STYLES = {
    "direct": 0.52,
    "concession": 0.15,
    "negation": 0.10,
    "comparison": 0.10,
    "expectation_gap": 0.10,
    "irony": 0.03,
}

TONE_STYLES = ("polite_yo", "banmal", "short_memo")
TONE_HINTS_KO = {
    "polite_yo": "해요체로 씁니다. 예: \"좋았어요\", \"괜찮더라고요\"",
    "banmal": "반말로 씁니다. 예: \"좋았어\", \"괜찮더라\"",
    "short_memo": "짧은 메모체로 씁니다. 조사나 어미를 생략한 키워드 위주 메모처럼 씁니다. 예: \"청결 굿, 위치 최고\"",
}

TRAVELER_CONTEXT_LIST = sorted(TRAVELER_CONTEXTS)

# aspect별 attribute 값이 실제로 어떤 한국어 표현으로 나타나는지 예시를 준다.
# "보통"을 "느림"으로 쓰는 식의 혼동(2번 요구사항)을 막기 위해 aspect마다 따로 둔다
# (예: parking_availability의 unavailable과 amenities의 unavailable은 뜻이 다르다).
ASPECT_VALUE_EXAMPLES_KO: dict[str, dict[str, str]] = {
    "cleanliness": {
        "clean": "깨끗했어요, 청소가 잘 되어 있었어요",
        "average": "청결은 보통이었어요, 그럭저럭 깨끗했어요",
        "dirty": "지저분했어요, 청소 상태가 안 좋았어요",
    },
    "noise_level": {
        "quiet": "조용했어요, 소음이 없었어요",
        "moderate": "소음은 보통이었어요, 크게 거슬리지 않았어요",
        "noisy": "시끄러웠어요, 소음이 심했어요",
    },
    "bed_comfort": {
        "comfortable": "침대가 편안했어요, 푹 잘 잤어요",
        "average": "침대는 무난했어요, 그냥저냥 잘 만했어요",
        "uncomfortable": "침대가 불편했어요, 잠자리가 편하지 않았어요",
    },
    "room_size": {
        "spacious": "방이 넓었어요, 공간이 여유로웠어요",
        "average": "방 크기는 보통이었어요, 넓지도 좁지도 않았어요",
        "cramped": "방이 좁았어요, 답답했어요",
    },
    "bathroom_quality": {
        "good": "수압이 좋았어요, 욕실 상태가 좋았어요",
        "average": "욕실은 무난했어요, 수압이 그냥 보통이었어요",
        "poor": "수압이 약했어요, 욕실 상태가 안 좋았어요",
    },
    "room_condition": {
        "well_kept": "시설 관리가 잘 되어 있었어요, 방이 깔끔하게 관리되고 있었어요",
        "average": "시설 상태는 보통이었어요, 낡지도 새것 같지도 않았어요",
        "worn": "시설이 낡았어요, 오래된 티가 났어요",
    },
    "view_quality": {
        "good": "전망이 좋았어요, 창밖 풍경이 멋있었어요",
        "average": "전망은 그냥 그랬어요, 특별한 건 없었어요",
        "poor": "전망이 안 좋았어요, 창밖에 볼 게 없었어요",
    },
    "staff_service": {
        "friendly": "직원분들이 친절했어요, 응대가 좋았어요",
        "average": "응대는 무난했어요, 특별히 친절하지도 불친절하지도 않았어요",
        "unfriendly": "직원이 불친절했어요, 응대가 무뚝뚝했어요",
    },
    "breakfast_quality": {
        "good": "조식이 맛있었어요, 조식 구성이 좋았어요",
        "average": "조식은 그냥 무난했어요, 평범했어요",
        "poor": "조식이 별로였어요, 조식 맛이 아쉬웠어요",
    },
    "amenities": {
        "available": "부대시설이 잘 갖춰져 있었어요, 수영장이나 헬스장이 있었어요",
        "unavailable": "부대시설이 따로 없었어요, 편의시설이 부족했어요",
    },
    "parking_availability": {
        "available": "주차장이 있었어요, 주차가 가능했어요",
        "limited": "주차 자리가 몇 개 없었어요, 주차 공간이 제한적이었어요",
        "unavailable": "주차장이 없었어요, 주차를 못 했어요",
    },
    "parking_experience": {
        "easy": "주차가 쉬웠어요, 자리 찾기가 편했어요",
        "average": "주차는 그냥 무난했어요, 특별히 어렵지 않았어요",
        "difficult": "주차가 어려웠어요, 자리 찾기 힘들었어요",
    },
    "food_quality": {
        "good": "맛있었어요, 음식 맛이 좋았어요",
        "average": "맛은 무난했어요, 그냥 먹을 만했어요",
        "poor": "맛이 별로였어요, 입맛에 안 맞았어요",
    },
    "freshness": {
        "fresh": "재료가 신선했어요, 신선도가 좋았어요",
        "average": "신선도는 보통이었어요, 특별히 신선하지도 않았어요",
        "not_fresh": "재료가 신선하지 않았어요, 신선도가 떨어졌어요",
    },
    "portion": {
        "large": "양이 많았어요, 푸짐했어요",
        "normal": "양은 적당했어요, 보통이었어요",
        "small": "양이 적었어요, 부족했어요",
    },
    "waiting_time": {
        "none": "웨이팅 없이 바로 들어갔어요, 기다리지 않았어요",
        "short": "웨이팅이 짧았어요, 잠깐 기다렸어요",
        "long": "웨이팅이 길었어요, 한참을 기다렸어요",
    },
    "serving_speed": {
        "fast": "음식이 빨리 나왔어요, 서빙이 빨랐어요",
        "average": "음식 나오는 속도는 보통이었어요, 그냥 무난했어요",
        "slow": "음식이 늦게 나왔어요, 서빙이 느렸어요, 오래 걸렸어요",
    },
    "atmosphere": {
        "good": "분위기가 좋았어요, 인테리어가 예뻤어요",
        "average": "분위기는 무난했어요, 평범했어요",
        "poor": "분위기가 별로였어요, 아쉬웠어요",
    },
    "seating_comfort": {
        "comfortable": "좌석이 편안했어요, 앉아 있기 편했어요",
        "average": "좌석은 보통이었어요, 그냥저냥 앉을 만했어요",
        "uncomfortable": "좌석이 불편했어요, 앉아 있기 힘들었어요",
    },
    "family_friendly": {
        "suitable": "아이 의자가 있었어요, 가족 손님이 많았어요, 아이 데리고 가기 좋아 보였어요",
        "unsuitable": "아이와 함께 가기엔 불편해 보였어요, 아이 의자가 없었어요",
    },
    "scenery": {
        "sea": "바다가 보였어요, 바다 전망이었어요",
        "mountain": "산이 보였어요, 산 경치였어요",
        "city": "도시 풍경이 보였어요, 시내가 내려다보였어요",
        "river": "강이 보였어요, 강변 풍경이었어요",
        "night_view": "야경이 예뻤어요, 밤에 보는 경치가 좋았어요",
    },
    "photo_spots": {
        "good": "사진 찍기 좋았어요, 포토스팟이 많았어요",
        "average": "사진은 그냥 무난하게 나왔어요, 특별한 포토스팟은 없었어요",
        "poor": "사진 찍을 만한 곳이 마땅치 않았어요",
    },
    "walking_burden": {
        "low": "많이 걷지 않아도 됐어요, 걷기 부담이 적었어요",
        "medium": "적당히 걸었어요, 걷기 부담은 보통이었어요",
        "high": "많이 걸어야 했어요, 걷기 부담이 컸어요",
    },
    "slope_stairs": {
        "low": "오르막이나 계단이 거의 없었어요, 평지라 편했어요",
        "medium": "계단이 좀 있었어요, 오르내림이 적당히 있었어요",
        "high": "계단이 많았어요, 오르막이 심했어요",
    },
    "activity_variety": {
        "many": "볼거리가 많았어요, 즐길 거리가 다양했어요",
        "average": "볼거리는 보통이었어요, 딱히 많지도 적지도 않았어요",
        "few": "볼거리가 별로 없었어요, 즐길 거리가 적었어요",
    },
    "stay_duration": {
        "short": "잠깐 들르기 좋았어요, 금방 다 봤어요",
        "medium": "한두 시간 정도 걸렸어요, 둘러보는 데 적당한 시간이 걸렸어요",
        "long": "반나절은 걸렸어요, 다 보려니 시간이 오래 걸렸어요",
    },
    "rest_facilities": {
        "sufficient": "쉴 곳이 충분했어요, 벤치나 정자가 있었어요",
        "lacking": "쉴 곳이 마땅치 않았어요, 휴식 공간이 부족했어요",
    },
    "toilet_facilities": {
        "sufficient": "화장실이 잘 갖춰져 있었어요, 화장실 찾기 편했어요",
        "lacking": "화장실이 부족했어요, 화장실 찾기 힘들었어요",
    },
    "weather_sensitivity": {
        "high": "야외라 날씨 영향을 많이 받았어요, 비 오면 가기 힘들 것 같았어요",
        "low": "실내라 날씨 상관없이 다닐 수 있었어요, 비가 와도 문제없었어요",
    },
}
# ATTRIBUTES에 있는 모든 aspect/attribute 조합에 예시가 있는지 import 시점에 확인한다.
for _category, _aspect_map in ATTRIBUTES.items():
    for _aspect, _values in _aspect_map.items():
        for _value in _values:
            if _value not in ASPECT_VALUE_EXAMPLES_KO.get(_aspect, {}):
                raise ValueError(f"예시 문구가 없는 aspect/attribute: {_aspect}/{_value}")

# 어떤 aspect로도 나타낼 수 없어 항상 금지해야 하는 평가 주제.
ALWAYS_FORBIDDEN_TOPICS_KO = ["가격", "재방문 의사"]
# 계획에 해당 aspect가 없을 때만 금지해야 하는 평가 주제.
CONDITIONAL_TOPIC_TO_ASPECT_KO = {"맛": "food_quality", "분위기": "atmosphere"}

# 리뷰에 남으면 안 되는 영어 라벨 이름 (aspect 코드, attribute 값, traveler_context 코드, sentiment).
FORBIDDEN_TOKENS = (
    list(ASPECT_NAMES_KO.keys())
    + [attr for cat_attrs in ATTRIBUTES.values() for attrs in cat_attrs.values() for attr in attrs]
    + list(TRAVELER_CONTEXTS)
    + ["positive", "negative", "neutral"]
)
# 한글 사이에 낀 영어 토큰은 \b 경계가 한글을 \w로 취급해 깨질 수 있어 경계 없이 검사한다.
FORBIDDEN_RE = re.compile("(" + "|".join(re.escape(t) for t in FORBIDDEN_TOKENS) + ")", re.IGNORECASE)

# 추측·전언 표현 (직접 겪은 사실이 아님을 암시) — 5번 요구사항.
SPECULATIVE_TOKENS = ["것 같", "걱정", "듯"]
SPECULATIVE_RE = re.compile("(" + "|".join(re.escape(t) for t in SPECULATIVE_TOKENS) + ")")

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


def load_existing_records(out_path: Path) -> list[dict]:
    if not out_path.exists():
        return []
    records = []
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}시간 {minutes}분"
    if minutes:
        return f"{minutes}분 {secs}초"
    return f"{secs}초"


def print_summary(out_path: Path) -> None:
    records = load_existing_records(out_path)
    total = len(records)
    if total == 0:
        print("요약: 저장된 리뷰가 없습니다.")
        return

    category_counts = Counter(r["category"] for r in records)
    split_counts = Counter(r["split"] for r in records)
    expression_counts = Counter(r["generation"]["expression"] for r in records)
    place_counts = Counter(r["place_id"] for r in records)
    per_place = list(place_counts.values())

    print("--- 요약 ---")
    print(f"전체 리뷰 수: {total}")
    print(
        "카테고리별: "
        + ", ".join(
            f"{c}={category_counts.get(c, 0)}건 ({category_counts.get(c, 0) / total:.1%})"
            for c in CATEGORY_RATIOS
        )
    )
    print("split별: " + ", ".join(f"{s}={n}건" for s, n in sorted(split_counts.items())))
    print(
        f"장소당 리뷰 수 (장소 {len(per_place)}곳): "
        f"최소 {min(per_place)}, 평균 {sum(per_place) / len(per_place):.2f}, 최대 {max(per_place)}"
    )
    print("표현 방식별: " + ", ".join(f"{e}={n}건" for e, n in sorted(expression_counts.items())))


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


def aspect_line_ko(item: dict, plan: dict[str, Any]) -> str:
    aspect_ko = ASPECT_NAMES_KO[item["aspect"]]
    value_ko = VALUE_NAMES_KO[item["attribute"]]
    example = ASPECT_VALUE_EXAMPLES_KO[item["aspect"]][item["attribute"]]

    if item["mismatched_tone"]:
        line = (
            f"- {aspect_ko}: 상태는 '{value_ko}'({example})이지만, 그 상태에 대한 화자의 평가는 상태와 어긋나게 표현합니다"
            f"(예: 웨이팅이 길었지만 기다릴 만했다처럼, 상태를 부정적으로 말하면서 평가는 긍정적으로, 또는 그 반대로)."
        )
    else:
        line = f"- {aspect_ko}: '{value_ko}' (표현 예: {example})"

    if item["aspect"] == "family_friendly" and "family_with_kids" not in plan["traveler_context"]:
        line += " 화자가 직접 아이를 데려간 것처럼 쓰지 말고, 아이 의자나 다른 가족 손님을 본 것 같은 관찰한 사실로 씁니다."
    return line


def expression_instruction_ko(plan: dict[str, Any]) -> str:
    style = plan["expression"]
    aspect_names = [ASPECT_NAMES_KO[a["aspect"]] for a in plan["aspects"]]

    if style == "direct":
        return "생각을 있는 그대로 직접 말합니다. 예: \"방이 넓고 좋았어요.\""

    if style == "negation":
        target = f" ({aspect_names[0]} 등)" if aspect_names else ""
        return (
            f"계획에 있는 항목 중 최소 하나{target}를 부정문으로 표현합니다. "
            "예: \"별로 안 시끄러웠어요\", \"주차가 어렵지 않았어요\""
        )

    if style == "concession":
        if len(aspect_names) >= 2:
            a, b = aspect_names[0], aspect_names[1]
            return (
                f"\"~긴 한데 ~\" 형태로 두 항목을 대조합니다. {a}은(는) 아쉬운 점으로 인정하고, "
                f"{b}은(는) 그래도 좋았던 점으로 말합니다. 예: \"방은 좀 좁긴 한데 전망은 정말 좋았어요.\""
            )
        return (
            "\"~긴 한데 ~\" 형태로, 그 항목의 상태와 그에 대한 화자의 느낌을 대조해서 씁니다. "
            "예: \"웨이팅이 길긴 한데 기다릴 만했어요.\""
        )

    if style == "comparison":
        return "다른 곳과 비교하는 표현을 씁니다. 예: \"다른 호텔보다 조용했어요.\""

    if style == "expectation_gap":
        return "기대했던 것과 실제 경험이 어떻게 달랐는지 대조해서 씁니다. 예: \"생각보다 방이 넓었어요.\""

    return (  # irony
        "반어법을 씁니다. 실제로는 불만이지만 겉으로는 칭찬하듯 말합니다. "
        "예: \"웨이팅 1시간이라니 정말 최고네요^^\", "
        "\"주차장 찾느라 30분 돌았네요, 덕분에 동네 구경 잘했습니다^^\""
    )


def forbidden_topics_ko(plan: dict[str, Any]) -> list[str]:
    planned_aspects = {a["aspect"] for a in plan["aspects"]}
    topics = list(ALWAYS_FORBIDDEN_TOPICS_KO)
    for topic, aspect in CONDITIONAL_TOPIC_TO_ASPECT_KO.items():
        if aspect not in planned_aspects:
            topics.append(topic)
    return topics


def scenery_rule_ko(plan: dict[str, Any]) -> str | None:
    """attraction 리뷰에서 경관 감상이 계획 없이 새는 문제(파일럿2에서 3/20 발견)를 막는다."""
    if plan["category"] != "attraction":
        return None
    if any(a["aspect"] == "scenery" for a in plan["aspects"]):
        return "경관은 계획에 있는 종류(바다/산/도시 등)만 드러내고, 그 외에 경관이 좋았다는 감탄은 덧붙이지 않습니다."
    return "경관(scenery)은 계획에 없으니, \"바다 뷰가 최고였다\"처럼 경관 자체에 대한 감상은 쓰지 않습니다."


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
        lines.append(aspect_line_ko(item, plan))

    lines += [
        "",
        f"표현 방식: {expression_instruction_ko(plan)}",
        f"말투: {TONE_HINTS_KO[plan['tone']]}",
        f"길이: {plan['sentence_count']}문장.",
        "",
        "규칙:",
        "- 계획의 모든 항목을 하나도 빠짐없이 리뷰에 넣습니다.",
        "- 실제 사람이 쓴 후기처럼 자연스럽게 씁니다. 항목을 나열하듯 쓰지 않습니다.",
        "- 위에 나온 상태나 동행을 나타내는 한국어 표현은 자연스러운 문장 속에 녹여 씁니다.",
        "- 상태값은 위에 적힌 뜻 그대로 씁니다. 예를 들어 '보통'을 '느렸어요'라고 쓰거나, '느림'을 '보통이었어요'라고 쓰면 안 됩니다.",
        f"- 위 목록에 있는 항목만 평가합니다. {', '.join(forbidden_topics_ko(plan))}처럼 계획에 없는 항목의 좋고 나쁨은 쓰지 않습니다.",
        "- 장소 이름이나 메뉴 이름은 언급해도 되지만, 그것 자체에 대한 좋고 나쁨 평가는 붙이지 않습니다.",
        "- 추측이나 걱정하는 투로 쓰지 말고, 직접 겪은 사실처럼 씁니다 (\"~것 같아요\", \"~할까 걱정돼요\" 같은 표현은 쓰지 않습니다).",
        "- 영어 단어나 라벨 이름(예: cleanliness, positive, solo 같은 영어 코드)은 절대 쓰지 않습니다.",
        "- 장소 이름을 리뷰 본문에 그대로 반복하지 않아도 됩니다.",
        "- 다른 설명 없이 리뷰 본문만 씁니다.",
    ]

    scenery_rule = scenery_rule_ko(plan)
    if scenery_rule:
        lines.append(f"- {scenery_rule}")

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


def generate_review_text(model: str, place: dict, plan: dict[str, Any], rng: Random) -> tuple[str, int]:
    """리뷰 본문과, 그걸 만드는 데 걸린 시도 횟수(1이면 재시도 없음)를 함께 돌려준다."""
    prompt = build_prompt(place, plan)
    last_review = ""
    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        seed = rng.randrange(2**31)
        last_review = call_ollama(model, prompt, seed)
        if not FORBIDDEN_RE.search(last_review) and not SPECULATIVE_RE.search(last_review):
            return last_review, attempt
    return last_review, MAX_GENERATION_ATTEMPTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help=(
            "최종 목표 리뷰 수 (기본 100). 파일에 이미 있는 리뷰까지 합친 개수로, "
            "이미 있는 만큼은 건너뛰고 모자란 만큼만 새로 생성한다."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="생성 계획 난수 시드 (기본 42)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama 모델 이름 (기본 {DEFAULT_MODEL})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help="출력 JSONL 경로")
    args = parser.parse_args()

    places = load_places()
    split = load_split()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    existing_records = load_existing_records(args.out)
    existing_ids = {r["review_id"] for r in existing_records}
    category_counts = Counter(r["category"] for r in existing_records)
    initial_existing = len(existing_ids)

    if initial_existing >= args.count:
        print(f"이미 {initial_existing}건이 있어 목표 {args.count}건을 채웠습니다. 새로 생성할 것이 없습니다.")
        print_summary(args.out)
        return

    plan_rng = Random(args.seed)
    chosen_places = pick_places(plan_rng, places, args.count)

    made = 0
    skipped = 0
    retries_total = 0
    start_time = time.monotonic()
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
            review_text, attempts = generate_review_text(args.model, place, plan, gen_rng)
            retries_total += attempts - 1

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
            category_counts[place["category"]] += 1

            if made % 10 == 0:
                completed = initial_existing + made
                elapsed = time.monotonic() - start_time
                avg = elapsed / made
                remaining = max(args.count - completed, 0)
                eta = format_duration(avg * remaining)
                cat_str = ", ".join(f"{c}={category_counts[c]}" for c in CATEGORY_RATIOS)
                print(
                    f"[진행] {completed}/{args.count} | 카테고리 누적 {cat_str} | "
                    f"평균 {avg:.2f}초/건 | 예상 남은 시간 {eta} | 누적 재시도 {retries_total}회"
                )

    print(f"완료: 새로 생성 {made}건, 이미 있어서 건너뜀 {skipped}건 → {args.out}")
    print_summary(args.out)


if __name__ == "__main__":
    main()
