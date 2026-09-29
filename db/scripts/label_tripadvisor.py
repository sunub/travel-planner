"""Tripadvisor 리뷰(db/tripadvisor_reviews.xlsx)를 파인튜닝 모델로 라벨링해 JSONL로 저장한다.

  uv run python db/scripts/label_tripadvisor.py --model hinoonyaso/exaone-tripfit:qlora-v2

모델 출력은 스키마(common/common/schema.py)로 걸러낸다. 카테고리에 없는 aspect·attribute,
허용되지 않은 sentiment, 리뷰 원문에 없는 evidence는 버리고 이유를 dropped에 남긴다.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd

from common.common.schema import ASPECTS, ATTRIBUTES, SENTIMENTS, TRAVELER_CONTEXTS

DB_DIR = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = """당신은 부산 장소 리뷰 정보 추출기다.
주어진 카테고리와 리뷰 원문에서만 정보를 추출하라.
반드시 JSON 객체 하나만 출력하라. 정확한 형식은
{\"traveler_context\": [...], \"aspects\": [{\"category\": \"...\", \"attribute\": \"...\", \"sentiment\": \"...\", \"evidence\": \"원문 그대로의 연속 구절\"}]} 이다.
리뷰에 없는 사실을 만들지 말고, evidence는 반드시 리뷰 원문을 그대로 인용하라."""


def chat(base_url: str, model: str, category: str, review: str) -> str:
    body = {
        "model": model, "stream": False, "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"[카테고리]\n{category}\n\n[리뷰]\n{review}"},
        ],
    }
    request = urllib.request.Request(f"{base_url}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)["message"]["content"]


def parse(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def clean(label: dict, category: str, review: str) -> tuple[dict, list[dict]]:
    """스키마에 맞는 라벨만 남긴다. evidence는 앞뒤 공백만 정리하고 원문 그대로여야 한다."""
    dropped: list[dict] = []
    contexts = [c for c in label.get("traveler_context", []) if c in TRAVELER_CONTEXTS]
    dropped += [{"traveler_context": c, "reason": "unknown_context"}
                for c in label.get("traveler_context", []) if c not in TRAVELER_CONTEXTS]
    aspects, seen = [], set()
    for item in label.get("aspects", []):
        if not isinstance(item, dict):
            continue
        aspect, attribute, sentiment = item.get("category"), item.get("attribute"), item.get("sentiment")
        evidence = item.get("evidence").strip() if isinstance(item.get("evidence"), str) else ""
        if aspect not in ASPECTS[category]:
            reason = "aspect_not_in_category"
        elif attribute not in ATTRIBUTES[category].get(aspect, ()):
            reason = "attribute_not_allowed"
        elif sentiment not in SENTIMENTS:
            reason = "invalid_sentiment"
        elif not evidence or evidence not in review:
            reason = "evidence_not_in_review"
        elif (aspect, attribute, sentiment, evidence) in seen:
            reason = "duplicate"
        else:
            seen.add((aspect, attribute, sentiment, evidence))
            aspects.append({"category": aspect, "attribute": attribute, "sentiment": sentiment, "evidence": evidence})
            continue
        dropped.append({**item, "reason": reason})
    return {"traveler_context": contexts, "aspects": aspects}, dropped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=Path, default=DB_DIR / "tripadvisor_reviews.xlsx")
    parser.add_argument("--out", type=Path, default=DB_DIR / "tripadvisor_labels.jsonl")
    parser.add_argument("--model", default="hinoonyaso/exaone-tripfit:qlora-v2")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()

    reviews = pd.read_excel(args.xlsx, sheet_name="Reviews")
    rows, kept, dropped_total = [], 0, 0
    for index, review in enumerate(reviews.itertuples(index=False), start=1):
        category = str(review.category).lower()
        raw = chat(args.ollama_url, args.model, category, review.text)
        parsed = parse(raw)
        label, dropped = clean(parsed, category, review.text) if parsed else ({"traveler_context": [], "aspects": []}, [])
        kept += len(label["aspects"])
        dropped_total += len(dropped)
        rows.append({
            "review_id": int(review.review_id), "location_id": int(review.location_id), "category": category,
            "model": args.model, "json_valid": parsed is not None, "raw_output": raw,
            "label": label, "dropped": dropped,
        })
        print(f"[{index}/{len(reviews)}] {review.review_id} aspects={len(label['aspects'])} dropped={len(dropped)}", flush=True)

    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"저장: {args.out} | 리뷰 {len(rows)} · JSON 실패 {sum(not r['json_valid'] for r in rows)} · "
          f"aspect 유지 {kept} · 버림 {dropped_total}")


if __name__ == "__main__":
    main()
