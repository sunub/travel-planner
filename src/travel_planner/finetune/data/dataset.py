"""라벨 JSONL 읽기와 레코드 검사. 원본 파일은 읽기만 하고 절대 고치지 않는다.

레코드 모양 (datas/lkh/out/gold_review.jsonl에서 확인):
  {"review_id", "place_id", "category", "synthetic", "review",
   "label": {"traveler_context": [...], "aspects": [{"category", "attribute", "sentiment", "evidence"}]},
   "tier": "gold" | "silver"}

review_id · place_id · tier · synthetic은 데이터 관리용 metadata다. 모델이 생성하는 정답은 label뿐이다.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Final

from ..labels import label_issues

REQUIRED_FIELDS: Final = ("review_id", "place_id", "category", "synthetic", "review", "label", "tier")
TIERS: Final = frozenset({"gold", "silver"})


class DatasetError(ValueError):
    pass


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_issues(record: dict) -> list[str]:
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        return [f"필수 필드 누락: {missing}"]
    problems = []
    if record["tier"] not in TIERS:
        problems.append(f"알 수 없는 tier: {record['tier']!r}")
    if not isinstance(record["synthetic"], bool):
        problems.append("synthetic은 true/false여야 함")
    if not isinstance(record["review"], str) or not record["review"].strip():
        return problems + ["review가 비어 있음"]
    return problems + [issue.message for issue in label_issues(record["label"], record["category"], record["review"])]


def load_records(path: Path) -> list[dict]:
    """JSONL을 읽고 모든 레코드를 검사한다. 문제가 하나라도 있으면 목록을 보여주고 멈춘다."""
    if not path.exists():
        raise DatasetError(f"데이터 파일이 없습니다: {path}")
    records = read_jsonl(path)
    problems = [f"{i}행 {r.get('review_id')}: {p}" for i, r in enumerate(records, 1) for p in record_issues(r)]
    duplicated = [rid for rid, n in Counter(r.get("review_id") for r in records).items() if n > 1]
    if duplicated:
        problems.append(f"review_id 중복: {duplicated[:10]}")
    if problems:
        shown = "\n  ".join(problems[:20])
        raise DatasetError(f"{path}: 문제 {len(problems)}건\n  {shown}")
    return records


def summarize(records: list[dict]) -> dict:
    return {
        "records": len(records),
        "places": len({r["place_id"] for r in records}),
        "by_category": dict(sorted(Counter(r["category"] for r in records).items())),
        "by_tier": dict(sorted(Counter(r["tier"] for r in records).items())),
        "synthetic": sum(1 for r in records if r["synthetic"]),
        "aspects": sum(len(r["label"]["aspects"]) for r in records),
        "empty_aspects": sum(1 for r in records if not r["label"]["aspects"]),
    }
