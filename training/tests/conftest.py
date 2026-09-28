import json
from pathlib import Path

import pytest

from travel_planner.finetune.config import REPO_ROOT, load_config

GOLD_PATH = REPO_ROOT / "datas" / "lkh" / "out" / "gold_review.jsonl"
MANIFEST_PATH = REPO_ROOT / "training" / "data" / "splits" / "gold_split_v1.json"
CONFIG_DIR = REPO_ROOT / "training" / "configs"

requires_gold = pytest.mark.skipif(not GOLD_PATH.exists(), reason=f"{GOLD_PATH} 없음 (Git 제외 파일)")

# 카테고리마다 허용 목록 안의 aspect 하나씩 (utils/common/schema.py 기준)
ASPECT_BY_CATEGORY = {
    "hotel": ("cleanliness", "clean", "방이 깨끗했어요"),
    "restaurant": ("food_quality", "good", "음식이 맛있어요"),
    "attraction": ("scenery", "sea", "바다가 보여요"),
}


def make_record(index: int, category: str, place: str, *, tier: str = "gold", empty: bool = False) -> dict:
    aspect, attribute, evidence = ASPECT_BY_CATEGORY[category]
    review = f"리뷰 {index}번. {evidence}. 다음에 또 올게요."
    aspects = [] if empty else [{"category": aspect, "attribute": attribute, "sentiment": "positive", "evidence": evidence}]
    return {
        "review_id": f"T-{index:04d}",
        "place_id": place,
        "category": category,
        "synthetic": False,
        "review": review,
        "label": {"traveler_context": [], "aspects": aspects},
        "tier": tier,
    }


@pytest.fixture
def small_records() -> list[dict]:
    """카테고리당 장소 10곳 · 리뷰 1~3건. 한 호텔은 리뷰가 많은 대형 장소."""
    records, index = [], 0
    for category in ("hotel", "restaurant", "attraction"):
        for p in range(10):
            for _ in range(1 + p % 3):
                records.append(make_record(index, category, f"{category}:{p}", empty=index % 7 == 0))
                index += 1
    for _ in range(12):
        records.append(make_record(index, "hotel", "hotel:big"))
        index += 1
    return records


@pytest.fixture
def write_jsonl(tmp_path: Path):
    def _write(rows: list[dict], name: str = "data.jsonl") -> Path:
        path = tmp_path / name
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def tmp_config(tmp_path: Path):
    """실험 설정을 읽고 출력 경로를 tmp_path로 돌린다."""

    def _load(name: str, **data_overrides) -> dict:
        overrides = [f"output.artifacts_root={tmp_path / 'artifacts'}", f"output.experiments_root={tmp_path / 'experiments'}"]
        overrides += [f"data.{key}={value}" for key, value in data_overrides.items()]
        return load_config(CONFIG_DIR / name, overrides)

    return _load
