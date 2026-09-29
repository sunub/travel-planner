"""place_id 단위 Train / Validation / Test 분할과 split manifest.

리뷰가 아니라 장소를 나눈다. 같은 장소의 리뷰가 Train과 Test에 함께 있으면 모델이 추출 능력이 아니라
장소를 외워서 맞힐 수 있다. 목적은 "학습 때 보지 않은 장소의 리뷰"에서의 성능이다.

분할 방법 (카테고리마다 따로, 재현 가능):
  1. 리뷰가 large_place_min_reviews건 이상인 대형 장소는 Train에 고정한다.
     (예: 호텔 380건 중 179건이 한 장소다. 이런 장소가 Test에 가면 Test가 한 장소로 쏠린다.)
  2. 나머지 장소를 f"{seed}:{category}"로 만든 난수로 섞는다.
  3. 섞인 순서대로, 넣어도 목표 리뷰 수(카테고리 리뷰 수 × 비율)를 넘지 않는 장소를 Test, 다음으로 Validation에
     채운다. 어느 쪽에도 들어가지 않은 장소는 Train이다.

한 번 만든 manifest는 모든 실험이 그대로 쓴다. 이미 있으면 --force 없이는 덮어쓰지 않는다.
"""

import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

SPLITS: Final = ("train", "validation", "test")
HOLDOUT_SPLITS: Final = ("test", "validation")  # 이 순서로 채운다


class SplitError(ValueError):
    pass


def assign_places(
    records: list[dict], *, seed: int, ratios: dict[str, float], large_place_min_reviews: int | None
) -> dict[str, str]:
    """place_id → split."""
    if abs(sum(ratios[s] for s in SPLITS) - 1.0) > 1e-6:
        raise SplitError(f"ratios의 합이 1이어야 합니다: {ratios}")
    categories_of: dict[str, set[str]] = defaultdict(set)
    for record in records:
        categories_of[record["place_id"]].add(record["category"])
    mixed = sorted(pid for pid, cats in categories_of.items() if len(cats) > 1)
    if mixed:
        raise SplitError(f"카테고리가 여러 개인 place_id가 있습니다: {mixed[:5]}")

    assignment: dict[str, str] = {}
    for category in sorted({r["category"] for r in records}):
        reviews_per_place = Counter(r["place_id"] for r in records if r["category"] == category)
        total = sum(reviews_per_place.values())
        targets = {split: round(total * ratios[split]) for split in HOLDOUT_SPLITS}
        filled = dict.fromkeys(HOLDOUT_SPLITS, 0)

        large = {p for p, n in reviews_per_place.items() if large_place_min_reviews and n >= large_place_min_reviews}
        assignment |= dict.fromkeys(large, "train")

        candidates = sorted(p for p in reviews_per_place if p not in large)
        random.Random(f"{seed}:{category}").shuffle(candidates)
        for place in candidates:
            n = reviews_per_place[place]
            split = next((s for s in HOLDOUT_SPLITS if filled[s] + n <= targets[s]), "train")
            if split != "train":
                filled[split] += n
            assignment[place] = split
    return assignment


def build_manifest(
    records: list[dict],
    assignment: dict[str, str],
    *,
    version: str,
    seed: int,
    ratios: dict[str, float],
    large_place_min_reviews: int | None,
    source: dict,
    git: dict,
) -> dict:
    review_ids = {split: sorted(r["review_id"] for r in records if assignment[r["place_id"]] == split) for split in SPLITS}
    counts = {split: Counter() for split in SPLITS}
    for record in records:
        counts[assignment[record["place_id"]]][record["category"]] += 1
    reviews_per_place = Counter(r["place_id"] for r in records)
    return {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git,
        "group_key": "place_id",
        "stratify_by": "category",
        "seed": seed,
        "ratios": ratios,
        "large_place_min_reviews": large_place_min_reviews,
        "source": source,
        "counts": {
            split: {**dict(sorted(counts[split].items())), "total": sum(counts[split].values())} for split in SPLITS
        },
        "places": {split: sum(1 for s in assignment.values() if s == split) for split in SPLITS},
        "forced_train_places": sorted(
            p for p, n in reviews_per_place.items() if large_place_min_reviews and n >= large_place_min_reviews
        ),
        "assignment": dict(sorted(assignment.items())),
        "review_ids": review_ids,
    }


def save_manifest(manifest: dict, path: Path, *, force: bool = False) -> None:
    if path.exists() and not force:
        raise SplitError(f"split manifest가 이미 있습니다: {path} (모든 실험이 같은 분할을 쓰도록 덮어쓰지 않습니다)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise SplitError(f"split manifest가 없습니다: {path}\n먼저 training/scripts/prepare_split.py를 실행하세요.")
    return json.loads(path.read_text(encoding="utf-8"))


def apply_split(records: list[dict], manifest: dict) -> dict[str, list[dict]]:
    """manifest의 review_id 목록대로 레코드를 나눈다. manifest에 없는 리뷰가 있으면 멈춘다."""
    split_of = {rid: split for split, ids in manifest["review_ids"].items() for rid in ids}
    unknown = [r["review_id"] for r in records if r["review_id"] not in split_of]
    if unknown:
        raise SplitError(
            f"manifest {manifest['version']}에 없는 리뷰 {len(unknown)}건: {unknown[:5]}. "
            "데이터가 바뀌었다면 새 버전의 split을 만드세요."
        )
    splits: dict[str, list[dict]] = {split: [] for split in SPLITS}
    for record in records:
        splits[split_of[record["review_id"]]].append(record)
    check_no_leakage(splits)
    return splits


def check_no_leakage(splits: dict[str, list[dict]]) -> None:
    places = {split: {r["place_id"] for r in rows} for split, rows in splits.items()}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            shared = places.get(a, set()) & places.get(b, set())
            if shared:
                raise SplitError(f"{a}와 {b}에 같은 place_id가 있습니다: {sorted(shared)[:5]}")


def held_out_places(splits: dict[str, list[dict]]) -> set[str]:
    return {r["place_id"] for split in HOLDOUT_SPLITS for r in splits[split]}


def filter_extra_train(records: list[dict], held_out: set[str]) -> tuple[list[dict], int]:
    """추가 학습 데이터(예: Silver)에서 Validation/Test 장소(held_out)의 리뷰를 뺀다. (남은 레코드, 뺀 수)."""
    kept = [r for r in records if r["place_id"] not in held_out]
    return kept, len(records) - len(kept)
