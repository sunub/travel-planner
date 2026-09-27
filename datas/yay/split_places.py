"""장소를 train/val/test로 나눠 datas/yay/out/split.json에 저장한다.

- 비율: train 80% / val 10% / test 10%
- 카테고리 x district 조합마다 층화 분할
- 이름(공백·괄호 제거 후 비교) 또는 도로명 주소가 같은 장소는 같은 묶음으로 묶어 같은 쪽에 배정
- seed 42로 고정, split.json이 있으면 그대로 사용
"""

import json
import re
from collections import Counter
from pathlib import Path
from random import Random

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "out"
SPLIT_PATH = OUT_DIR / "split.json"
CATEGORIES = ["hotel", "restaurant", "attraction"]
RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42

PAREN_RE = re.compile(r"\([^)]*\)")
WHITESPACE_RE = re.compile(r"\s+")
# 도로명 주소로 인정하는 패턴: "OO로" 또는 "OO길"(번길 포함) 뒤에 공백이 오는 경우.
# "암남동", "동삼동"처럼 도로명 없이 행정동만 적힌 주소는 장소를 특정할 수 없으므로 제외한다.
ROAD_ADDRESS_RE = re.compile(r"(로|길)(\d+번길)?\s")


def normalize_name(name: str) -> str:
    name = PAREN_RE.sub("", name)
    name = WHITESPACE_RE.sub("", name)
    return name


def normalize_address(address: str) -> str:
    return WHITESPACE_RE.sub(" ", address.strip())


def is_road_address(address: str) -> bool:
    return bool(ROAD_ADDRESS_RE.search(address.strip() + " "))


def load_places() -> list[dict]:
    places = []
    for category in CATEGORIES:
        path = OUT_DIR / f"places_{category}.json"
        with path.open(encoding="utf-8") as f:
            places.extend(json.load(f))
    return places


class UnionFind:
    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_clusters(places: list[dict]) -> dict[str, list[str]]:
    uf = UnionFind(p["place_id"] for p in places)

    by_name: dict[str, str] = {}
    by_address: dict[str, str] = {}
    for p in places:
        pid = p["place_id"]

        name_key = normalize_name(p["name"])
        if name_key in by_name:
            uf.union(by_name[name_key], pid)
        else:
            by_name[name_key] = pid

        address = normalize_address(p["address"])
        if is_road_address(address):
            if address in by_address:
                uf.union(by_address[address], pid)
            else:
                by_address[address] = pid

    clusters: dict[str, list[str]] = {}
    for p in places:
        pid = p["place_id"]
        root = uf.find(pid)
        clusters.setdefault(root, []).append(pid)
    return clusters


def stratum_key_for_cluster(member_ids: list[str], places_by_id: dict[str, dict]) -> tuple[str, str]:
    pairs = [(places_by_id[pid]["category"], places_by_id[pid]["district"]) for pid in member_ids]
    counts = Counter(pairs)
    max_count = max(counts.values())
    for pair in pairs:
        if counts[pair] == max_count:
            return pair
    raise AssertionError("unreachable")


def largest_remainder_counts(total: int, ratios: list[float]) -> list[int]:
    raw = [total * r for r in ratios]
    floors = [int(x) for x in raw]
    remainder = total - sum(floors)
    order = sorted(range(len(ratios)), key=lambda i: (-(raw[i] - floors[i]), i))
    for i in range(remainder):
        floors[order[i]] += 1
    return floors


def assign_clusters_to_split(clusters: list[tuple[str, list[str]]], rng: Random) -> dict[str, str]:
    clusters = sorted(clusters, key=lambda c: c[0])  # deterministic order before shuffling
    rng.shuffle(clusters)

    total = sum(len(members) for _, members in clusters)
    buckets = ["train", "val", "test"]
    targets = dict(zip(buckets, largest_remainder_counts(total, [RATIOS[b] for b in buckets])))
    counts = {b: 0 for b in buckets}

    assignment: dict[str, str] = {}
    for _, members in clusters:
        needs = [(targets[b] - counts[b], -buckets.index(b)) for b in buckets]
        bucket = buckets[needs.index(max(needs))]
        counts[bucket] += len(members)
        for pid in members:
            assignment[pid] = bucket
    return assignment


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if SPLIT_PATH.exists():
        print(f"{SPLIT_PATH}가 이미 있어 기존 파일을 그대로 씁니다.")
        with SPLIT_PATH.open(encoding="utf-8") as f:
            split = json.load(f)
        places = load_places()
        print_summary(places, split)
        return

    places = load_places()
    places_by_id = {p["place_id"]: p for p in places}

    clusters = build_clusters(places)

    strata: dict[tuple[str, str], list[tuple[str, list[str]]]] = {}
    for root, member_ids in clusters.items():
        key = stratum_key_for_cluster(member_ids, places_by_id)
        strata.setdefault(key, []).append((root, member_ids))

    rng = Random(SEED)
    split: dict[str, str] = {}
    for key in sorted(strata.keys()):
        split.update(assign_clusters_to_split(strata[key], rng))

    with SPLIT_PATH.open("w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"{SPLIT_PATH}를 새로 만들었습니다.")

    print_summary(places, split)


def print_summary(places: list[dict], split: dict[str, str]):
    counts = {cat: {"train": 0, "val": 0, "test": 0} for cat in CATEGORIES}
    for p in places:
        counts[p["category"]][split[p["place_id"]]] += 1

    header = f"{'category':<12}{'train':>8}{'val':>8}{'test':>8}{'total':>8}"
    print(header)
    print("-" * len(header))
    for cat in CATEGORIES:
        c = counts[cat]
        total = c["train"] + c["val"] + c["test"]
        print(f"{cat:<12}{c['train']:>8}{c['val']:>8}{c['test']:>8}{total:>8}")
    grand_total = {b: sum(counts[cat][b] for cat in CATEGORIES) for b in ("train", "val", "test")}
    total_all = sum(grand_total.values())
    print("-" * len(header))
    print(f"{'total':<12}{grand_total['train']:>8}{grand_total['val']:>8}{grand_total['test']:>8}{total_all:>8}")


if __name__ == "__main__":
    main()
