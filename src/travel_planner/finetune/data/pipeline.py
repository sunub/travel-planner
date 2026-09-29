"""설정 → Train / Validation / Test 레코드. 학습 · 평가 스크립트가 모두 이 함수로 데이터를 준비한다.

데이터를 주는 방법은 두 가지다.
  - data.dataset_path + data.split_manifest: JSONL 하나를 manifest의 review_id 목록대로 나눈다 (gold_split_v1).
  - data.split_files: {train, validation, test} 경로. 이미 나뉜 파일을 그대로 읽는다 (팀 공유 데이터).
두 방법 모두 같은 레코드 검사 · 장소 누수 검사 · Test Gold 검사를 거친다.
"""

from ..config import resolve_path
from .dataset import DatasetError, file_sha256, load_records
from .split import SPLITS, apply_split, check_no_leakage, filter_extra_train, held_out_places, load_manifest


class DataPolicyError(ValueError):
    pass


def _from_manifest(data: dict) -> tuple[dict[str, list[dict]], dict]:
    dataset_path = resolve_path(data["dataset_path"])
    manifest = load_manifest(resolve_path(data["split_manifest"]))
    splits = apply_split(load_records(dataset_path), manifest)
    sha256 = file_sha256(dataset_path)
    return splits, {
        "dataset_path": str(data["dataset_path"]),
        "dataset_sha256": sha256,
        "dataset_matches_split_source": sha256 == manifest["source"].get("sha256"),
        "split_manifest": str(data["split_manifest"]),
        "split_version": manifest["version"],
    }


def _from_split_files(data: dict, split_version: str | None) -> tuple[dict[str, list[dict]], dict]:
    files = data["split_files"]
    missing = [s for s in SPLITS if not files.get(s)]
    if missing:
        raise DatasetError(f"data.split_files에 {missing} 경로가 없습니다")
    splits = {s: load_records(resolve_path(files[s])) for s in SPLITS}
    seen: dict[str, str] = {}
    for split, rows in splits.items():
        for r in rows:
            if r["review_id"] in seen:
                raise DatasetError(f"review_id {r['review_id']}가 {seen[r['review_id']]}와 {split}에 모두 있습니다")
            seen[r["review_id"]] = split
    check_no_leakage(splits)
    return splits, {
        "split_files": {s: {"path": str(files[s]), "sha256": file_sha256(resolve_path(files[s]))} for s in SPLITS},
        "split_version": split_version,
    }


def load_splits(config: dict) -> tuple[dict[str, list[dict]], dict]:
    """(split별 레코드, metrics.json에 남길 데이터 정보)."""
    data = config["data"]
    if data.get("split_files"):
        splits, source = _from_split_files(data, config.get("split", {}).get("version"))
    else:
        splits, source = _from_manifest(data)

    # 프로젝트 규칙: Test는 Gold만. 모든 실험이 같은 Gold Test를 쓴다.
    not_gold = [r["review_id"] for r in splits["test"] if r["tier"] != "gold"]
    if not_gold:
        raise DataPolicyError(f"Test에 Gold가 아닌 레코드 {len(not_gold)}건: {not_gold[:5]}")

    # 나중의 Gold + Silver 실험: 추가 데이터는 Train에만 넣고, Validation/Test 장소의 리뷰는 뺀다.
    extra_info = []
    held_out = held_out_places(splits)
    for extra_path in data.get("extra_train_paths") or []:
        extra, removed = filter_extra_train(load_records(resolve_path(extra_path)), held_out)
        splits["train"] += extra
        extra_info.append({"path": str(extra_path), "sha256": file_sha256(resolve_path(extra_path)), "added": len(extra), "removed_heldout_places": removed})

    if data.get("max_samples"):  # 스모크 실행용. split마다 앞에서부터 N건
        splits = {name: rows[: data["max_samples"]] for name, rows in splits.items()}

    info = {
        **source,
        "dataset_version": data.get("dataset_version"),
        "counts": {name: len(rows) for name, rows in splits.items()},
        "tiers": {name: sorted({r["tier"] for r in rows}) for name, rows in splits.items()},
        "synthetic": {name: sum(1 for r in rows if r["synthetic"]) for name, rows in splits.items()},
        "extra_train": extra_info,
        "max_samples": data.get("max_samples"),
    }  # fmt: skip
    return splits, info
