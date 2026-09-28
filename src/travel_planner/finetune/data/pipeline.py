"""설정 → Train / Validation / Test 레코드. 학습 · 평가 스크립트가 모두 이 함수로 데이터를 준비한다."""

from ..config import resolve_path
from .dataset import file_sha256, load_records
from .split import apply_split, filter_extra_train, load_manifest


class DataPolicyError(ValueError):
    pass


def load_splits(config: dict) -> tuple[dict[str, list[dict]], dict]:
    """(split별 레코드, metrics.json에 남길 데이터 정보)."""
    data = config["data"]
    dataset_path = resolve_path(data["dataset_path"])
    manifest_path = resolve_path(data["split_manifest"])
    records = load_records(dataset_path)
    manifest = load_manifest(manifest_path)
    splits = apply_split(records, manifest)

    # 프로젝트 규칙: Test는 Gold만. 모든 실험이 같은 Gold Test를 쓴다.
    not_gold = [r["review_id"] for r in splits["test"] if r["tier"] != "gold"]
    if not_gold:
        raise DataPolicyError(f"Test에 Gold가 아닌 레코드 {len(not_gold)}건: {not_gold[:5]}")

    # 나중의 Gold + Silver 실험: 추가 데이터는 Train에만 넣고, Validation/Test 장소의 리뷰는 뺀다.
    extra_info = []
    for extra_path in data.get("extra_train_paths") or []:
        extra, removed = filter_extra_train(load_records(resolve_path(extra_path)), manifest)
        splits["train"] += extra
        extra_info.append({"path": str(extra_path), "sha256": file_sha256(resolve_path(extra_path)), "added": len(extra), "removed_heldout_places": removed})

    sha256 = file_sha256(dataset_path)
    if data.get("max_samples"):  # 스모크 실행용. split마다 앞에서부터 N건
        splits = {name: rows[: data["max_samples"]] for name, rows in splits.items()}

    info = {
        "dataset_path": str(data["dataset_path"]),
        "dataset_version": data.get("dataset_version"),
        "dataset_sha256": sha256,
        "dataset_matches_split_source": sha256 == manifest["source"].get("sha256"),
        "split_manifest": str(data["split_manifest"]),
        "split_version": manifest["version"],
        "counts": {name: len(rows) for name, rows in splits.items()},
        "tiers": {name: sorted({r["tier"] for r in rows}) for name, rows in splits.items()},
        "extra_train": extra_info,
        "max_samples": data.get("max_samples"),
    }  # fmt: skip
    return splits, info
