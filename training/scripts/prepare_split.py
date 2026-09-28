"""place_id 단위 Train / Validation / Test split manifest를 만든다 (한 번만).

사용법:
  uv run python training/scripts/prepare_split.py --config training/configs/common.yaml
  uv run python training/scripts/prepare_split.py --config training/configs/common.yaml --dry-run   # 저장 없이 보기

이미 manifest가 있으면 덮어쓰지 않는다. 새 분할이 필요하면 split.version과 data.split_manifest를 새 이름으로 바꾼다.
"""

import argparse
import sys

from travel_planner.finetune.config import load_config, resolve_path
from travel_planner.finetune.data.dataset import file_sha256, load_records, summarize
from travel_planner.finetune.data.split import SPLITS, SplitError, assign_places, build_manifest, check_no_leakage, save_manifest
from travel_planner.finetune.utils.tracking import git_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="설정 값 바꾸기")
    parser.add_argument("--dry-run", action="store_true", help="분할 결과만 보여주고 저장하지 않는다")
    parser.add_argument("--force", action="store_true", help="기존 manifest를 덮어쓴다 (모든 실험 비교가 깨지니 주의)")
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    data, split = config["data"], config["split"]
    dataset_path = resolve_path(data["dataset_path"])
    records = load_records(dataset_path)
    print(f"데이터 {dataset_path}: {summarize(records)}")

    assignment = assign_places(
        records, seed=config["seed"], ratios=split["ratios"], large_place_min_reviews=split.get("large_place_min_reviews")
    )
    manifest = build_manifest(
        records,
        assignment,
        version=split["version"],
        seed=config["seed"],
        ratios=split["ratios"],
        large_place_min_reviews=split.get("large_place_min_reviews"),
        source={
            "path": str(data["dataset_path"]),
            "dataset_version": data.get("dataset_version"),
            "sha256": file_sha256(dataset_path),
            "records": len(records),
        },
        git=git_info(),
    )
    check_no_leakage({s: [r for r in records if assignment[r["place_id"]] == s] for s in SPLITS})

    print(f"\n{'':12}" + "".join(f"{s:>12}" for s in SPLITS))
    for category in sorted({r["category"] for r in records}) + ["total"]:
        print(f"{category:12}" + "".join(f"{manifest['counts'][s].get(category, 0):>12}" for s in SPLITS))
    print(f"{'places':12}" + "".join(f"{manifest['places'][s]:>12}" for s in SPLITS))
    print(f"Train 고정 대형 장소: {manifest['forced_train_places']}")

    if args.dry_run:
        print("\ndry-run: 저장하지 않았습니다.")
        return
    out = resolve_path(data["split_manifest"])
    try:
        save_manifest(manifest, out, force=args.force)
    except SplitError as e:
        sys.exit(str(e))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
