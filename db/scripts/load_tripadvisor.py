"""Tripadvisor 실제 리뷰와 모델 라벨로 DB의 장소·리뷰 데이터를 교체한다.

  uv run python db/scripts/load_tripadvisor.py --database-url postgresql://...        # 교체
  uv run python db/scripts/load_tripadvisor.py --database-url postgresql://... --dry-run

기존 장소·리뷰·라벨(과 이를 참조하는 스크랩)은 모두 지우고, 지역·카테고리·동행 유형 시드와 사용자는 남긴다.
실제 리뷰에는 학습 분할·라벨 등급이 없으므로 reviews.dataset_split·label_tier를 NULL 허용으로 바꾼다.
전체를 한 트랜잭션으로 실행해 중간에 실패하면 아무것도 바뀌지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import psycopg

DB_DIR = Path(__file__).resolve().parents[1]
CATEGORY_NAMES = {"hotel": "호텔", "restaurant": "식당", "attraction": "관광지"}
COMPANION_NAMES = {"solo": "혼자", "couple": "연인", "friends": "친구", "parents": "부모님", "family_with_kids": "아이 동반 가족"}


def _clean(value):
    """NaN을 None으로 바꿔 JSON·SQL에 넣을 수 있게 한다."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value.item() if hasattr(value, "item") else value


def records(xlsx: Path, labels_path: Path) -> tuple[list[dict], list[dict], dict[int, dict]]:
    places = [{k: _clean(v) for k, v in row.items()} for row in pd.read_excel(xlsx, sheet_name="Locations").to_dict("records")]
    reviews = [{k: _clean(v) for k, v in row.items()} for row in pd.read_excel(xlsx, sheet_name="Reviews").to_dict("records")]
    labels = {row["review_id"]: row for row in map(json.loads, labels_path.read_text(encoding="utf-8").splitlines())}
    missing = [r["review_id"] for r in reviews if r["review_id"] not in labels]
    if missing:
        raise SystemExit(f"라벨 파일에 없는 리뷰가 있습니다: {missing[:5]}")
    for review in reviews:
        for item in labels[review["review_id"]]["label"]["aspects"]:
            if item["evidence"] not in review["text"]:
                raise SystemExit(f"{review['review_id']}: evidence가 리뷰 원문에 없습니다: {item['evidence']}")
    return places, reviews, labels


def one(cursor, sql: str, params: tuple):
    cursor.execute(sql, params)
    return cursor.fetchone()[0]


def load(cursor, places: list[dict], reviews: list[dict], labels: dict[int, dict], manifest: dict) -> int:
    cursor.execute("ALTER TABLE reviews ALTER COLUMN dataset_split DROP NOT NULL, ALTER COLUMN label_tier DROP NOT NULL")
    cursor.execute("""TRUNCATE places, reviews, review_annotations, review_companions, aspects, aspect_values,
                      place_images, place_tags, tags, districts, dataset_imports RESTART IDENTITY CASCADE""")

    category_ids = {
        code: one(cursor, """INSERT INTO place_categories (category_code, category_name) VALUES (%s, %s)
            ON CONFLICT (category_code) DO UPDATE SET category_name = EXCLUDED.category_name RETURNING category_id""",
            (code, name))
        for code, name in CATEGORY_NAMES.items()
    }
    place_ids = {}
    for place in places:
        category = str(place["category"]).lower()
        place_ids[place["location_id"]] = one(cursor, """INSERT INTO places
            (category_id, source, source_place_id, place_name, address, latitude, longitude)
            VALUES (%s, 'tripadvisor', %s, %s, %s, %s, %s) RETURNING place_id""",
            (category_ids[category], str(place["location_id"]), place["name"], place["address"], place["latitude"], place["longitude"]))

    annotations = 0
    aspect_ids: dict[tuple[int, str], int] = {}
    for review in reviews:
        category = str(review["category"]).lower()
        label = labels[review["review_id"]]
        review_id = one(cursor, """INSERT INTO reviews
            (place_id, source, external_review_id, source_member, dataset_split, label_tier, is_synthetic, review_text, raw_record)
            VALUES (%s, 'tripadvisor', %s, NULL, NULL, NULL, FALSE, %s, %s::jsonb) RETURNING review_id""",
            (place_ids[review["location_id"]], str(review["review_id"]), review["text"], json.dumps(review, ensure_ascii=False)))
        for code in label["label"]["traveler_context"]:
            companion_id = one(cursor, """INSERT INTO companion_types (companion_code, companion_name) VALUES (%s, %s)
                ON CONFLICT (companion_code) DO UPDATE SET companion_name = EXCLUDED.companion_name RETURNING companion_type_id""",
                (code, COMPANION_NAMES.get(code, code)))
            cursor.execute("INSERT INTO review_companions (review_id, companion_type_id) VALUES (%s, %s)", (review_id, companion_id))
        for item in label["label"]["aspects"]:
            key = (category_ids[category], item["category"])
            if key not in aspect_ids:
                aspect_ids[key] = one(cursor, """INSERT INTO aspects (category_id, aspect_name) VALUES (%s, %s)
                    ON CONFLICT (category_id, aspect_name) DO UPDATE SET is_active = TRUE RETURNING aspect_id""", key)
            cursor.execute("""INSERT INTO aspect_values (aspect_id, value_name) VALUES (%s, %s)
                ON CONFLICT (aspect_id, value_name) DO NOTHING""", (aspect_ids[key], item["attribute"]))
            start = review["text"].index(item["evidence"])
            cursor.execute("""INSERT INTO review_annotations
                (review_id, aspect_id, attribute_value, sentiment, evidence_text, evidence_start, evidence_end, annotation_tier, created_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'model', %s)""",
                (review_id, aspect_ids[key], item["attribute"], item["sentiment"], item["evidence"],
                 start, start + len(item["evidence"]), f"model:{label['model']}"))
            annotations += 1

    cursor.execute("""INSERT INTO dataset_imports (dataset_name, manifest, review_count, annotation_count)
        VALUES (%s, %s::jsonb, %s, %s)""", (manifest["dataset_name"], json.dumps(manifest, ensure_ascii=False), len(reviews), annotations))
    return annotations


def local_url() -> str:
    from dotenv import dotenv_values

    env = dotenv_values(DB_DIR / ".env")
    return "postgresql://{}:{}@{}:{}/{}".format(env["POSTGRES_USER"], env["POSTGRES_PASSWORD"],
                                                env.get("POSTGRES_HOST") or "localhost", env.get("POSTGRES_PORT") or "5432", env["POSTGRES_DB"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=Path, default=DB_DIR / "tripadvisor_reviews.xlsx")
    parser.add_argument("--labels", type=Path, default=DB_DIR / "tripadvisor_labels.jsonl")
    parser.add_argument("--database-url", help="생략하면 db/.env의 POSTGRES_* 로컬 DB")
    parser.add_argument("--dry-run", action="store_true", help="검증만 하고 DB는 바꾸지 않는다")
    args = parser.parse_args()

    places, reviews, labels = records(args.xlsx, args.labels)
    manifest = {
        "dataset_name": "tripadvisor_reviews_v1",
        "xlsx": args.xlsx.name, "xlsx_sha256": hashlib.sha256(args.xlsx.read_bytes()).hexdigest(),
        "labels": args.labels.name, "label_model": next(iter(labels.values()))["model"],
        "places": len(places), "reviews": len(reviews),
    }
    print(f"검증 완료: 장소 {len(places)}, 리뷰 {len(reviews)}, 라벨 {sum(len(l['label']['aspects']) for l in labels.values())}")
    if args.dry_run:
        return
    # 원격 DB를 실수로 바꾸지 않도록 환경 변수 DATABASE_URL은 보지 않는다. 원격은 --database-url로만 지정한다.
    with psycopg.connect(args.database_url or local_url()) as conn, conn.cursor() as cursor:
        annotations = load(cursor, places, reviews, labels, manifest)
    print(f"교체 완료: 장소 {len(places)}, 리뷰 {len(reviews)}, 라벨 {annotations}")


if __name__ == "__main__":
    main()
