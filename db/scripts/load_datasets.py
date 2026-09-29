"""Load canonical TripFit datasets JSONL into PostgreSQL; safe to rerun."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"
DB_DIR = ROOT / "db"
CATEGORY_NAMES = {"hotel": "호텔", "restaurant": "식당", "attraction": "관광지"}
COMPANION_NAMES = {"solo": "혼자", "couple": "연인", "friends": "친구", "parents": "부모님", "family_with_kids": "아이 동반 가족"}


def records() -> tuple[list[dict], dict]:
    manifest = json.loads((DATASETS / "manifest.json").read_text(encoding="utf-8"))
    result, seen = [], set()
    expected = {"train": ("silver", "train"), "validation": ("silver", "val"), "test": ("gold", "test")}
    for file_name, (tier, split) in expected.items():
        for line in (DATASETS / f"{file_name}.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["review_id"] in seen or record["tier"] != tier or record["split"] != split:
                raise ValueError(f"데이터 분할 오류: {record['review_id']}")
            seen.add(record["review_id"])
            for aspect in record["label"]["aspects"]:
                if aspect["evidence"] not in record["review"]:
                    raise ValueError(f"evidence 오류: {record['review_id']}")
                if aspect["sentiment"] not in {"positive", "negative", "neutral"}:
                    raise ValueError(f"sentiment 오류: {record['review_id']}")
            result.append(record)
    return result, manifest


def db_url() -> str:
    load_dotenv(DB_DIR / ".env", override=False)
    return os.getenv("DATABASE_URL") or "postgresql://{user}:{password}@{host}:{port}/{database}".format(
        user=os.environ["POSTGRES_USER"], password=os.environ["POSTGRES_PASSWORD"],
        host=os.getenv("POSTGRES_HOST", "localhost"), port=os.getenv("POSTGRES_PORT", "5432"),
        database=os.environ["POSTGRES_DB"],
    )


def one(cursor, sql: str, params: tuple) -> int:
    cursor.execute(sql, params)
    return cursor.fetchone()[0]


def upsert(cursor, record: dict) -> int:
    category_id = one(cursor, """INSERT INTO place_categories (category_code, category_name) VALUES (%s, %s)
        ON CONFLICT (category_code) DO UPDATE SET category_name = EXCLUDED.category_name RETURNING category_id""", (record["category"], CATEGORY_NAMES[record["category"]]))
    place_id = one(cursor, """INSERT INTO places (category_id, source, source_place_id) VALUES (%s, 'datasets', %s)
        ON CONFLICT (source, source_place_id) DO UPDATE SET category_id = EXCLUDED.category_id, updated_at = NOW() RETURNING place_id""", (category_id, record["place_id"]))
    review_id = one(cursor, """INSERT INTO reviews (place_id, source, external_review_id, source_member, dataset_split, label_tier, is_synthetic, review_text, raw_record)
        VALUES (%s, 'datasets', %s, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (source, external_review_id) DO UPDATE SET place_id=EXCLUDED.place_id, source_member=EXCLUDED.source_member,
          dataset_split=EXCLUDED.dataset_split, label_tier=EXCLUDED.label_tier, is_synthetic=EXCLUDED.is_synthetic,
          review_text=EXCLUDED.review_text, raw_record=EXCLUDED.raw_record, updated_at=NOW() RETURNING review_id""",
        (place_id, record["review_id"], record.get("source_member"), record["split"], record["tier"], record["synthetic"], record["review"], json.dumps(record, ensure_ascii=False)))
    cursor.execute("DELETE FROM review_annotations WHERE review_id = %s", (review_id,))
    cursor.execute("DELETE FROM review_companions WHERE review_id = %s", (review_id,))
    for code in record["label"]["traveler_context"]:
        companion_id = one(cursor, """INSERT INTO companion_types (companion_code, companion_name) VALUES (%s, %s)
            ON CONFLICT (companion_code) DO UPDATE SET companion_name=EXCLUDED.companion_name RETURNING companion_type_id""", (code, COMPANION_NAMES.get(code, code)))
        cursor.execute("INSERT INTO review_companions (review_id, companion_type_id) VALUES (%s, %s)", (review_id, companion_id))
    seen_annotations = set()
    for item in record["label"]["aspects"]:
        identity = (item["category"], item["attribute"], item["sentiment"], item["evidence"])
        if identity in seen_annotations:
            continue
        seen_annotations.add(identity)
        aspect_id = one(cursor, """INSERT INTO aspects (category_id, aspect_name) VALUES (%s, %s)
            ON CONFLICT (category_id, aspect_name) DO UPDATE SET is_active=TRUE RETURNING aspect_id""", (category_id, item["category"]))
        cursor.execute("""INSERT INTO aspect_values (aspect_id, value_name) VALUES (%s, %s)
            ON CONFLICT (aspect_id, value_name) DO UPDATE SET is_active=TRUE""", (aspect_id, item["attribute"]))
        start = record["review"].index(item["evidence"])
        cursor.execute("""INSERT INTO review_annotations (review_id, aspect_id, attribute_value, sentiment, evidence_text, evidence_start, evidence_end, annotation_tier, created_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""", (review_id, aspect_id, item["attribute"], item["sentiment"], item["evidence"], start, start + len(item["evidence"]), record["tier"], f"dataset:{record.get('source_member', 'unknown')}"))
    return len(seen_annotations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows, manifest = records()
    annotation_count = sum(len(row["label"]["aspects"]) for row in rows)
    print(f"검증 완료: reviews={len(rows):,}, annotations={annotation_count:,}, splits={dict(Counter(row['split'] for row in rows))}")
    if args.dry_run:
        return
    with psycopg.connect(db_url()) as conn, conn.cursor() as cursor:
        loaded = sum(upsert(cursor, row) for row in rows)
        cursor.execute("INSERT INTO dataset_imports (dataset_name, manifest, review_count, annotation_count) VALUES (%s, %s::jsonb, %s, %s)", (manifest["dataset_name"], json.dumps(manifest, ensure_ascii=False), len(rows), loaded))
    print(f"적재 완료: reviews={len(rows):,}, annotations={loaded:,}")


if __name__ == "__main__":
    main()
