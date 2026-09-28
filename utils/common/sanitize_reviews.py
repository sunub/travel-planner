"""로컬 CSV/JSON/XLSX를 대조하고 라벨링용 최소 데이터만 저장한다.

표준 라이브러리만 사용한다. 원문/식별자/삭제어를 로그에 출력하지 않는다.
자유 서술의 인명은 자동 판별을 보장할 수 없어 로컬 삭제어 목록을 함께 받는다.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from schema import CATEGORIES

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
URL = re.compile(
    r"(?:[a-z][a-z0-9+.-]*://|www\.)[^\s<>\[\]\"']+"
    r"|(?<![\w@])(?:[a-z0-9-]+\.)+(?:[a-z]{2,63})(?:/[^\s<>\[\]\"']*)?",
    re.IGNORECASE,
)
EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.IGNORECASE)
HANDLE = re.compile(r"(?<!\w)@[\w.-]+")
PHONE = re.compile(r"(?<!\d)(?:\+82[- .]?|0)(?:\d[- .]?){7,9}\d(?!\d)")
IDENTITY_KEYS = {"author", "author_name", "username", "user_name", "nickname", "display_name", "email", "profile", "profile_url"}


def identity_values(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in IDENTITY_KEYS and isinstance(item, str) and item.strip():
                yield item.strip()
            yield from identity_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from identity_values(item)


def clean_text(text, terms):
    for pattern in (EMAIL, URL, HANDLE, PHONE):
        text = pattern.sub("[삭제]", text)
    for term in sorted(terms, key=len, reverse=True):
        text = re.sub(re.escape(term), "[삭제]", text, flags=re.IGNORECASE)
    return text


def excel_rows(path, sheet_name):
    """읽기 전용 XLSX 처리. 하이퍼링크/주석/숨은 메타데이터는 복사하지 않는다."""
    with ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(node.itertext()) for node in root]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet = next(s for s in workbook.find("m:sheets", NS) if s.attrib["name"] == sheet_name)
        rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(r.attrib["Target"] for r in rels if r.attrib["Id"] == rel_id)
        target = target.lstrip("/") if target.startswith("/") else "xl/" + target
        root = ET.fromstring(archive.read(target))
        rows = []
        for row in root.findall("m:sheetData/m:row", NS):
            cells = {}
            for cell in row:
                column = re.match(r"[A-Z]+", cell.attrib["r"])[0]
                index = 0
                for letter in column:
                    index = index * 26 + ord(letter) - ord("A") + 1
                value = cell.find("m:v", NS)
                text = value.text if value is not None else ""
                if cell.attrib.get("t") == "s":
                    text = shared[int(text)]
                elif cell.attrib.get("t") == "inlineStr":
                    text = "".join(n.text or "" for n in cell.findall("m:is//m:t", NS))
                cells[index - 1] = text
            rows.append(cells)
        header = rows[0]
        return [{name: row.get(index, "") for index, name in header.items()} for row in rows[1:]]


def convert(root, redactions_path):
    places = json.loads((root / "reviews_cleaned_validated.json").read_text(encoding="utf-8-sig"))
    with (root / "location_ids_cleaned_validated.csv").open(encoding="utf-8-sig", newline="") as handle:
        locations = list(csv.DictReader(handle))
    excel = excel_rows(root / "reviews_cleaned_validated.xlsx", "Reviews")
    expected = Counter((str(p["location_id"]), str(r["review_id"]), r["text"]) for p in places for r in p["reviews"])
    actual = Counter((r["location_id"], r["review_id"], r["text"]) for r in excel)
    if expected != actual:
        raise ValueError("JSON과 Excel 리뷰가 일치하지 않습니다. 원본을 확인하세요.")
    if Counter(str(p["location_id"]) for p in places) != Counter(r["location_id"] for r in locations):
        raise ValueError("JSON과 CSV 장소가 일치하지 않습니다.")
    terms = json.loads(redactions_path.read_text(encoding="utf-8-sig"))
    if not isinstance(terms, list) or any(not isinstance(t, str) or not t.strip() for t in terms):
        raise ValueError("삭제어 파일은 빈 문자열이 없는 문자열 배열이어야 합니다.")
    terms = set(terms) | set(identity_values(places))
    output_places = []
    reviews = []
    for number, place in enumerate(places, 1):
        category = place["category"].lower()
        if category not in CATEGORIES:
            raise ValueError("지원하지 않는 장소 카테고리입니다.")
        place_id = f"local_place_{number:04d}"
        output_places.append({
            "place_id": place_id, "category": category,
            "name": clean_text(place["name"], terms), "district": None,
            "address": "", "lat": None, "lng": None, "phone": None,
            "sources": [], "facts": {},
        })
        for review in place["reviews"]:
            text = clean_text(review["text"], terms)
            if not text.strip():
                raise ValueError("빈 리뷰가 있습니다.")
            reviews.append({
                "review_id": f"local_review_{len(reviews) + 1:05d}",
                "place_id": place_id, "category": category,
                "synthetic": False, "tier": "unlabeled", "review": text,
                "language": review["language"],
                "translated": review["source_language"] != review["language"],
                "redacted": text != review["text"],
            })
    # 출력 전체의 URL/연락처/삭제어 잔존 여부를 저장 전에 검사한다.
    payload = json.dumps([output_places, reviews], ensure_ascii=False)
    if clean_text(payload, terms) != payload:
        raise ValueError("정제 결과에 삭제 대상이 남아 있습니다.")
    out = root / "sanitized" / "out"
    out.mkdir(parents=True, exist_ok=True)
    for category in CATEGORIES:
        (out / f"places_{category}.json").write_text(json.dumps(
            [p for p in output_places if p["category"] == category], ensure_ascii=False, indent=2
        ) + "\n", encoding="utf-8")
    (out / "real_reviews.jsonl").write_text("".join(
        json.dumps(r, ensure_ascii=False) + "\n" for r in reviews
    ), encoding="utf-8")
    with (out / "locations_sanitized.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["place_id", "category", "name"])
        writer.writeheader()
        writer.writerows({key: p[key] for key in writer.fieldnames} for p in output_places)
    print(json.dumps({"places": len(output_places), "reviews": len(reviews),
                      "redacted_reviews": sum(r["redacted"] for r in reviews),
                      "categories": dict(Counter(r["category"] for r in reviews)),
                      "json_excel_match": True, "csv_places_match": True}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datas", type=Path, default=Path(__file__).resolve().parents[2] / "datas")
    parser.add_argument("--redactions", type=Path, required=True, help="로컬 삭제어 JSON 배열 (Git 제외)")
    args = parser.parse_args()
    convert(args.datas, args.redactions)


if __name__ == "__main__":
    main()
