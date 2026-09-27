"""z4(원도심·영도) 장소 목록을 Tour API에서 받아 카테고리별 JSON으로 저장.

결과: datas/yay/out/places_{hotel,restaurant,attraction}.json
필드는 팀 공통 형식(place_id, category, district, name, address, lat, lng, phone, sources, facts)을 따른다.

사용법 (PowerShell, 레포 루트에서):
    $env:TOUR_API_KEY="공공데이터포털 Decoding 키"
    python datas/yay/fetch_places.py
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE_URL = "https://apis.data.go.kr/B551011/KorService2/areaBasedList2"
BUSAN = "26"  # 법정동 시도 코드
SIGUNGU = {"jung": "110", "seo": "140", "dong": "170", "yeongdo": "200"}  # 중구, 서구, 동구, 영도구
CONTENT_TYPES = [("hotel", "32"), ("restaurant", "39"), ("attraction", "12"), ("attraction", "14"), ("attraction", "28")]
ROWS = 100
OUT_DIR = Path(__file__).parent / "out"

def fetch_page(key: str, sigungu: str, content_type: str, page: int) -> dict:
    params = {
        "serviceKey": key, "MobileOS": "ETC", "MobileApp": "TripFit", "_type": "json",
        "lDongRegnCd": BUSAN, "lDongSignguCd": sigungu, "contentTypeId": content_type,
        "numOfRows": ROWS, "pageNo": page,
    }
    url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as res:
        text = res.read().decode("utf-8")
    try:
        return json.loads(text)["response"]["body"]
    except (json.JSONDecodeError, KeyError):
        msg = re.search(r"<returnAuthMsg>(.*?)</returnAuthMsg>", text)
        sys.exit(f"API 오류: {msg.group(1) if msg else text[:200]}")

def items_of(body: dict) -> list[dict]:
    items = body.get("items") or {}
    item = items.get("item", []) if isinstance(items, dict) else []
    return item if isinstance(item, list) else [item]

def to_place(it: dict, category: str, district: str) -> dict:
    facts = {k: v for k, v in (("content_type_id", it.get("contenttypeid")), ("class_code", it.get("lclsSystm3"))) if v}
    return {
        "place_id": f"tourapi:{it['contentid']}",
        "category": category,
        "district": district,
        "name": it["title"].strip(),
        "address": it.get("addr1", "").strip(),
        "lat": float(it["mapy"]) if it.get("mapy") else None,
        "lng": float(it["mapx"]) if it.get("mapx") else None,
        "phone": (it.get("tel") or "").replace(" ", "") or None,
        "sources": ["tourapi"],
        "facts": facts,
    }

def main() -> None:
    key = os.environ.get("TOUR_API_KEY")
    if not key:
        sys.exit("TOUR_API_KEY 환경변수를 먼저 설정해 주세요.")

    places: dict[str, dict] = {}
    calls = 0
    for district, sigungu in SIGUNGU.items():
        for category, content_type in CONTENT_TYPES:
            page = 1
            while True:
                body = fetch_page(key, sigungu, content_type, page)
                calls += 1
                for it in items_of(body):
                    places.setdefault(f"tourapi:{it['contentid']}", to_place(it, category, district))
                if page * ROWS >= int(body.get("totalCount", 0)):
                    break
                page += 1
                time.sleep(0.2)

    OUT_DIR.mkdir(exist_ok=True)
    for category in ("hotel", "restaurant", "attraction"):
        rows = sorted((p for p in places.values() if p["category"] == category), key=lambda p: p["place_id"])
        path = OUT_DIR / f"places_{category}.json"
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{category}: {len(rows)}곳 → {path}")
    print(f"API 호출 {calls}회")

if __name__ == "__main__":
    main()
