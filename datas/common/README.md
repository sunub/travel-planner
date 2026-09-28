# datas/common — 팀 공통 데이터 도구

팀원마다 원본 데이터는 달라도, 결과는 모두 같은 형식(`schema.py`의 `Place`, `Label`)으로 맞춘다.

```
datas/
  common/            공통 코드, 팀 공유 분할(split.json)
  <이니셜>/          내 원본 데이터 (JSON, CSV 등 자유)
    out/             공통 형식 결과 (도구가 만든다)
      places_{hotel,restaurant,attraction}.json
      real_reviews.jsonl        원본에 리뷰가 있었다면
      runs/<실행 이름>/          합성 리뷰 · Silver · Gold
```

모든 명령은 저장소 루트에서 `uv run python datas/common/<도구>.py ...`로 실행한다.

## 1. 내 데이터를 공통 형식으로

| 원본 | 명령 |
|---|---|
| JSON (장소 목록, 장소 안의 `reviews`) | `normalize.py <이니셜>` |
| TourAPI·부산시 API·숙박 CSV로 새로 수집 | `collect_places.py --member <이니셜> --districts suyeong nam --dry-run` 후 `--dry-run` 없이 |

`normalize.py`는 필드 이름이 달라도 흔한 이름(`location_id`/`contentid`, `name`/`title`, `latitude`/`mapy`, `regions` 등)을 알아서 맞춘다. 못 읽은 레코드는 건너뛰고 이유를 보여준다. 구·군 id는 `schema.DISTRICT_NAMES_KO`를 본다.

`collect_places.py`는 TourAPI 하루 한도(1,000회)를 쓴다. 반드시 `--dry-run`으로 호출 수를 먼저 본다. 숙박 CSV는 내 폴더의 `*숙박*.csv`를 찾아 읽는다.

## 2. Silver 만들기 (로컬 Ollama gemma4)

```bash
uv run python datas/common/make_silver.py --member <이니셜> --run pilot --category hotel restaurant attraction --count 34
```

생성 → 라벨 → Silver 검사를 한 번에 한다. 멈춰도 같은 명령으로 이어서 한다.

## 3. 분할과 Gold 검수

```bash
uv run python datas/common/split_places.py --member <이니셜> --run pilot
uv run python datas/common/review_gold.py datas/<이니셜>/out/runs/pilot/test/silver.jsonl --reviewer <이름>
```

- 분할은 팀 전체가 `common/split.json` 하나를 쓴다. 이미 배정된 장소는 바뀌지 않고 새 장소만 배정된다. 바뀐 `split.json`은 커밋해서 공유한다.
- 검수 방법은 `docs/annotation-guideline.md` 7절을 따른다.

## 도구 목록

| 파일 | 역할 |
|---|---|
| `schema.py` | 장소·라벨 형식, 허용값, 부산 구·군 |
| `paths.py` | 팀원 폴더 규칙, 팀 전체 장소 읽기 |
| `normalize.py` | 원본 JSON → 공통 형식 |
| `collect_places.py` | TourAPI·부산시 API·숙박 CSV 수집 |
| `generate_reviews.py` / `extract_labels.py` / `label_check.py` | 합성 리뷰 생성 / 라벨 추출 / Silver 검사 |
| `make_silver.py` | 위 세 단계를 한 번에 |
| `split_places.py` | 장소 단위 분할, Gold 검수 대기 파일 |
| `review_gold.py` + `.html` | 브라우저 검수 화면 |
