# training/data

## 원본 데이터

| 항목 | 값 |
| --- | --- |
| 파일 | `datas/lkh/out/gold_review.jsonl` (`.gitignore`의 `datas/*/out/`라서 Git에 없음) |
| 레코드 | 1,147건 · 장소 444곳 · 모두 `tier: gold`, `synthetic: false` |
| 카테고리 | 식당 387 · 호텔 380 · 관광지 380 |
| 라벨 | aspect 1,494개 · aspect가 없는 리뷰 91건 · evidence는 모두 원문의 부분 문자열 |

원본 파일은 코드가 읽기만 하고 고치지 않는다. 설정의 `data.dataset_path`로 위치를 바꿀 수 있다.

```json
{"review_id": "...", "place_id": "lkh:...", "category": "hotel", "synthetic": false, "review": "리뷰 원문",
 "label": {"traveler_context": [], "aspects": [{"category": "amenities", "attribute": "available",
           "sentiment": "positive", "evidence": "리뷰 원문에 있는 구절"}]},
 "tier": "gold"}
```

- 모델이 생성하는 정답은 `label`뿐이다. `review_id`, `place_id`, `tier`, `synthetic`는 데이터 관리용 metadata다.
- 허용값(aspect, attribute, sentiment, traveler_context)은 `utils/common/schema.py`에만 정의되어 있다.
- 불러올 때 모든 레코드를 검사한다 (필수 필드, 허용값, evidence가 원문에 있는지, review_id 중복). 하나라도 틀리면 멈춘다.

## Split: `splits/gold_split_v1.json`

`place_id` 단위 group split이다. 같은 장소가 Train과 Validation/Test에 함께 들어가지 않는다.

| | train | validation | test |
| --- | ---: | ---: | ---: |
| attraction | 304 | 38 | 38 |
| hotel | 304 | 38 | 38 |
| restaurant | 309 | 39 | 39 |
| 합계 | 917 | 115 | 115 |
| 장소 수 | 345 | 48 | 51 |

만드는 방법 (`src/travel_planner/finetune/data/split.py`):

1. 카테고리마다 따로 나눈다. 목표는 리뷰 수 기준 80 / 10 / 10이다.
2. 리뷰가 30건 이상인 대형 장소 3곳은 Train에 고정한다. 호텔 한 곳이 179건(호텔 리뷰의 47%)이라 이 장소가 Test에 가면 Test가 한 장소 위주가 된다.
3. 나머지 장소는 `seed 42`와 카테고리로 만든 난수로 섞는다. 그다음 목표 리뷰 수를 넘지 않는 장소부터 Test, Validation 순서로 채운다.

manifest에는 seed, 비율, 원본 파일 SHA-256, 장소별 배정, split별 review_id 목록이 들어 있다. 모든 Base / LoRA / QLoRA 실험이 이 파일 하나를 쓴다.

- 다시 만들 필요가 없다. `prepare_split.py`는 manifest가 이미 있으면 덮어쓰지 않는다.
- 원본의 라벨만 고쳐져 SHA-256이 달라져도 review_id가 같으면 같은 분할을 쓴다. metrics.json의 `data.dataset_matches_split_source`에 일치 여부가 남는다.
- 리뷰가 추가되는 등 분할을 새로 해야 하면 `split.version`과 `data.split_manifest`를 `gold_split_v2`로 바꿔 새 파일을 만든다. 버전이 다른 실험끼리는 점수를 비교하지 않는다.
- Test는 Gold만 쓴다. Test에 Gold가 아닌 레코드가 있으면 학습과 평가를 시작하지 않는다.

## Gold + Silver 실험

`data.extra_train_paths`에 Silver JSONL을 넣는다. 추가 데이터는 Train에만 들어간다. Validation/Test 장소의 리뷰는 자동으로 빠지고, 뺀 건수는 metrics.json의 `data.extra_train`에 남는다. Test는 그대로라서 Gold Only 실험과 바로 비교할 수 있다.
