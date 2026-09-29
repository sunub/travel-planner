# TripFit PostgreSQL

`datasets/train.jsonl`, `datasets/validation.jsonl`, `datasets/test.jsonl`을 적재한다. 원본 JSONL은 수정하지 않으며 Gold/Silver와 train/val/test 분할도 그대로 저장한다.

1. `.env.example`을 참고해 Git에서 제외되는 `db/.env`를 만든다.
2. `db`에서 `docker compose up -d`를 실행한다.
3. 저장소 루트에서 `uv run python db/scripts/load_datasets.py`를 실행한다.

재실행하면 같은 `review_id`의 review, annotation, 동행 유형만 갱신하므로 중복이 생기지 않는다.

## Tripadvisor 실제 리뷰로 교체

서비스용 데이터는 Tripadvisor 실제 리뷰 엑셀(`db/tripadvisor_reviews.xlsx`, Git 제외)과 파인튜닝 모델 라벨을 쓴다. 기존 장소·리뷰·라벨을 모두 지우고 새로 넣으므로 먼저 백업한다.

```bash
# 1. 리뷰를 파인튜닝 모델로 라벨링 → db/tripadvisor_labels.jsonl (Ollama 필요)
uv run python db/scripts/label_tripadvisor.py --model hinoonyaso/exaone-tripfit:qlora-v2

# 2. 검증만 먼저, 그다음 로컬 DB 교체 (옵션 없이 실행하면 db/.env의 로컬 DB만 바꾼다)
uv run python db/scripts/load_tripadvisor.py --dry-run
uv run python db/scripts/load_tripadvisor.py

# 3. 원격 DB는 URL을 명시했을 때만 바꾼다
uv run python db/scripts/load_tripadvisor.py --database-url "$SUPABASE_URL"
```

- 모델 라벨은 `annotation_tier='model'`로 저장하고, 스키마에 맞지 않는 출력(카테고리에 없는 aspect, 원문에 없는 evidence 등)은 버린다.
- 실제 리뷰에는 학습 분할·라벨 등급이 없어 `dataset_split`·`label_tier`가 NULL이다.
- 지역·카테고리·동행 유형 시드와 사용자는 지우지 않는다. 장소를 참조하는 스크랩은 함께 지워진다.

상태와 접속:

```bash
cd /home/sang/travel-planner/db
docker compose ps
docker compose exec postgres psql -U sang -d postDB
```

`docker compose down -v`는 DB 볼륨을 지우므로 사용하지 않는다.
