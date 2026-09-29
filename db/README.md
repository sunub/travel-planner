# TripFit PostgreSQL

`datasets/train.jsonl`, `datasets/validation.jsonl`, `datasets/test.jsonl`을 적재한다. 원본 JSONL은 수정하지 않으며 Gold/Silver와 train/val/test 분할도 그대로 저장한다.

1. `.env.example`을 참고해 Git에서 제외되는 `db/.env`를 만든다.
2. `db`에서 `docker compose up -d`를 실행한다.
3. 저장소 루트에서 `uv run python db/scripts/load_datasets.py`를 실행한다.

재실행하면 같은 `review_id`의 review, annotation, 동행 유형만 갱신하므로 중복이 생기지 않는다.

상태와 접속:

```bash
cd /home/sang/travel-planner/db
docker compose ps
docker compose exec postgres psql -U sang -d postDB
```

`docker compose down -v`는 DB 볼륨을 지우므로 사용하지 않는다.
