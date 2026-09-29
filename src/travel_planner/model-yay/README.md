# model-yay — EXAONE-3.5-7.8B-Instruct QLoRA

리뷰 → `{"traveler_context", "aspects"}` 추출 모델을 `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct`로 QLoRA
파인튜닝하기 위한 코드. 프롬프트 형식·데이터 형식·평가 지표는 팀원 브랜치(`origin/lkh_train`)의
`training/` 코드를 그대로 따른다 (`git show origin/lkh_train:<경로>`로 읽기만 했고, merge·checkout은
하지 않았다). 같은 기준이어야 Gemma(팀원)·Qwen 모델과 비교할 수 있다.

이 폴더 밖의 파일은 건드리지 않는다. `pyproject.toml`에도 의존성을 추가하지 않았으므로, 아래
`requirements.txt`는 이 폴더 전용으로 별도 설치한다.

## 파일

| 파일 | 역할 |
| --- | --- |
| `config.yaml` | 모델·양자화·LoRA·학습 하이퍼파라미터, 데이터 경로 |
| `data.py` | JSONL 레코드 → 대화형 prompt-completion 예제 (schema.py 허용값 사용) |
| `train.py` | 4bit QLoRA 학습. LoRA target_modules는 모델에서 자동으로 찾는다 |
| `evaluate.py` | Base/QLoRA를 같은 조건(그리디, 같은 max_new_tokens)으로 평가·채점 |
| `requirements.txt` | 이 폴더 전용 의존성 |

## 데이터셋

`config.yaml`의 `data.{train_path,validation_path,test_path}`는 팀 공통 데이터셋
(`datas/yay/in/datasets/{train,validation,test}.jsonl`)을 가리킨다.

| split | 건수 | tier | 카테고리 |
| --- | --- | --- | --- |
| train | 3,087 | silver | attraction 1186 · hotel 651 · restaurant 1250 |
| validation | 371 | silver | attraction 154 · hotel 80 · restaurant 137 |
| test | 172 | gold (+ `silver_label`, `decision` 필드) | attraction 54 · hotel 53 · restaurant 65 |

`data.py`는 세 파일 모두 `category`/`review`/`label` 필드만 읽는다. `generation`, `labeling`,
`silver_label`, `decision`, `split`, `source_member` 같은 부가 필드는 무시되므로 그대로 읽을 수 있다.
정답은 항상 `label` 필드다 (`test.jsonl`의 `silver_label`은 Teacher-LLM 초안이라 정답으로 쓰지 않는다).

## 아직 정해지지 않은 것

- **Train/Validation/Test 분할 방법**: 이 데이터셋은 이미 나뉜 상태로 받는다. 팀원처럼 `place_id` 단위
  group split을 다시 만들지, 이 분할을 그대로 쓸지는 아직 정하지 않았다.
- Base와 QLoRA 평가 결과는 `metrics_base.json` / `metrics_qlora.json`으로 따로 남긴다 (팀원처럼 run
  디렉터리로 나누지 않으므로, 파일명으로 구분한다).

이 상태에서는 `train.py`/`evaluate.py`를 `--dry-run` 없이 실행하면 데이터 경로가 없다는 안내와 함께
멈춘다. 모델 다운로드·실제 학습은 아직 하지 않는다.

## 설치

```bash
uv venv .venv-exaone  # 또는 팀에서 쓰는 가상환경
.venv-exaone/Scripts/activate  # Windows
pip install -r src/travel_planner/model-yay/requirements.txt
```

GPU 학습에는 CUDA용 torch가 필요하다. PyPI 기본 torch가 CPU 전용이면 팀 GPU의 CUDA 버전에 맞는
휠을 따로 설치한다 (예: `pip install torch --index-url https://download.pytorch.org/whl/cu126`).

저장소 루트의 `.env`에 `HF_TOKEN`, `HF_HOME`을 채운다 (`.env`는 Git에서 제외된다).

## 학습 (QLoRA)

```bash
# 데이터 없이 첫 학습 예제(prompt/completion)만 확인
uv run python src/travel_planner/model-yay/train.py --config src/travel_planner/model-yay/config.yaml --dry-run

# 실제 학습 (데이터 경로를 config.yaml에 채운 뒤)
uv run python src/travel_planner/model-yay/train.py --config src/travel_planner/model-yay/config.yaml

# 짧게 스모크 실행
uv run python src/travel_planner/model-yay/train.py --config src/travel_planner/model-yay/config.yaml --max-samples 8
```

학습이 끝나면 `output.artifacts_root`(기본 `src/travel_planner/model-yay/artifacts`) 아래에
`adapter/`(LoRA 어댑터 + tokenizer), `checkpoints/`(epoch별 체크포인트), `metrics.json`(파라미터 수·최대
VRAM·학습 시간·loss), `log_history.json`(step별 loss 기록)이 생긴다.

## 평가

```bash
# Base 모델
uv run python src/travel_planner/model-yay/evaluate.py --config src/travel_planner/model-yay/config.yaml

# QLoRA 어댑터
uv run python src/travel_planner/model-yay/evaluate.py --config src/travel_planner/model-yay/config.yaml \
    --adapter src/travel_planner/model-yay/artifacts/adapter
```

`predictions/<method>_<split>.jsonl`(리뷰별 원문 출력·파싱 결과·채점)과 `metrics_<method>.json`
(aspect F1·attribute/sentiment 정확도·evidence 지표·JSON 유효성 등, 팀원과 동일한 지표)이 생긴다.
