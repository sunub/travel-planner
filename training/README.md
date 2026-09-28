# training — Gemma4 LoRA / QLoRA 학습 · 평가 환경

## 1. 이 브랜치(`lkh_train`)의 목적

리뷰 → `{"traveler_context", "aspects"}` 추출 모델을 Gemma4로 파인튜닝하고, Base / LoRA / QLoRA를 **같은 Gold Test**에서 비교한다.
이 브랜치에는 학습·평가·실험 기록 코드만 있다. 프론트엔드, 백엔드 API, 라벨 검증 UI는 다른 브랜치에서 만든다.

담당 범위: 데이터 검사 · place_id 단위 분할 · SFT 데이터 변환 · LoRA/QLoRA 학습 · Base/어댑터 평가 · loss·VRAM·학습 시간·어댑터 크기 기록 · run 버전 관리 · 결과 비교표.

## 2. 디렉터리 구조

```
src/travel_planner/finetune/     라이브러리 코드 (uv가 설치하는 패키지라서 스크립트에서 바로 import된다)
  config.py                      YAML 설정 읽기 (defaults 상속, --set 덮어쓰기)
  labels.py                      utils/common/schema.py의 허용값을 읽어 오고 라벨 모양을 검사
  data/dataset.py                JSONL 읽기 · 레코드 검사 (원본은 읽기만)
  data/split.py                  place_id group split · manifest · 누수 검사
  data/sft.py                    레코드 → prompt/completion 대화 (Gemma4 chat template용)
  data/pipeline.py               설정 → train/validation/test 레코드 (+ Silver 추가 경로)
  training/model.py              tokenizer · 모델(16bit / 4bit NF4) · LoRA 설정
  training/trainer.py            TRL SFTTrainer 학습 한 번 = run 하나
  inference/generate.py          greedy 생성 (채점 없음)
  inference/run_eval.py          Base/어댑터를 불러와 생성 → 채점 → metrics.json
  evaluation/parser.py           출력 문자열 → JSON · 형식/허용값/근거 검사
  evaluation/metrics.py          채점 지표 (순수 함수)
  evaluation/evaluator.py        정답 + 출력 → 전체·카테고리별 지표
  utils/tracking.py              run 이름 · 디렉터리 · git 정보 · metrics.json · summary.md
  utils/gpu.py                   VRAM 측정
  utils/compare.py               여러 metrics.json → 표 · CSV

training/
  configs/common.yaml            공통 설정 (model_id, 데이터, prompt, 출력 경로, seed)
  configs/sft_common.yaml        LoRA · QLoRA 공통 하이퍼파라미터
  configs/base_eval.yaml         Base 기준선
  configs/lora_gold_v1.yaml      LoRA · Gold Only
  configs/qlora_gold_v1.yaml     QLoRA · Gold Only
  prompts/system.txt, user.txt   학습·평가 공통 prompt 템플릿 (코드에 하드코딩하지 않음)
  data/splits/gold_split_v1.json place_id 분할 manifest (커밋)
  scripts/                       prepare_split.py · train.py · evaluate.py · compare_runs.py
  experiments/<run>/             config.yaml · metrics.json · summary.md · loss_history.csv (커밋)
  artifacts/<run>/               adapter/ · checkpoints/ · logs/ · predictions/ (Git 제외)
  tests/                         pytest (GPU 불필요)
```

제안된 `training/src/` 대신 `src/travel_planner/finetune/`에 코드를 둔 이유가 있다. `AGENTS.md`가 패키지 코드를 `src/travel_planner/`에 두도록 정했고, uv가 그 패키지만 설치한다. 그래서 `sys.path` 조작 없이 import된다. 설정, 데이터, 기록, 스크립트는 모두 `training/` 아래에 있다.

## 3. `gold_review.jsonl` 위치

`datas/lkh/out/gold_review.jsonl` (Git 제외). 구조와 분할은 [`data/README.md`](data/README.md)를 본다. 다른 위치의 파일을 쓰려면 `data.dataset_path`를 바꾼다.

## 4. 환경 설치

패키지 관리는 저장소 루트의 `uv` + `pyproject.toml`을 그대로 쓴다.

```bash
uv sync
```

GPU 학습에는 CUDA용 torch가 필요하다. PyPI의 Windows torch는 CPU 전용이다. `pyproject.toml`은 바꾸지 않기로 했으므로 직접 설치한다. RTX 3060 드라이버는 CUDA 12.6을 지원한다.

```bash
uv pip install --reinstall torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
```

```bash
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> ⚠️ `uv run`과 `uv sync`는 기본으로 `uv.lock`에 맞춰 환경을 되돌리므로 CPU torch가 다시 설치된다. 설치한 뒤에는 명령마다 `uv run --no-sync`를 쓰거나, PowerShell에서 `$env:UV_NO_SYNC = "1"`을 설정한다. 아래 명령은 `UV_NO_SYNC=1`이 설정되어 있다고 가정한다.

주요 버전 (uv.lock): transformers 5.17.0 (Gemma4 공식 지원), trl 1.14.0, peft 0.21.0, bitsandbytes 0.50.2, accelerate 1.15.0.

**원본 모델:** `google/gemma-4-E4B-it` (`configs/common.yaml`의 `model.name_or_path`). Base 평가와 LoRA/QLoRA 학습이 모두 이 모델을 쓴다. Teacher인 Ollama `gemma4:e4b`와 같은 E4B-it 계열이지만, 이쪽은 양자화하지 않은 Hugging Face 원본 가중치라 학습할 수 있다. 다른 모델로 실험하려면 `--model <id>`로 바꾼다.

## 5. Split 생성

이미 `data/splits/gold_split_v1.json`이 커밋되어 있으므로 다시 만들 필요가 없다. 기존 파일은 덮어쓰지 않는다.

```bash
uv run python training/scripts/prepare_split.py --config training/configs/common.yaml --dry-run
```

## 6. Base 평가 (학습 전 기준선)

```bash
uv run python training/scripts/evaluate.py --config training/configs/base_eval.yaml
```

`training/experiments/base_v001/metrics.json`이 생긴다. 예측 원문은 `training/artifacts/base_v001/predictions/test.jsonl`에 저장된다.

## 7. LoRA 학습

```bash
uv run python training/scripts/train.py --config training/configs/lora_gold_v1.yaml --dry-run
uv run python training/scripts/train.py --config training/configs/lora_gold_v1.yaml
```

`--dry-run`은 모델 없이 데이터 수와 첫 prompt/completion만 보여준다. 학습이 끝나면 같은 run에서 Gold Test 평가까지 한다 (`evaluation.run_after_train`).

> ⚠️ E4B는 임베딩을 포함하면 약 8B 파라미터라 16bit 가중치만 약 16GB다. 12GB GPU에서는 16bit LoRA가 메모리 부족으로 실패할 가능성이 높다. 더 큰 GPU에서 돌리거나 QLoRA를 먼저 한다.

## 8. QLoRA 학습

```bash
uv run python training/scripts/train.py --config training/configs/qlora_gold_v1.yaml
```

먼저 짧게 돌려 보고 싶으면 `--max-samples 8 --set training.num_train_epochs=1`을 붙인다. 스모크 run도 `experiments/`에 기록되니 커밋하지 않을 run 폴더는 지운다.

## 9. 어댑터 평가

```bash
uv run python training/scripts/evaluate.py --config training/configs/qlora_gold_v1.yaml --adapter training/artifacts/qlora_gold_v001/adapter --run-name qlora_gold_v001
```

`--run-name`을 주면 그 run의 metrics.json에 결과를 더하고, 없으면 `<experiment>_eval_v00N` run을 새로 만든다. 하이퍼파라미터를 고를 때는 `--split validation`을 쓴다. Test를 보고 설정을 고치지 않는다 (`docs/evaluation-plan.md` 1-3).

## 10. 여러 run 비교

```bash
uv run python training/scripts/compare_runs.py --csv training/experiments/comparison.csv
```

인자가 없으면 `training/experiments/*/metrics.json`을 모두 읽는다. 폴더나 파일을 인자로 주면 그것만 비교한다. `--all-columns`를 붙이면 터미널 표에 모든 열이 나온다. CSV에는 항상 모든 열이 들어간다.

## 11. Artifact와 Git 관리

| 커밋한다 (`training/`) | 커밋하지 않는다 (`.gitignore`) |
| --- | --- |
| 코드, 설정, prompt 템플릿 | 모델 원본·weight, HF cache (`.cache/`, `hf_cache/`) |
| split manifest | 어댑터 weight (`*.safetensors`, `*.bin`) |
| `experiments/<run>/` metrics · summary · config · loss CSV | 체크포인트, optimizer state (`checkpoints/`, `checkpoint-*/`, `optimizer.pt`) |
| README, pyproject / uv.lock | `artifacts/`, wandb, mlruns, GPU 프로파일 파일 |

- 예측 결과(`predictions/*.jsonl`)에는 리뷰 원문이 들어 있다. 원본 데이터처럼 Git에 올리지 않고 artifacts에 둔다.
- 저장 위치는 `output.artifacts_root` 또는 `--artifacts-root`로 바꾼다 (예: `D:/tripfit_artifacts`). 코드는 어댑터 위치를 가정하지 않는다.
- 결과를 커밋하기 전에 `metrics.json`의 `git.dirty`가 `false`인지 본다. 커밋하지 않은 코드로 돌린 run은 재현할 수 없다.

### metrics.json 필드

`experiment_name`, `run_name`, `timestamp`, `git{commit,branch,dirty}`, `model_id`, `method`(base/lora/qlora), `seed`, `precision`, `device`,
`data{dataset_path, dataset_version, dataset_sha256, split_version, counts{train,validation,test}, extra_train}`,
`hyperparameters{lora, training, quantization}`,
`training{final_train_loss, mean_train_loss, eval_loss, best_eval_loss, duration_sec, peak_vram_allocated_gb, peak_vram_reserved_gb, adapter_size_mb, trainable_params, token_lengths, adapter_path}`,
`evaluation{split, num_samples, overall, by_category, inference{duration_sec, sec_per_review, tokens_per_sec, peak_vram_*}, decoding}`.
Base run은 `hyperparameters`와 `training`이 `null`이다.

### 평가 지표 정의

정의는 `docs/evaluation-plan.md` 2절을 따른다. 코드는 `evaluation/metrics.py` 맨 위 주석에 있다.

- **짝짓기:** 한 리뷰 안에서 aspect category가 같은 예측·정답끼리 짝을 짓는다. 같은 category가 여럿이면 evidence 겹침이 큰 쌍부터 짝짓는다. 남은 예측은 FP, 남은 정답은 FN이다.
- **Aspect F1:** 짝지어진 쌍 수로 계산한 micro P/R/F1이다. aspect 종류별 macro F1과 리뷰별 category 집합 F1(`aspect_set_f1`)도 함께 낸다.
- **Attribute / Sentiment Accuracy:** 짝지어진 쌍 중 값이 같은 비율이다. aspect를 놓친 것과 값을 틀린 것을 분리한다. Sentiment는 macro F1도 낸다.
- **Evidence:**
  - `evidence_in_source_rate`: 전체 예측 aspect 중 evidence가 원문의 부분 문자열인 비율
  - `evidence_exact_match`: 짝지어진 쌍 중 evidence가 정답과 글자까지 같은 비율
  - `evidence_overlap_f1`: 원문에서의 글자 위치 겹침 F1
- **JSON:**
  - `json_parse_rate`: JSON 객체를 읽었는가
  - `schema_valid_rate`: 필수 필드와 타입이 맞는가
  - `json_valid_rate`: 위 조건 + 허용되지 않은 값이 없는가. 비교표의 "JSON Valid"
  - `invalid_value_rate`: 허용값 위반 수 ÷ 예측 aspect 수
- **채점 규칙:** 파싱에 실패한 출력은 빈 예측으로 채점한다. 필드가 빠진 aspect는 버리고, 허용값을 벗어난 aspect는 남겨서 틀린 예측으로 센다.
- **그 밖:** `traveler_context_f1`, `exact_review_match`(리뷰를 통째로 맞힌 비율), `empty_review_accuracy`(정답이 비었을 때 예측도 빈 비율).
- 모든 지표는 전체(`overall`)와 카테고리별(`by_category`)로 나온다. 분모가 0이면 `null`이다.

### SFT 형식

- `prompt = [system, user]`, `completion = [assistant]`인 TRL 대화형 prompt-completion 데이터다.
  - system: 규칙과 출력 형식
  - user: 카테고리, 그 카테고리의 허용 aspect/attribute 목록, 리뷰
  - completion: label JSON(키 순서 고정)
- 특수 토큰을 직접 붙이지 않는다. TRL이 `tokenizer.apply_chat_template()`로 토큰화하고, `completion_only_loss=True`라서 assistant 응답에만 loss가 걸린다.
- Gemma4 chat template에는 `{% generation %}` 표시가 없다. 그래서 TRL의 `assistant_only_loss` 대신 prompt-completion 방식을 쓴다.
- 생각 모드는 `prompt.chat_template_kwargs.enable_thinking: false`로 끈다. 학습과 평가가 같은 값을 쓴다.
- 학습 전에 모든 예제의 토큰 길이를 재고, `max_length`를 넘는 예제가 있으면 멈춘다. 정답 JSON이 잘리지 않게 하기 위해서다.

### LoRA target modules

Gemma4 E4B 체크포인트는 멀티모달이다 (`Gemma4ForConditionalGeneration`). vision/audio 타워에도 `q_proj` 같은 이름이 있지만, 이 모듈들은 PEFT가 감쌀 수 없는 `Gemma4ClippableLinear`다. 그래서 `target_modules`는 텍스트 타워만 고르는 정규식을 쓴다.

`.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$`

KV를 공유하는 뒤쪽 레이어에는 `k_proj`/`v_proj`가 없어서 해당 레이어에서는 자동으로 빠진다. `tests/test_gemma4_integration.py`가 가중치 없이 meta device 모델로 이것을 확인한다.

## 12. 다른 브랜치에서 어댑터 사용하기

`artifacts/<run>/adapter/` 폴더 하나에 어댑터를 쓰는 데 필요한 것이 모두 들어 있다. 폴더째 복사하면 된다.

```
adapter/
  adapter_config.json, adapter_model.safetensors   PEFT 어댑터
  tokenizer files                                  학습 때 쓴 tokenizer
  prompt/system.txt, prompt/user.txt               학습 때 쓴 prompt 템플릿
  tripfit_adapter.json                             base_model_id, method(lora/qlora), 4bit 설정, chat_template_kwargs, split·데이터 버전, git commit
```

```python
import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

adapter = "D:/tripfit_artifacts/qlora_gold_v001/adapter"
meta = json.load(open(f"{adapter}/tripfit_adapter.json", encoding="utf-8"))
quant = None
if meta["method"] == "qlora":  # 학습 때와 같은 4bit 설정 (meta["quantization"])
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(meta["base_model_id"], dtype="auto", quantization_config=quant, device_map={"": 0})
model = PeftModel.from_pretrained(model, adapter)
tokenizer = AutoTokenizer.from_pretrained(adapter)
```

같은 입력 형식을 쓰려면 `prompt/`의 템플릿과 `meta["chat_template_kwargs"]`를 그대로 쓴다. 이 저장소가 있으면 `travel_planner.finetune.data.sft.build_messages()`를 쓸 수 있다. `evaluate.py --adapter <경로>`는 어댑터가 어디에 있든 평가한다.

## 테스트

```bash
uv run pytest
```

GPU와 모델 다운로드 없이 돌아간다. 확인하는 것은 다음과 같다.

- JSONL 로딩, 스키마, place_id 누수, manifest 일관성
- prompt/target 생성, 지표 계산, config 로딩, run 버전 관리, 비교표
- 8건 스모크 파이프라인
- Gemma4 LoRA target (meta device)

실제 tokenizer로 chat template 경계를 확인하려면 `TRIPFIT_TOKENIZER=<model_id>`를 설정하고 실행한다. 이때 tokenizer 파일을 받는다.
