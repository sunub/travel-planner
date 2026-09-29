# model-yay — EXAONE-4.0-1.2B LoRA/QLoRA

리뷰 → `{"traveler_context", "aspects"}` 추출 모델을 `LGAI-EXAONE/EXAONE-4.0-1.2B`로 LoRA/QLoRA
파인튜닝하기 위한 코드. 프롬프트 형식·데이터 형식·평가 지표는 팀원 브랜치(`origin/lkh_train`)의
`training/` 코드를 그대로 따른다 (`git show origin/lkh_train:<경로>`로 읽기만 했고, merge·checkout은
하지 않았다). 같은 기준이어야 Gemma(팀원)·Qwen 모델과 비교할 수 있다.

`--method qlora`(4bit NF4 양자화)와 `--method lora`(bf16, 양자화 없음) 두 방식을 지원한다 — LoRA
r/alpha, 학습률, epoch, 데이터 등 나머지 설정은 완전히 같고, base 모델을 4bit로 올리느냐만 다르다.
`train.py`/`evaluate.py`/`select_checkpoint.py` 모두 `--method`로 고른다.

이 폴더 밖의 파일은 건드리지 않는다. `pyproject.toml`에도 의존성을 추가하지 않았으므로, 아래
`requirements.txt`는 이 폴더 전용으로 별도 설치한다.

## 파일

| 파일 | 역할 |
| --- | --- |
| `config.yaml` | 모델·양자화·LoRA·학습 하이퍼파라미터, 데이터 경로 |
| `data.py` | JSONL 레코드 → 대화형 prompt-completion 예제 (schema.py 허용값 사용) |
| `train.py` | LoRA/QLoRA 학습 (`--method qlora\|lora`). LoRA target_modules는 모델에서 자동으로 찾는다 |
| `evaluate.py` | Base/LoRA/QLoRA(또는 특정 checkpoint)를 같은 조건(그리디, 같은 max_new_tokens)으로 평가·채점 |
| `select_checkpoint.py` | epoch별 checkpoint를 validation으로 평가해 best를 고른다 (test는 안 본다) |
| `requirements.txt` | 이 폴더 전용 의존성 |

## 왜 EXAONE 4.0인가 (EXAONE 3.5에서 전환)

원래는 `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct`로 시작했다. EXAONE 3.5(`model_type "exaone"`)는
transformers에 내장돼 있지 않아 HF 리포의 원격 코드(`trust_remote_code=True`)가 필요했는데, 그 원격
코드가 어떤 released transformers 버전과도 API가 맞지 않았다 — `transformers==5.0.0`에는
`AttentionInterface.get_interface`가 없고(`AttributeError`), `transformers==5.17.0`에는 있지만
`create_causal_mask`의 인자(`input_embeds`→`inputs_embeds` 개명, `cache_position` 제거)가 달라서
`TypeError`가 났다. 원격 코드를 `exaone_local/`에 로컬로 복사해 `create_causal_mask` 호출부만
`5.17.0`에 맞게 패치하는 방식으로 일단 해결했었지만, 이어서 `AutoTokenizer.from_pretrained` 쪽에서
`PreTrainedConfig.convert_rope_params_to_dict`/`standardize_rope_params` 오류가 추가로 났다. 시간
관계상 더 파지 않고, transformers 5.17.0에 **완전히 내장**돼 있는 `LGAI-EXAONE/EXAONE-4.0-1.2B`
(`model_type "exaone4"`, 클래스 `Exaone4ForCausalLM`)로 전환했다 — 원격 코드도, `trust_remote_code`도,
로컬 패치(`exaone_local/`, 지웠다)도 전혀 필요 없다. `train.py`/`evaluate.py`/`load_check.py`는 모두
`AutoModelForCausalLM`/`AutoTokenizer`로 그냥 불러온다.

파라미터 수가 7.8B → 1.2B로 작아진 것은 이 전환의 부작용이다 — 팀원 Gemma·Qwen과 크기를 맞춰 비교하려면
추후 더 큰 EXAONE 4.0 변형이 나오면 바꿔야 할 수 있다.

**사고 모드(enable_thinking)는 끈다.** EXAONE 4.0의 `chat_template.jinja`는 `enable_thinking`이
`true`일 때만 `<think>\n`을 열어 두고, 그 외(`false`·미지정)엔 `<think>\n\n</think>\n\n`로 바로
닫는다 — 즉 기본값이 이미 꺼짐이지만, 템플릿이 나중에 바뀌어도 안 흔들리게 `data.py`의
`prompt_text()`가 `apply_chat_template(..., enable_thinking=False)`를 명시적으로 넘긴다(학습·추론
공용 함수라 한 곳만 고치면 둘 다 반영된다). `load_check.py`의 데모 프롬프트도 같은 이유로
`enable_thinking=False`를 넘긴다.

## 의존성 버전

`requirements.txt`는 팀원 브랜치(`origin/lkh_train`)의 Gemma4 QLoRA와 같은 조합 —
`transformers==5.17.0` / `trl==1.14.0` / `peft==0.21.0` / `bitsandbytes==0.50.2` /
`accelerate==1.15.0` — 을 쓴다. `trl==1.14.0`이 `transformers!=5.1.0`만 명시적으로 막아서(알려진
버그, transformers#43780) `5.17.0`과는 충돌하지 않는다. peft/bitsandbytes/accelerate는 팀원과 같은
버전이다 (`uv pip compile`로 충돌 없이 풀리는 것까지는 확인했지만, 팀원처럼 실제 학습으로 검증된
조합은 아니다).

transformers 5는 `TrainingArguments.warmup_ratio`를 없애고 `warmup_steps`가 1 미만 실수면 비율로
취급하도록 바꿨다 (5.0.0부터 이미 이 규칙). `config.yaml`의 `training.warmup_steps: 0.03`과 `train.py`의
`build_sft_config`가 이 규칙을 따른다.

## assistant 응답만 학습하기 (labels 마스킹)

세 단계를 거쳐 지금 방식이 됐다.

1. 팀원 Gemma4 파이프라인과 같은 방식 — `to_sft_example()`로 prompt/completion을 만들고, TRL의
   `SFTConfig(completion_only_loss=True)`가 마스킹을 하게 — 을 그대로 썼다.
2. EXAONE 채팅 템플릿에서 `Mismatch between tokenized prompt and the start of tokenized
   prompt+completion` 경고가 났다. TRL은 이 경고가 나도 "prompt만 토큰화했을 때의 토큰 개수"만 믿고
   그 위치에서 마스킹 경계를 자르므로(`trl/trainer/sft_trainer.py`의 `tokenize_fn`), 실제 내용이
   어긋나면 엉뚱한 자리에서 잘릴 수 있었다. 그래서 prompt만 토큰화한 결과와 prompt+completion 전체를
   토큰화한 결과를 직접 비교해 실제로 같은 구간(공통 접두사)까지만 마스킹하는 방식으로 바꿨다.
3. 그런데 이 공통 접두사 방식도 실제로 문제가 있었다: assistant 턴을 여는 템플릿 텍스트
   `"[|assistant|]"`의 `"]"`와 정답 JSON의 `"{"`가 `"]{"`라는 토큰 하나로 합쳐져서, **학습 때 쓰는
   prompt 토큰 수가 추론 때(evaluate.py가 만드는, 뒤에 아무것도 안 붙는 진짜 prompt) 토큰 수와
   달라졌다.** 모델이 학습 때 한 번도 보지 못한 prompt로 추론을 받게 되는 문제라 단순 경고보다 훨씬
   심각했다.

그래서 지금은 `data.py`의 `encode_example()`이 prompt·completion·종료 토큰을 **절대 같이 토큰화하지
않고** 각각 따로 토큰화해 토큰 id 리스트로 이어 붙인다:

- `prompt_ids`: `prompt_text(record, tokenizer)`(`apply_chat_template(tokenize=False,
  add_generation_prompt=True)`로 문자열을 만든 뒤 `tokenizer(text, add_special_tokens=False)`로
  토큰화)로 만든다. **`evaluate.py`의 `generate_outputs()`도 정확히 같은 `prompt_text()` 함수를
  쓴다** — 두 파일이 비슷하게 각자 짜는 게 아니라 함수 하나를 같이 쓰므로, 학습 prompt와 추론 prompt가
  달라질 수가 없다.
- `completion_ids`: 정답 JSON 문자열만 따로 `tokenizer(text, add_special_tokens=False)`로 토큰화.
- 그 뒤에 `tokenizer.eos_token_id` 하나를 붙인다 (턴이 끝난다는 신호도 학습해야 생성이 끝없이
  이어지지 않는다).

문자열을 합쳐서 다시 토큰화하는 게 아니라 토큰 id 리스트를 이어 붙이기만 하므로, 조각 경계에서 BPE가
다르게 합쳐질 여지 자체가 없다. `labels`는 `prompt_ids` 구간 전부 `-100`, `completion_ids`와 eos
구간만 그대로 둔다. `train.py`는 이 `{input_ids, labels}`를 그대로 `SFTTrainer`에 넘기고, TRL은
`labels` 컬럼이 있으면 자체 마스킹을 건너뛰므로 `SFTConfig`에 `completion_only_loss`는 주지 않는다.
`to_sft_example()`은 `--dry-run` 미리보기(토크나이저 불필요)에만 쓴다.

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
- 평가 결과는 `metrics_<run>_<split>.json`으로 남긴다 (`run`은 `base` 또는 어댑터/checkpoint 폴더
  이름. 팀원처럼 run 디렉터리로 나누지 않으므로, 파일명으로 구분한다).

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

## 학습 (LoRA/QLoRA)

```bash
# 데이터 없이 첫 학습 예제(prompt/completion)만 확인 (--method는 형식상 필요하지만 dry-run엔 안 쓰인다)
uv run python src/travel_planner/model-yay/train.py --config src/travel_planner/model-yay/config.yaml --method qlora --dry-run

# 실제 학습 (데이터 경로를 config.yaml에 채운 뒤)
uv run python src/travel_planner/model-yay/train.py --config src/travel_planner/model-yay/config.yaml --method qlora
uv run python src/travel_planner/model-yay/train.py --config src/travel_planner/model-yay/config.yaml --method lora

# 짧게 스모크 실행: train/validation 앞 32건만, 5 스텝만, 결과는 별도 폴더에
uv run python src/travel_planner/model-yay/train.py --config src/travel_planner/model-yay/config.yaml --method qlora \
    --limit 32 --max-steps 5 --output-dir src/travel_planner/model-yay/artifacts/smoke
```

학습이 끝나면 `output.artifacts_root/<method>`(기본 `src/travel_planner/model-yay/artifacts/qlora`
또는 `.../lora`, `--output-dir`로 바꿀 수 있다) 아래에 `adapter/`(LoRA 어댑터 + tokenizer),
`checkpoints/`(epoch별 체크포인트), `metrics.json`(파라미터 수·최대 VRAM·학습 시간·loss),
`log_history.json`(step별 loss 기록)이 생긴다. qlora와 lora는 서로 다른 폴더를 쓰므로 산출물이 섞이지
않는다. `--max-steps`로 시험 실행하면 `metrics.json`의 `time_estimate`에 스텝 1회 평균 시간(첫 스텝
제외)과 config의 전체 train 데이터 기준 예상 전체 학습 시간이 함께 남는다. 학습 중에는 로그에 현재
스텝/전체 스텝·경과 시간·예상 남은 시간이 주기적으로 찍힌다.

## 평가

```bash
# Base 모델 (기본 --split test, --method 필요 없음 — Base는 항상 bf16)
uv run python src/travel_planner/model-yay/evaluate.py --config src/travel_planner/model-yay/config.yaml

# 학습이 끝난 최종 어댑터, validation으로 (--method는 그 어댑터를 학습할 때 쓴 것과 같아야 한다)
uv run python src/travel_planner/model-yay/evaluate.py --config src/travel_planner/model-yay/config.yaml \
    --method qlora --adapter src/travel_planner/model-yay/artifacts/qlora/adapter --split validation

# epoch 2 checkpoint 하나만 평가 (train.py가 자동으로 남기는 checkpoints/checkpoint-N도 그대로 넣을 수 있다)
uv run python src/travel_planner/model-yay/evaluate.py --config src/travel_planner/model-yay/config.yaml \
    --method lora --adapter src/travel_planner/model-yay/artifacts/lora/checkpoints/checkpoint-2 --split validation
```

`predictions/<run>_<split>.jsonl`(리뷰별 원문 출력·파싱 결과·채점)과 `metrics_<run>_<split>.json`
(aspect F1·attribute/sentiment 정확도·evidence 지표·JSON 유효성 등, 팀원과 동일한 지표)이 생긴다 —
`--adapter`가 있으면 `output.artifacts_root/<method>` 아래(그 방식의 checkpoint·adapter와 같은
폴더), 없으면(Base) `output.artifacts_root` 바로 아래. `run`은 `base` 또는 `--adapter`로 준 폴더
이름(`adapter`, `checkpoint-2` 등)이라서, 여러 checkpoint를 돌아가며 평가해도 서로 덮어쓰지 않는다.

첫 배치를 처리한 직후, 그때까지의 리뷰 1건당 평균 추론 시간을 기준으로 validation 전체와 test 전체를
평가하면 얼마나 걸릴지 한 번 출력한다 (`--split`이나 `--max-samples`로 일부만 돌려도 두 split 전체
기준으로 보여준다).

## Checkpoint 선택 (best 고르기)

**규칙: checkpoint 선택은 validation으로만 한다. test는 최종적으로 고른 best checkpoint로 딱 한 번만
평가한다.** test 점수를 보면서 checkpoint를 고르면(=test로 여러 번 돌려서 제일 잘 나온 걸 고르면) test가
더는 "한 번도 학습에 안 쓰인 기준"이 아니게 되어, 그 이후의 모든 비교(Base vs LoRA vs QLoRA, Gold vs
Gold+Silver)가 낙관적으로 부풀려진다.

```bash
# epoch 3개(config.yaml의 training.save_total_limit: 3) checkpoint를 모두 validation으로 평가
uv run python src/travel_planner/model-yay/select_checkpoint.py --config src/travel_planner/model-yay/config.yaml --method qlora
uv run python src/travel_planner/model-yay/select_checkpoint.py --config src/travel_planner/model-yay/config.yaml --method lora
```

checkpoint별 `eval_loss`(학습 로그, `checkpoint-N/trainer_state.json`에서 읽는다)와 validation
Aspect F1·리뷰 완전 일치율·JSON 준수율을 표로 보여주고, **validation Aspect F1이 가장 높은
checkpoint**를 `<output.artifacts_root>/<method>/best.json`에 기록한다. test는 이 스크립트에서 절대
평가하지 않는다 — best가 정해지면 아래처럼 `evaluate.py`를 `--split test`로 한 번만 따로 돌린다.

```bash
uv run python src/travel_planner/model-yay/evaluate.py --config src/travel_planner/model-yay/config.yaml \
    --method qlora --adapter <best.json의 best_checkpoint_path> --split test
```
