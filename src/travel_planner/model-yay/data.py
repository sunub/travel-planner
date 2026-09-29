"""JSONL 리뷰 레코드 -> EXAONE SFT 예제 (TRL 대화형 prompt-completion 형식).

  prompt     = [system, user]   system: 규칙과 출력 형식 / user: 카테고리 + 허용 aspect·attribute 목록 + 리뷰
  completion = [assistant]      정답 label JSON(traveler_context, aspects)만. review_id 등 metadata는 넣지 않는다.

레코드 모양은 datas/common/README.md의 공통 파이프라인 결과를 따른다:
  {"review_id", "place_id", "category", "synthetic", "review",
   "label": {"traveler_context": [...], "aspects": [{"category","attribute","sentiment","evidence"}]},
   "tier": "gold" | "silver"}

프롬프트 문구는 팀원 브랜치(origin/lkh_train)의 training/prompts/{system.txt,user.txt}를 그대로 옮긴 것이다
(`git show origin/lkh_train:training/prompts/system.txt` 등으로 읽기만 했다. merge·checkout은 하지 않았다).
같은 문구를 써야 Base/LoRA/QLoRA뿐 아니라 팀원의 Gemma·Qwen 모델과도 같은 기준으로 비교할 수 있다.

to_sft_example()의 prompt/completion은 --dry-run 미리보기용(토크나이저 불필요)이고, 실제 학습은
encode_example()이 만드는 미리 토큰화된 {input_ids, labels}를 쓴다.

encode_example()은 prompt와 completion을 절대 같이(한 문자열로 합쳐서) 토큰화하지 않는다. 처음에는
"prompt만 토큰화한 결과"와 "prompt+completion 전체를 토큰화한 결과"를 비교해 공통 접두사까지 마스킹
하는 방식을 썼는데, EXAONE 토크나이저에서 실제 오류가 났다: assistant 턴을 여는 템플릿 텍스트
"[|assistant|]"의 "]"와 정답 JSON의 "{"가 "]{"라는 토큰 하나로 합쳐져 버려서, 학습 때 쓰는 prompt
토큰 수가 추론 때(evaluate.py가 만드는, 뒤에 아무것도 안 붙는 진짜 prompt) 토큰 수와 달라졌다.
즉 모델이 학습 때 본 적 없는 prompt로 추론을 받게 되는, 훨씬 심각한 문제였다.

그래서 지금은 셋을 완전히 따로 토큰화해서 이어 붙인다:
  1. prompt_ids: build_messages() → apply_chat_template(tokenize=False, add_generation_prompt=True)로
     문자열을 만들고, 그 문자열을 tokenizer(text, add_special_tokens=False)로 토큰화한다. evaluate.py의
     generate_outputs()가 추론 prompt를 만드는 것과 정확히 같은 두 단계다(코드도 동일해야 하므로, 이
     두 단계는 이 함수 하나에만 있고 evaluate.py도 이 함수가 쓰는 것과 같은 모양의 문자열을 만든다).
  2. completion_ids: 정답 JSON 문자열을 tokenizer(text, add_special_tokens=False)로 따로 토큰화한다.
  3. eos: tokenizer.eos_token_id 하나를 completion 뒤에 붙인다 (턴이 끝났다는 신호도 학습해야
     생성이 끝없이 이어지지 않는다).
각 조각을 토큰 id 리스트로 만든 다음 이어 붙이므로(문자열을 합쳐서 다시 토큰화하는 게 아니므로),
조각 경계에서 BPE가 다르게 합쳐질 여지 자체가 없다. labels는 prompt_ids 구간 전부 -100, completion과
eos 구간만 그대로 둔다.

라벨 허용값(aspect·attribute·sentiment·traveler_context)은 datas/common/schema.py 한 곳만 본다.
"""

import importlib.util
import json
from pathlib import Path
from string import Template
from types import ModuleType
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
SCHEMA_PATH: Final = REPO_ROOT / "datas" / "common" / "schema.py"
IGNORE_INDEX: Final = -100

SYSTEM_PROMPT: Final = (
    "당신은 여행 리뷰에서 정보를 뽑아 정해진 JSON으로 정리하는 추출기입니다.\n"
    "\n"
    "[규칙]\n"
    "1. 리뷰에 실제로 적힌 내용만 구조화합니다. 추측하지 않고, 리뷰에 없는 내용은 넣지 않습니다.\n"
    "2. evidence는 판단 근거가 되는 구절을 리뷰 원문에서 한 글자도 바꾸지 않고 그대로 복사합니다.\n"
    "3. aspect의 category와 attribute는 사용자 메시지에 주어진 허용 목록에서만 고릅니다.\n"
    "4. attribute는 상태, sentiment는 평가(positive / negative / neutral)입니다. 평가 없이 사실만 말하면 neutral입니다.\n"
    "5. 가운데 값(average, moderate, medium, normal)은 리뷰에 '보통', '무난'처럼 적혀 있을 때만 씁니다.\n"
    "6. 한 aspect에 긍정과 부정이 섞이면 aspect를 두 개로 나눕니다.\n"
    "7. traveler_context는 누구와 갔는지 리뷰에 적혀 있을 때만 넣고, 없으면 빈 리스트로 둡니다.\n"
    "8. 해당하는 aspect가 없으면 aspects를 빈 리스트로 둡니다.\n"
    "\n"
    "[응답 형식]\n"
    "설명 없이 아래 모양의 JSON 객체 하나만 출력합니다.\n"
    '{"traveler_context": [], "aspects": [{"category": "...", "attribute": "...", "sentiment": "...", "evidence": "..."}]}'
)

USER_TEMPLATE: Final = Template(
    "[장소 종류] $category_ko ($category)\n"
    "\n"
    "[허용 aspect와 attribute]\n"
    "$aspect_table\n"
    "\n"
    "[traveler_context 허용값] $traveler_values\n"
    "\n"
    "[리뷰]\n"
    "$review"
)


def _load_schema() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tripfit_schema", SCHEMA_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"라벨 정의 파일을 읽을 수 없습니다: {SCHEMA_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


schema: Final = _load_schema()
ATTRIBUTES: Final[dict[str, dict[str, tuple[str, ...]]]] = schema.ATTRIBUTES
SENTIMENTS: Final = schema.SENTIMENTS
CATEGORY_NAMES_KO: Final[dict[str, str]] = schema.CATEGORY_NAMES_KO
ASPECT_NAMES_KO: Final[dict[str, str]] = schema.ASPECT_NAMES_KO
VALUE_NAMES_KO: Final[dict[str, str]] = schema.VALUE_NAMES_KO
TRAVELER_CONTEXTS: Final = schema.TRAVELER_CONTEXTS
TRAVELER_NAMES_KO: Final[dict[str, str]] = schema.TRAVELER_NAMES_KO


def aspect_table(category: str) -> str:
    """허용 aspect와 attribute 목록. schema.py의 정의를 그대로 글로 옮긴다."""
    lines = []
    for name, values in ATTRIBUTES[category].items():
        allowed = " / ".join(f"{value}({VALUE_NAMES_KO[value]})" for value in values)
        lines.append(f"- {name} ({ASPECT_NAMES_KO[name]}): {allowed}")
    return "\n".join(lines)


def traveler_values() -> str:
    return " / ".join(f"{key}({TRAVELER_NAMES_KO[key]})" for key in sorted(TRAVELER_CONTEXTS))


def build_messages(record: dict) -> list[dict]:
    category = record["category"]
    user = USER_TEMPLATE.substitute(
        category=category,
        category_ko=CATEGORY_NAMES_KO[category],
        aspect_table=aspect_table(category),
        traveler_values=traveler_values(),
        review=record["review"],
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def target_text(label: dict) -> str:
    """정답 문자열. 키 순서를 고정해 모든 예제가 같은 모양이 되게 한다."""
    canonical = {
        "traveler_context": list(label["traveler_context"]),
        "aspects": [
            {field: aspect[field] for field in ("category", "attribute", "sentiment", "evidence")}
            for aspect in label["aspects"]
        ],
    }
    return json.dumps(canonical, ensure_ascii=False)


def to_sft_example(record: dict) -> dict:
    """레코드 -> TRL 대화형 prompt-completion 예제 (--dry-run 미리보기용). 토크나이저가 필요 없다."""
    return {
        "prompt": build_messages(record),
        "completion": [{"role": "assistant", "content": target_text(record["label"])}],
    }


def prompt_text(record: dict, tokenizer) -> str:
    """추론(evaluate.py)과 학습(encode_example)이 똑같이 쓰는 prompt 문자열 하나.

    "다음은 assistant가 답할 차례"라는 표시(add_generation_prompt=True)까지 포함한, 실제로 모델에
    넣는 prompt 그 자체다. evaluate.py의 generate_outputs()도 이 함수로 prompt를 만든다 — 두 곳이
    각자 비슷하게 짜는 게 아니라 함수 하나를 같이 써야, 나중에 한쪽만 고쳐서 어긋나는 일이 없다.

    enable_thinking=False: EXAONE 4.0의 chat_template.jinja는 이 값이 true일 때만 <think>\n을 열어
    두고 나머지(false·미지정)는 <think>\n\n</think>\n\n로 바로 닫는다 — 이미 기본값이 꺼짐이지만,
    템플릿이 나중에 바뀌어도 안 흔들리도록 명시적으로 끈다.
    """
    return tokenizer.apply_chat_template(
        build_messages(record), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def encode_example(tokenizer, record: dict, max_seq_length: int) -> dict:
    """레코드 -> {input_ids, labels}. prompt·completion·종료 토큰을 각각 따로 토큰화해 이어 붙인다.

    - prompt_ids: prompt_text()가 만든 문자열을 tokenizer(text, add_special_tokens=False)로 토큰화.
      evaluate.py가 추론 때 만드는 prompt와 정확히 같은 두 단계(문자열 만들기 → add_special_tokens=False로
      토큰화)를 거친다.
    - completion_ids: 정답 JSON 문자열만 따로 tokenizer(text, add_special_tokens=False)로 토큰화.
    - eos: tokenizer.eos_token_id 하나를 completion 뒤에 붙인다. 턴이 끝난다는 신호도 학습해야
      생성이 끝없이 이어지지 않는다.

    문자열을 합쳐서 다시 토큰화하지 않고 토큰 id 리스트를 이어 붙이기만 하므로, 조각 경계에서 BPE가
    다르게 합쳐질 여지가 없다 (예전 방식은 "[|assistant|]"의 "]"와 JSON의 "{"가 토큰 하나로 합쳐져서
    학습 때 prompt 토큰 수가 추론 때와 달라지는 문제가 있었다).

    labels는 prompt 구간 전부 -100, completion과 eos 구간만 실제 토큰 값 그대로 둔다.
    """
    prompt_ids = tokenizer(prompt_text(record, tokenizer), add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(target_text(record["label"]), add_special_tokens=False)["input_ids"]

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer에 eos_token이 없습니다 (EXAONE tokenizer 설정을 확인하세요).")

    input_ids = prompt_ids + completion_ids + [eos_id]
    if len(input_ids) > max_seq_length:
        raise ValueError(f"{record.get('review_id')}: 토큰 길이 {len(input_ids)}가 max_seq_length({max_seq_length})를 넘습니다.")

    labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids + [eos_id]
    return {"input_ids": input_ids, "labels": labels}


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_records(path: str | Path) -> list[dict]:
    """경로가 상대경로면 저장소 루트 기준으로 읽는다."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return read_jsonl(p)


def format_duration(seconds: float | None) -> str:
    """초 -> '1시간 23분' / '5분 12초'처럼 사람이 읽기 좋은 문자열. train.py/evaluate.py의 진행 로그에서 쓴다."""
    if seconds is None:
        return "계산 불가"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}시간 {minutes}분"
    if minutes:
        return f"{minutes}분 {secs}초"
    return f"{secs}초"


def format_hm(seconds: float | None) -> str:
    """초 -> 'H:MM' 형식. 예상 전체 학습 시간처럼 표 형태로 남길 때 쓴다."""
    if seconds is None:
        return "-"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes = remainder // 60
    return f"{hours}:{minutes:02d}"
