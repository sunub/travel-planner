"""레코드 → Gemma4 SFT 예제 (TRL의 대화형 prompt-completion 형식).

  prompt     = [system, user]   system: 규칙과 출력 형식 / user: 카테고리 + 허용 목록 + 리뷰
  completion = [assistant]      label JSON (traveler_context, aspects)만. metadata는 넣지 않는다.

특수 토큰을 직접 붙이지 않는다. 토큰화는 TRL이 tokenizer.apply_chat_template()으로 하고,
prompt 부분은 loss에서 빠진다 (completion_only_loss). 학습·Base 평가·어댑터 평가가 같은 함수로 prompt를 만든다.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from string import Template

from ..config import resolve_path
from ..labels import ASPECT_NAMES_KO, ATTRIBUTES, CATEGORY_NAMES_KO, TRAVELER_CONTEXTS, TRAVELER_NAMES_KO, VALUE_NAMES_KO


@dataclass(frozen=True)
class PromptTemplates:
    system: str
    user: Template

    @classmethod
    def from_config(cls, config: dict) -> "PromptTemplates":
        return cls.from_files(resolve_path(config["prompt"]["system_template"]), resolve_path(config["prompt"]["user_template"]))

    @classmethod
    def from_files(cls, system_path: Path, user_path: Path) -> "PromptTemplates":
        return cls(system_path.read_text(encoding="utf-8").strip(), Template(user_path.read_text(encoding="utf-8").strip()))


def aspect_table(category: str) -> str:
    """허용 aspect와 attribute 목록. schema.py의 정의를 그대로 글로 옮긴다."""
    lines = []
    for name, values in ATTRIBUTES[category].items():
        allowed = " / ".join(f"{value}({VALUE_NAMES_KO[value]})" for value in values)
        lines.append(f"- {name} ({ASPECT_NAMES_KO[name]}): {allowed}")
    return "\n".join(lines)


def traveler_values() -> str:
    return " / ".join(f"{key}({TRAVELER_NAMES_KO[key]})" for key in sorted(TRAVELER_CONTEXTS))


def build_messages(record: dict, templates: PromptTemplates) -> list[dict]:
    category = record["category"]
    user = templates.user.substitute(
        category=category,
        category_ko=CATEGORY_NAMES_KO[category],
        aspect_table=aspect_table(category),
        traveler_values=traveler_values(),
        review=record["review"],
    )
    return [{"role": "system", "content": templates.system}, {"role": "user", "content": user}]


def target_text(label: dict) -> str:
    """정답 문자열. 키 순서를 고정해 모든 예제가 같은 모양이 되게 한다."""
    canonical = {
        "traveler_context": list(label["traveler_context"]),
        "aspects": [{field: aspect[field] for field in ("category", "attribute", "sentiment", "evidence")} for aspect in label["aspects"]],
    }
    return json.dumps(canonical, ensure_ascii=False)


def to_sft_example(record: dict, templates: PromptTemplates, chat_template_kwargs: dict | None = None) -> dict:
    example = {
        "prompt": build_messages(record, templates),
        "completion": [{"role": "assistant", "content": target_text(record["label"])}],
    }
    if chat_template_kwargs:  # TRL이 예제마다 apply_chat_template에 넘긴다 (예: enable_thinking)
        example["chat_template_kwargs"] = chat_template_kwargs
    return example
