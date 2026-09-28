"""라벨 허용값과 라벨 모양 검사.

허용값(aspect, attribute, sentiment, traveler_context)의 정의는 utils/common/schema.py 한 곳에만 있다.
이 모듈은 그 파일을 읽어 올 뿐 값을 다시 적지 않는다. 목록을 바꾸려면 schema.py를 고친다.
"""

import importlib.util
from dataclasses import dataclass
from types import ModuleType
from typing import Final, Literal

from .config import REPO_ROOT

SCHEMA_PATH: Final = REPO_ROOT / "utils" / "common" / "schema.py"
ASPECT_FIELDS: Final = ("category", "attribute", "sentiment", "evidence")


def _load_schema() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tripfit_schema", SCHEMA_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"라벨 정의 파일을 읽을 수 없습니다: {SCHEMA_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


schema: Final = _load_schema()
CATEGORIES: Final[tuple[str, ...]] = schema.CATEGORIES
ATTRIBUTES: Final[dict[str, dict[str, tuple[str, ...]]]] = schema.ATTRIBUTES
SENTIMENTS: Final[frozenset[str]] = schema.SENTIMENTS
TRAVELER_CONTEXTS: Final[frozenset[str]] = schema.TRAVELER_CONTEXTS
CATEGORY_NAMES_KO: Final[dict[str, str]] = schema.CATEGORY_NAMES_KO
ASPECT_NAMES_KO: Final[dict[str, str]] = schema.ASPECT_NAMES_KO
VALUE_NAMES_KO: Final[dict[str, str]] = schema.VALUE_NAMES_KO
TRAVELER_NAMES_KO: Final[dict[str, str]] = schema.TRAVELER_NAMES_KO

IssueKind = Literal["schema", "value", "evidence"]


@dataclass(frozen=True)
class Issue:
    """라벨 문제 하나. kind로 형식 오류(schema), 허용값 위반(value), 근거 오류(evidence)를 구분한다."""

    kind: IssueKind
    message: str


def aspect_issues(aspect: object, category: str, review: str) -> list[Issue]:
    if not isinstance(aspect, dict):
        return [Issue("schema", "aspect가 객체가 아님")]
    missing = [field for field in ASPECT_FIELDS if field not in aspect]
    if missing:
        return [Issue("schema", f"aspect 필드 누락: {missing}")]
    if not all(isinstance(aspect[field], str) for field in ASPECT_FIELDS):
        return [Issue("schema", "aspect 필드 값은 모두 문자열이어야 함")]

    name, issues = aspect["category"], []
    allowed = ATTRIBUTES[category]
    if name not in allowed:
        issues.append(Issue("value", f"{category}에 없는 aspect: {name!r}"))
    elif aspect["attribute"] not in allowed[name]:
        issues.append(Issue("value", f"{name}: 허용되지 않은 attribute {aspect['attribute']!r}"))
    if aspect["sentiment"] not in SENTIMENTS:
        issues.append(Issue("value", f"{name}: 허용되지 않은 sentiment {aspect['sentiment']!r}"))
    evidence = aspect["evidence"]
    if not evidence.strip():
        issues.append(Issue("evidence", f"{name}: evidence가 비어 있음"))
    elif evidence not in review:
        issues.append(Issue("evidence", f"{name}: 리뷰 원문에 없는 evidence {evidence!r}"))
    return issues


def label_issues(label: object, category: str, review: str) -> list[Issue]:
    """label = {"traveler_context": [...], "aspects": [...]}의 문제 목록. 빈 리스트면 통과."""
    if category not in ATTRIBUTES:
        return [Issue("schema", f"알 수 없는 카테고리: {category!r}")]
    if not isinstance(label, dict):
        return [Issue("schema", "label이 JSON 객체가 아님")]
    contexts, aspects = label.get("traveler_context"), label.get("aspects")
    if not isinstance(contexts, list) or not isinstance(aspects, list):
        return [Issue("schema", "traveler_context와 aspects는 리스트여야 함")]

    issues = [
        Issue("value", f"허용되지 않은 traveler_context: {c!r}")
        for c in contexts
        if not isinstance(c, str) or c not in TRAVELER_CONTEXTS
    ]
    for aspect in aspects:
        issues += aspect_issues(aspect, category, review)
    return issues
