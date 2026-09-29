import logging
from time import perf_counter

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from travel_planner.clients.ollama_client import OllamaClient
from travel_planner.core.config import get_settings
from travel_planner.models.review import Review
from travel_planner.models.scrap import Scrap
from travel_planner.repositories import recommendations as recommendation_repository
from travel_planner.repositories import scraps as scrap_repository
from travel_planner.schemas.recommendations import (
    RecommendationDecision,
    RecommendationRequest,
    RecommendationResponse,
)
from travel_planner.services.recommendation_prompt import build_recommendation_prompt
from travel_planner.services.scrap_processing import process_external_scrap

logger = logging.getLogger(__name__)
MAX_CANDIDATE_CHARS = 6000
MAX_TOTAL_MATERIAL_CHARS = 12000


class RecommendationError(Exception):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


async def _review_material(session: AsyncSession, review: Review) -> str:
    lines = [
        f"리뷰 {review.review_id} (label_tier={review.label_tier}, synthetic={review.is_synthetic}): {review.review_text}"
    ]
    for annotation, aspect_name in await recommendation_repository.get_review_annotations(session, review.review_id):
        if annotation.evidence_text not in review.review_text:
            continue
        lines.append(
            f"분석 근거 ({aspect_name}, {annotation.attribute_value}, {annotation.sentiment}, "
            f"tier={annotation.annotation_tier}): {annotation.evidence_text}"
        )
    companions = await recommendation_repository.get_review_companions(session, review.review_id)
    if companions:
        lines.append(f"동행 정보: {', '.join(companions)}")
    return "\n".join(lines)


async def _candidate(session: AsyncSession, scrap: Scrap, max_chars: int) -> tuple[dict[str, str | int], list[str]]:
    if scrap.source_url:
        content = await process_external_scrap(session, scrap.scrap_id)
        kind = "external_url"
        title = scrap.title or "외부 스크랩"
        material = content
        evidence_sources = [content]
    elif scrap.review_id is not None:
        review = await recommendation_repository.get_review(session, scrap.review_id)
        if review is None:
            raise RecommendationError(f"Scrap {scrap.scrap_id} has no available review", 422)
        kind = "review"
        title = scrap.title or f"리뷰 {review.review_id}"
        material = await _review_material(session, review)
        evidence_sources = [review.review_text]
    elif scrap.place_id is not None:
        place = await recommendation_repository.get_place(session, scrap.place_id)
        if place is None:
            raise RecommendationError(f"Scrap {scrap.scrap_id} has no available place", 422)
        kind = "place"
        title = scrap.title or place.place_name or f"장소 {place.place_id}"
        lines = [f"장소명: {place.place_name or '정보 없음'}", f"주소: {place.address or '정보 없음'}"]
        if place.intro_text:
            lines.append(f"소개: {place.intro_text}")
        evidence_sources = []
        for review in await recommendation_repository.get_place_reviews(session, place.place_id):
            lines.append(await _review_material(session, review))
            evidence_sources.append(review.review_text)
        material = "\n".join(lines)
    else:
        raise RecommendationError(f"Scrap {scrap.scrap_id} has no available target", 422)
    return ({"scrap_id": scrap.scrap_id, "kind": kind, "title": title, "material": material[:max_chars]}, evidence_sources)


async def recommend(
    session: AsyncSession, request: RecommendationRequest, ollama: OllamaClient | None = None
) -> RecommendationResponse:
    started = perf_counter()
    if not get_settings().ollama_model:
        raise RuntimeError("OLLAMA_MODEL is required for recommendations")
    scrap_ids = list(dict.fromkeys(request.scrap_ids))
    scraps = {scrap.scrap_id: scrap for scrap in await scrap_repository.list_scraps(session, scrap_ids)}
    missing = [scrap_id for scrap_id in scrap_ids if scrap_id not in scraps]
    if missing:
        raise RecommendationError(f"Scrap not found: {missing[0]}", 404)

    max_chars = min(MAX_CANDIDATE_CHARS, MAX_TOTAL_MATERIAL_CHARS // len(scrap_ids))
    candidate_data = [await _candidate(session, scraps[scrap_id], max_chars) for scrap_id in scrap_ids]
    candidates = [candidate for candidate, _ in candidate_data]
    prompt = build_recommendation_prompt(request.requirements, candidates)
    ollama_started = perf_counter()
    try:
        raw = await (ollama or OllamaClient()).generate(prompt)
    finally:
        logger.info("ollama_request duration_ms=%.1f", (perf_counter() - ollama_started) * 1000)
    try:
        decision = RecommendationDecision.model_validate_json(raw)
    except ValidationError as exc:
        raise RecommendationError("Ollama returned an invalid recommendation") from exc
    material_by_id = {int(candidate["scrap_id"]): str(candidate["material"]) for candidate in candidates}
    evidence_by_id = {int(candidate["scrap_id"]): sources for candidate, sources in candidate_data}
    if decision.selected_scrap_id not in material_by_id:
        raise RecommendationError("Ollama selected a scrap outside the request")
    for evidence in decision.evidence:
        if (
            evidence.scrap_id not in material_by_id
            or not evidence.text.strip()
            or evidence.text not in material_by_id[evidence.scrap_id]
            or not any(evidence.text in source for source in evidence_by_id[evidence.scrap_id])
        ):
            raise RecommendationError("Ollama returned evidence absent from the supplied material")
    logger.info("recommendation success=true duration_ms=%.1f candidate_count=%d", (perf_counter() - started) * 1000, len(candidates))
    return RecommendationResponse(**decision.model_dump(), compared_scrap_ids=scrap_ids)
