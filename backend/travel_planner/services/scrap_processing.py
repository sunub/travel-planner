import logging
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from travel_planner.clients.jina_client import JinaClient, JinaError
from travel_planner.models.scrap import ScrapContent
from travel_planner.repositories import scraps as scrap_repository

logger = logging.getLogger(__name__)


class ScrapProcessingError(Exception):
    pass


async def process_external_scrap(
    session: AsyncSession, scrap_id: int, *, force: bool = False, client: JinaClient | None = None
) -> str:
    scrap = await scrap_repository.get_scrap_for_processing(session, scrap_id)
    if scrap is None:
        await session.rollback()
        raise ScrapProcessingError("Scrap not found")
    if not scrap.source_url:
        await session.rollback()
        raise ScrapProcessingError("Scrap has no external URL")

    existing = await scrap_repository.get_scrap_content(session, scrap_id)
    if existing and existing.content_text and existing.content_text.strip() and not force:
        if scrap.status not in {"ready", "processing"}:
            scrap.status = "ready"
            scrap.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        await session.commit()
        return existing.content_text
    if scrap.status == "processing" and not force:
        await session.rollback()
        raise ScrapProcessingError("Scrap is already processing")

    source_url = scrap.source_url
    scrap.status = "processing"
    scrap.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    await session.commit()

    try:
        content_text = await (client or JinaClient()).extract(source_url)
    except JinaError:
        await session.rollback()
        scrap = await scrap_repository.get_scrap_for_processing(session, scrap_id)
        if scrap is not None:
            scrap.status = "failed"
            scrap.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            await session.commit()
        logger.warning("scrap_extraction success=false scrap_id=%d", scrap_id)
        raise

    scrap = await scrap_repository.get_scrap_for_processing(session, scrap_id)
    if scrap is None:
        await session.rollback()
        raise ScrapProcessingError("Scrap was removed during extraction")
    existing = await scrap_repository.get_scrap_content(session, scrap_id)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if existing is None:
        session.add(ScrapContent(scrap_id=scrap_id, content_text=content_text, extraction_method="jina", fetched_at=now))
    else:
        existing.content_text = content_text
        existing.extraction_method = "jina"
        existing.fetched_at = now
        existing.updated_at = now
    scrap.status = "ready"
    scrap.updated_at = now
    await session.commit()
    logger.info("scrap_extraction success=true scrap_id=%d content_length=%d", scrap_id, len(content_text))
    return content_text
