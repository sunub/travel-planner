from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from travel_planner.models.scrap import Scrap, ScrapContent
from travel_planner.schemas.scraps import ScrapCreate


async def create_scrap(session: AsyncSession, *, user_id: int, request: ScrapCreate) -> Scrap:
    scrap = Scrap(user_id=user_id, status="pending" if request.source_url else "ready", **request.model_dump())
    session.add(scrap)
    await session.flush()
    await session.refresh(scrap)
    return scrap


async def get_scrap(session: AsyncSession, *, scrap_id: int, user_id: int) -> Scrap | None:
    return await session.scalar(
        select(Scrap).where(Scrap.scrap_id == scrap_id, Scrap.user_id == user_id)
    )


async def delete_scrap(session: AsyncSession, scrap: Scrap) -> None:
    await session.delete(scrap)


async def get_scrap_for_processing(session: AsyncSession, scrap_id: int) -> Scrap | None:
    return await session.scalar(select(Scrap).where(Scrap.scrap_id == scrap_id).with_for_update())


async def get_scrap_content(session: AsyncSession, scrap_id: int) -> ScrapContent | None:
    return await session.get(ScrapContent, scrap_id)


async def list_scraps(session: AsyncSession, scrap_ids: list[int]) -> list[Scrap]:
    result = await session.scalars(select(Scrap).where(Scrap.scrap_id.in_(scrap_ids)))
    return list(result)
