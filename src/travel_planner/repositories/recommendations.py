from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from travel_planner.models.annotation import ReviewAnnotation
from travel_planner.models.aspect import Aspect
from travel_planner.models.companion import CompanionType, ReviewCompanion
from travel_planner.models.place import Place
from travel_planner.models.review import Review


async def get_place(session: AsyncSession, place_id: int) -> Place | None:
    return await session.get(Place, place_id)


async def get_review(session: AsyncSession, review_id: int) -> Review | None:
    return await session.get(Review, review_id)


async def get_place_reviews(session: AsyncSession, place_id: int) -> list[Review]:
    result = await session.scalars(
        select(Review).where(Review.place_id == place_id).order_by(Review.review_id).limit(5)
    )
    return list(result)


async def get_review_annotations(session: AsyncSession, review_id: int) -> list[tuple[ReviewAnnotation, str]]:
    result = await session.execute(
        select(ReviewAnnotation, Aspect.aspect_name)
        .join(Aspect, ReviewAnnotation.aspect_id == Aspect.aspect_id)
        .where(ReviewAnnotation.review_id == review_id)
        .order_by(ReviewAnnotation.annotation_id)
    )
    return list(result.tuples())


async def get_review_companions(session: AsyncSession, review_id: int) -> list[str]:
    result = await session.scalars(
        select(CompanionType.companion_code)
        .join(ReviewCompanion, CompanionType.companion_type_id == ReviewCompanion.companion_type_id)
        .where(ReviewCompanion.review_id == review_id)
    )
    return list(result)
