import logging
from collections.abc import AsyncIterator
from time import perf_counter
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from travel_planner.clients.jina_client import JinaError
from travel_planner.db.session import get_sessionmaker
from travel_planner.schemas.recommendations import RecommendationRequest, RecommendationResponse
from travel_planner.services.recommendations import RecommendationError, recommend as recommend_service
from travel_planner.services.scrap_processing import ScrapProcessingError

router = APIRouter(prefix="/recommendations", tags=["recommendations"])
logger = logging.getLogger(__name__)

async def get_recommendation_session() -> AsyncIterator[AsyncSession]:
    try:
        sessionmaker = get_sessionmaker()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Database is not configured") from exc
    async with sessionmaker() as session:
        yield session


Session = Annotated[AsyncSession, Depends(get_recommendation_session)]


@router.post("", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest, session: Session) -> RecommendationResponse:
    started = perf_counter()
    try:
        return await recommend_service(session, request)
    except RecommendationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except ScrapProcessingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except JinaError as exc:
        status_code = 503 if "JINA_API_KEY" in str(exc) else 422 if str(exc) == "Invalid external URL" else 502
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Required service configuration is missing") from exc
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="Ollama request failed") from exc
    except SQLAlchemyError as exc:
        await session.rollback()
        logger.error("recommendation_database_error")
        raise HTTPException(status_code=503, detail="Database operation failed") from exc
    finally:
        logger.info("recommendation_request duration_ms=%.1f", (perf_counter() - started) * 1000)
