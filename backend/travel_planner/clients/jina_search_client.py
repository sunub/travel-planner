import logging
from dataclasses import dataclass
from time import perf_counter
from urllib.parse import urlsplit

import httpx

from travel_planner.core.config import get_settings

logger = logging.getLogger(__name__)
JINA_SEARCH_ENDPOINT = "https://s.jina.ai/"
JINA_SEARCH_TIMEOUT_SECONDS = 30.0


class JinaSearchError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class JinaSearchResult:
    rank: int
    source_url: str
    title: str
    content_text: str


class JinaSearchClient:
    async def search(self, query: str, *, limit: int = 5) -> list[JinaSearchResult]:
        query = query.strip()
        if not query:
            raise ValueError("Search query must not be empty")
        if not 3 <= limit <= 5:
            raise ValueError("Search limit must be between 3 and 5")
        api_key = get_settings().jina_api_key
        if not api_key:
            raise JinaSearchError("JINA_API_KEY is required for Jina Search")

        started = perf_counter()
        try:
            async with httpx.AsyncClient(timeout=JINA_SEARCH_TIMEOUT_SECONDS) as client:
                response = await client.get(
                    JINA_SEARCH_ENDPOINT,
                    params={"q": query, "num": limit},
                    headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
                )
            if response.is_error:
                raise JinaSearchError(f"Jina Search returned HTTP {response.status_code}")
            payload = response.json()
            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, list):
                raise JinaSearchError("Jina Search returned an invalid response")

            results = []
            for rank, item in enumerate(data[:limit], start=1):
                if not isinstance(item, dict) or not isinstance(item.get("url"), str):
                    continue
                url = item["url"]
                try:
                    parsed = urlsplit(url)
                except ValueError:
                    continue
                if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                    continue
                title = item.get("title")
                content = item.get("content")
                results.append(
                    JinaSearchResult(
                        rank=rank,
                        source_url=url,
                        title=title if isinstance(title, str) else "",
                        content_text=content if isinstance(content, str) else "",
                    )
                )
            logger.info("jina_search success=true duration_ms=%.1f result_count=%d", (perf_counter() - started) * 1000, len(results))
            return results
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            logger.warning("jina_search success=false duration_ms=%.1f reason=network", (perf_counter() - started) * 1000)
            raise JinaSearchError("Jina Search request timed out or failed") from exc
        except ValueError as exc:
            logger.warning("jina_search success=false duration_ms=%.1f reason=response", (perf_counter() - started) * 1000)
            raise JinaSearchError("Jina Search returned an invalid response") from exc
        except JinaSearchError:
            logger.warning("jina_search success=false duration_ms=%.1f reason=http_or_response", (perf_counter() - started) * 1000)
            raise
