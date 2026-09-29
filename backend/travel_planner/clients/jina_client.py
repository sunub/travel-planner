import ipaddress
import logging
from time import perf_counter
from urllib.parse import urlsplit

import httpx

from travel_planner.core.config import get_settings

logger = logging.getLogger(__name__)
JINA_TIMEOUT_SECONDS = 30.0
JINA_READER_ENDPOINT = "https://r.jina.ai/"


class JinaError(Exception):
    pass


def validate_source_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        if parsed.scheme not in {"http", "https"} or not host or parsed.username or parsed.password:
            raise ValueError
        if host.lower() == "localhost" or host.lower().endswith(".localhost"):
            raise ValueError
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if not any(character.isalpha() for character in host):
                raise ValueError("Invalid host") from None
        else:
            if not address.is_global:
                raise ValueError("Non-public address")
    except (ValueError, UnicodeError) as exc:
        raise JinaError("Invalid external URL") from exc


class JinaClient:
    """Read one external URL through Jina Reader; search lives in JinaSearchClient."""

    async def extract(self, url: str) -> str:
        validate_source_url(url)
        api_key = get_settings().jina_api_key
        if not api_key:
            raise JinaError("JINA_API_KEY is required for external URL extraction")

        started = perf_counter()
        try:
            async with httpx.AsyncClient(timeout=JINA_TIMEOUT_SECONDS) as client:
                response = await client.get(
                    f"{JINA_READER_ENDPOINT}{url}",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Accept": "application/json",
                        "X-Retain-Images": "none",
                    },
                )
            if response.is_error:
                raise JinaError(f"Jina Reader returned HTTP {response.status_code}")
            payload = response.json()
            if not isinstance(payload, dict):
                raise JinaError("Jina Reader returned an invalid response")
            data = payload.get("data", payload)
            content = data.get("content") if isinstance(data, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise JinaError("Jina Reader returned empty content")
            content = content.strip()
            logger.info("jina_request success=true duration_ms=%.1f content_length=%d", (perf_counter() - started) * 1000, len(content))
            return content
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            logger.warning("jina_request success=false duration_ms=%.1f reason=network", (perf_counter() - started) * 1000)
            raise JinaError("Jina Reader request timed out or failed") from exc
        except ValueError as exc:
            logger.warning("jina_request success=false duration_ms=%.1f reason=response", (perf_counter() - started) * 1000)
            raise JinaError("Jina Reader returned an invalid response") from exc
        except JinaError:
            logger.warning("jina_request success=false duration_ms=%.1f reason=http_or_empty", (perf_counter() - started) * 1000)
            raise
