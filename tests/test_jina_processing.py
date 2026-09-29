import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx

from travel_planner.clients.jina_client import JinaClient, JinaError
from travel_planner.services.scrap_processing import process_external_scrap


class JinaClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_extract_reads_json_content_without_exposing_key(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            self.assertEqual(str(request.url), "https://r.jina.ai/https://example.com/review")
            self.assertEqual(request.headers["Authorization"], "Bearer secret-for-test")
            return httpx.Response(200, json={"data": {"content": "  실제 리뷰 문장  "}})

        original_client = httpx.AsyncClient
        transport = httpx.MockTransport(respond)
        with (
            patch("travel_planner.clients.jina_client.get_settings", return_value=SimpleNamespace(jina_api_key="secret-for-test")),
            patch("travel_planner.clients.jina_client.httpx.AsyncClient", side_effect=lambda **kwargs: original_client(transport=transport, **kwargs)),
        ):
            self.assertEqual(await JinaClient().extract("https://example.com/review"), "실제 리뷰 문장")

    async def test_http_failure_is_sanitized(self) -> None:
        original_client = httpx.AsyncClient
        transport = httpx.MockTransport(lambda request: httpx.Response(503, text="secret-for-test"))
        with (
            patch("travel_planner.clients.jina_client.get_settings", return_value=SimpleNamespace(jina_api_key="secret-for-test")),
            patch("travel_planner.clients.jina_client.httpx.AsyncClient", side_effect=lambda **kwargs: original_client(transport=transport, **kwargs)),
        ):
            with self.assertRaisesRegex(JinaError, "HTTP 503") as caught:
                await JinaClient().extract("https://example.com/review")
            self.assertNotIn("secret-for-test", str(caught.exception))

    async def test_missing_key_does_not_affect_client_initialization(self) -> None:
        with patch("travel_planner.clients.jina_client.get_settings", return_value=SimpleNamespace(jina_api_key=None)):
            client = JinaClient()
            with self.assertRaisesRegex(JinaError, "JINA_API_KEY"):
                await client.extract("https://example.com/review")

    async def test_invalid_url_and_empty_content_are_rejected(self) -> None:
        with self.assertRaisesRegex(JinaError, "Invalid external URL"):
            await JinaClient().extract("http://localhost/private")
        original_client = httpx.AsyncClient
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"data": {"content": "  "}}))
        with (
            patch("travel_planner.clients.jina_client.get_settings", return_value=SimpleNamespace(jina_api_key="secret-for-test")),
            patch("travel_planner.clients.jina_client.httpx.AsyncClient", side_effect=lambda **kwargs: original_client(transport=transport, **kwargs)),
        ):
            with self.assertRaisesRegex(JinaError, "empty content"):
                await JinaClient().extract("https://example.com/review")

    async def test_timeout_is_sanitized(self) -> None:
        def timeout(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("secret-for-test")

        original_client = httpx.AsyncClient
        transport = httpx.MockTransport(timeout)
        with (
            patch("travel_planner.clients.jina_client.get_settings", return_value=SimpleNamespace(jina_api_key="secret-for-test")),
            patch("travel_planner.clients.jina_client.httpx.AsyncClient", side_effect=lambda **kwargs: original_client(transport=transport, **kwargs)),
        ):
            with self.assertRaises(JinaError) as caught:
                await JinaClient().extract("https://example.com/review")
            self.assertNotIn("secret-for-test", str(caught.exception))


class ScrapProcessingTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_content_is_reused_without_jina_call(self) -> None:
        scrap = SimpleNamespace(scrap_id=7, source_url="https://example.com/review", status="ready", updated_at=None)
        content = SimpleNamespace(content_text="저장된 본문")
        session = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock())
        client = SimpleNamespace(extract=AsyncMock())
        with (
            patch("travel_planner.services.scrap_processing.scrap_repository.get_scrap_for_processing", new=AsyncMock(return_value=scrap)),
            patch("travel_planner.services.scrap_processing.scrap_repository.get_scrap_content", new=AsyncMock(return_value=content)),
        ):
            self.assertEqual(await process_external_scrap(session, 7, client=client), "저장된 본문")
        client.extract.assert_not_awaited()
        session.commit.assert_awaited_once()

    async def test_jina_failure_marks_scrap_failed(self) -> None:
        scrap = SimpleNamespace(scrap_id=7, source_url="https://example.com/review", status="pending", updated_at=None)
        session = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock())
        client = SimpleNamespace(extract=AsyncMock(side_effect=JinaError("Jina Reader returned HTTP 503")))
        with (
            patch("travel_planner.services.scrap_processing.scrap_repository.get_scrap_for_processing", new=AsyncMock(return_value=scrap)),
            patch("travel_planner.services.scrap_processing.scrap_repository.get_scrap_content", new=AsyncMock(return_value=None)),
        ):
            with self.assertRaises(JinaError):
                await process_external_scrap(session, 7, client=client)
        self.assertEqual(scrap.status, "failed")
        self.assertEqual(session.commit.await_count, 2)

    async def test_success_stores_content_and_ready_state(self) -> None:
        scrap = SimpleNamespace(scrap_id=7, source_url="https://example.com/review", status="pending", updated_at=None)
        session = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock(), add=Mock())
        client = SimpleNamespace(extract=AsyncMock(return_value="실제 본문"))
        with (
            patch("travel_planner.services.scrap_processing.scrap_repository.get_scrap_for_processing", new=AsyncMock(return_value=scrap)),
            patch("travel_planner.services.scrap_processing.scrap_repository.get_scrap_content", new=AsyncMock(return_value=None)),
        ):
            self.assertEqual(await process_external_scrap(session, 7, client=client), "실제 본문")
        stored = session.add.call_args.args[0]
        self.assertEqual((stored.scrap_id, stored.content_text, stored.extraction_method), (7, "실제 본문", "jina"))
        self.assertIsNotNone(stored.fetched_at)
        self.assertEqual(scrap.status, "ready")
        self.assertEqual(session.commit.await_count, 2)


if __name__ == "__main__":
    unittest.main()
