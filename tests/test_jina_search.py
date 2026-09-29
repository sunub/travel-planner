import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from travel_planner.clients.jina_search_client import JinaSearchClient, JinaSearchError
from travel_planner.services.jina_search import search_place_reviews


class JinaSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_returns_only_top_results_with_bearer_auth(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            self.assertEqual(str(request.url.copy_with(query=None)), "https://s.jina.ai/")
            self.assertEqual(request.url.params["q"], "부산 해운대 후기 리뷰")
            self.assertEqual(request.url.params["num"], "3")
            self.assertEqual(request.headers["Authorization"], "Bearer secret-for-test")
            return httpx.Response(200, json={"data": [
                {"url": f"https://example.com/{i}", "title": f"후기 {i}", "content": f"본문 {i}"}
                for i in range(1, 5)
            ]})

        original_client = httpx.AsyncClient
        transport = httpx.MockTransport(respond)
        with (
            patch("travel_planner.clients.jina_search_client.get_settings", return_value=SimpleNamespace(jina_api_key="secret-for-test")),
            patch("travel_planner.clients.jina_search_client.httpx.AsyncClient", side_effect=lambda **kwargs: original_client(transport=transport, **kwargs)),
        ):
            results = await search_place_reviews(" 부산 해운대 ", limit=3)
        self.assertEqual(len(results), 3)
        self.assertEqual((results[0].rank, results[0].source_url, results[0].content_text), (1, "https://example.com/1", "본문 1"))

    async def test_missing_key_is_checked_only_when_search_is_called(self) -> None:
        with patch("travel_planner.clients.jina_search_client.get_settings", return_value=SimpleNamespace(jina_api_key=None)):
            client = JinaSearchClient()
            with self.assertRaisesRegex(JinaSearchError, "JINA_API_KEY"):
                await client.search("부산 후기")

    async def test_http_error_does_not_expose_response_body(self) -> None:
        original_client = httpx.AsyncClient
        transport = httpx.MockTransport(lambda request: httpx.Response(402, text="secret-for-test"))
        with (
            patch("travel_planner.clients.jina_search_client.get_settings", return_value=SimpleNamespace(jina_api_key="secret-for-test")),
            patch("travel_planner.clients.jina_search_client.httpx.AsyncClient", side_effect=lambda **kwargs: original_client(transport=transport, **kwargs)),
        ):
            with self.assertRaisesRegex(JinaSearchError, "HTTP 402") as caught:
                await JinaSearchClient().search("부산 후기")
        self.assertNotIn("secret-for-test", str(caught.exception))

    async def test_service_builds_review_query_without_db_or_reader(self) -> None:
        client = SimpleNamespace(search=AsyncMock(return_value=[]))
        self.assertEqual(await search_place_reviews("광안리", client=client), [])
        client.search.assert_awaited_once_with("광안리 후기 리뷰", limit=5)


if __name__ == "__main__":
    unittest.main()
