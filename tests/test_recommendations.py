import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from travel_planner.api.routes.recommendations import get_recommendation_session
from travel_planner.main import app
from travel_planner.schemas.recommendations import RecommendationRequest
from travel_planner.services.recommendations import RecommendationError, recommend


class RecommendationServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_scrap_fails_before_ollama(self) -> None:
        ollama = SimpleNamespace(generate=AsyncMock())
        with (
            patch("travel_planner.services.recommendations.get_settings", return_value=SimpleNamespace(ollama_model="test-model")),
            patch("travel_planner.services.recommendations.scrap_repository.list_scraps", new=AsyncMock(return_value=[])),
        ):
            with self.assertRaises(RecommendationError) as caught:
                await recommend(SimpleNamespace(), RecommendationRequest(scrap_ids=[999], requirements="조용한 곳"), ollama)
        self.assertEqual(caught.exception.status_code, 404)
        ollama.generate.assert_not_awaited()

    async def test_recommendation_requires_verbatim_source_evidence(self) -> None:
        scrap = SimpleNamespace(scrap_id=1)
        candidate = {"scrap_id": 1, "kind": "review", "title": "리뷰", "material": "실제 리뷰 문장"}
        ollama = SimpleNamespace(generate=AsyncMock(return_value='{"selected_scrap_id":1,"recommendation":"추천","reasoning":"근거","evidence":[{"scrap_id":1,"text":"없는 문장"}]}'))
        with (
            patch("travel_planner.services.recommendations.get_settings", return_value=SimpleNamespace(ollama_model="test-model")),
            patch("travel_planner.services.recommendations.scrap_repository.list_scraps", new=AsyncMock(return_value=[scrap])),
            patch("travel_planner.services.recommendations._candidate", new=AsyncMock(return_value=(candidate, ["실제 리뷰 문장"]))),
        ):
            with self.assertRaisesRegex(RecommendationError, "evidence absent"):
                await recommend(SimpleNamespace(), RecommendationRequest(scrap_ids=[1], requirements="조용한 곳"), ollama)


class RecommendationRouteTests(unittest.TestCase):
    def test_app_health_and_recommendation_route(self) -> None:
        async def fake_session():
            yield SimpleNamespace()

        response = {
            "selected_scrap_id": 1,
            "recommendation": "첫 번째 후보",
            "reasoning": "리뷰 근거",
            "evidence": [{"scrap_id": 1, "text": "실제 리뷰 문장"}],
            "compared_scrap_ids": [1],
        }
        app.dependency_overrides[get_recommendation_session] = fake_session
        try:
            with patch("travel_planner.api.routes.recommendations.recommend_service", new=AsyncMock(return_value=response)):
                with TestClient(app) as client:
                    self.assertEqual(client.get("/api/v1/health").json(), {"status": "ok"})
                    result = client.post("/api/v1/recommendations", json={"scrap_ids": [1], "requirements": "조용한 곳"})
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json(), response)
        finally:
            app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()
