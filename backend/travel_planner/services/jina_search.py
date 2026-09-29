from travel_planner.clients.jina_search_client import JinaSearchClient, JinaSearchResult


async def search_place_reviews(
    place_name: str, *, limit: int = 5, client: JinaSearchClient | None = None
) -> list[JinaSearchResult]:
    place_name = place_name.strip()
    if not place_name:
        raise ValueError("Place name must not be empty")
    return await (client or JinaSearchClient()).search(f"{place_name} 후기 리뷰", limit=limit)
