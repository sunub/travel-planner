# Jina backend experiment

This branch keeps the `feature/backend-base` FastAPI, SQLAlchemy, and Ollama layout.
Application code lives in `backend/travel_planner/`; run commands from the repository root.
It reads external URLs with the [Jina Reader API](https://jina.ai/reader/) using
`GET https://r.jina.ai/{source_url}`, `Authorization: Bearer JINA_API_KEY`, and
`Accept: application/json`, and `X-Retain-Images: none`. The `data.content` Markdown is stored in
`scrap_contents.content_text` without further trimming beyond surrounding
whitespace. `extraction_method` is `jina`; `fetched_at` is UTC recorded in the
existing timestamp column. No schema changes are needed.

Set `DATABASE_URL`, `OLLAMA_MODEL`, and `JINA_API_KEY` in a local `.env` or the
process environment. `OLLAMA_BASE_URL` defaults to `http://localhost:11434`.
Run `uv run uvicorn travel_planner.main:app --reload`, then check
`GET /api/v1/health`. Run `uv run python -m unittest discover -s tests -v`
for unit tests. The server can start without Jina or Ollama; requests
requiring an unavailable service fail at request time.

An external scrap created through the internal scrap service starts as
`pending`, changes to `processing` before the HTTP call, then becomes `ready`
after its content and metadata are committed. Jina failures set `failed`.
Previously saved nonempty content is reused. The recommendation service also
extracts selected external scraps on demand when their content is missing.
The existing HTTP Scrap routes still return 501 because authentication is not
implemented; they cannot currently create scraps through HTTP.

`POST /api/v1/recommendations` accepts the existing `scrap_ids` and
`requirements` fields. It reads internal place and review data from the current
database, including up to five reviews per place ordered by `review_id`, their
annotations, and companion labels. Synthetic reviews are explicitly marked.
Requests accept at most ten scraps and 2,000 requirement characters. Candidate
material uses the first `min(6000, 12000 / candidate_count)` characters, so
the combined candidate material stays within 12,000 characters. The full Jina
extraction remains in `scrap_contents`. Ollama selects
one of the requested scraps and returns a recommendation, reasoning, and
verbatim evidence from the supplied candidate material. A missing or failed
candidate fails the entire request, so the compared set never silently changes.
The endpoint currently has no user authentication or ownership check; deploy it
only in a trusted experiment environment until authentication is designed.

Jina and Ollama HTTP timeouts are each 30 seconds. There is no retry or worker
queue. Logs include Jina success/failure, duration and content length, Ollama
duration, and total recommendation duration. They omit credentials and full
page text. For a fair Non-Jina comparison, keep the recommendation path,
request/response schema, prompt, `OLLAMA_MODEL`, user requirements, scrap IDs,
and DB data identical; vary only external URL content acquisition.

## Optional Jina Search

`JinaSearchClient` uses `GET https://s.jina.ai/` with `q`, `num`,
`Authorization: Bearer JINA_API_KEY`, and `Accept: application/json`. The
`search_place_reviews(place_name, limit=5)` service builds a review-related
query and returns up to three to five ranked `JinaSearchResult` values. Each
value has `source_url`, `title`, and `content_text` for a future review analysis
input. Search is never called by the existing recommendation route, and its
results are not saved or added to Review Profiles automatically. The same
`JINA_API_KEY` setting is read from `.env`, which is ignored by Git.

```python
from travel_planner.services.jina_search import search_place_reviews

results = await search_place_reviews("해운대", limit=3)
```
