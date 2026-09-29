`backend_jina` 브랜치에 Jina 본문 추출과 실제 추천 흐름을 구현했습니다. **커밋·푸시는 하지 않았고 DB 스키마도 변경하지 않았습니다.**

💡 Jina Reader는 URL을 읽기 쉬운 본문으로 바꾸는 API입니다. 최종 후보 선택은 기존 Ollama 모델이 담당합니다.

1. **생성 파일:** [Jina Client](C:/Users/Admin/Desktop/travel_project/travel-planner-backend/backend/travel_planner/clients/jina_client.py), [스크랩 처리 Service](C:/Users/Admin/Desktop/travel_project/travel-planner-backend/backend/travel_planner/services/scrap_processing.py), [추천 Service](C:/Users/Admin/Desktop/travel_project/travel-planner-backend/backend/travel_planner/services/recommendations.py), [추천 Prompt](C:/Users/Admin/Desktop/travel_project/travel-planner-backend/backend/travel_planner/services/recommendation_prompt.py), [추천 Repository](C:/Users/Admin/Desktop/travel_project/travel-planner-backend/backend/travel_planner/repositories/recommendations.py), [운영 문서](C:/Users/Admin/Desktop/travel_project/travel-planner-backend/docs/backend-jina.md), 테스트 파일 2개를 추가했습니다.
2. **수정 파일:** [설정](C:/Users/Admin/Desktop/travel_project/travel-planner-backend/backend/travel_planner/core/config.py), [스크랩 Repository](C:/Users/Admin/Desktop/travel_project/travel-planner-backend/backend/travel_planner/repositories/scraps.py), [스크랩 Service](C:/Users/Admin/Desktop/travel_project/travel-planner-backend/backend/travel_planner/services/scraps.py), [추천 Route](C:/Users/Admin/Desktop/travel_project/travel-planner-backend/backend/travel_planner/api/routes/recommendations.py), [추천 Schema](C:/Users/Admin/Desktop/travel_project/travel-planner-backend/backend/travel_planner/schemas/recommendations.py), [앱 로깅](C:/Users/Admin/Desktop/travel_project/travel-planner-backend/backend/travel_planner/main.py), [.env.example](C:/Users/Admin/Desktop/travel_project/travel-planner-backend/.env.example)을 수정했습니다.
3. **추가 의존성:** 없습니다. 검증용 `uv`만 Git에서 제외되는 `.venv`에 설치했습니다.
4. **최종 구조:** 기존 `api/`, `clients/`, `repositories/`, `schemas/`, `services/` 구성을 유지하고 위 파일 및 `tests/`, `docs/backend-jina.md`를 추가했습니다.
5. **연동 방식:** `httpx.AsyncClient`로 Jina를 호출하고, DB 저장은 Service에서 처리합니다.
6. **공식 API:** [Jina Reader](https://jina.ai/reader/)의 `GET https://r.jina.ai/{source_url}`을 사용합니다. Bearer 인증과 JSON 응답 형식은 [공식 Reader 문서](https://github.com/jina-ai/reader)를 확인했습니다.
7. **환경변수:** `DATABASE_URL`, `OLLAMA_MODEL`, `JINA_API_KEY`가 해당 기능에 필요합니다. `OLLAMA_BASE_URL`은 선택 사항입니다. `.env`는 만들거나 수정하지 않았습니다.
8. **저장 흐름:** 외부 스크랩은 `pending → processing → ready`로 바뀌며, 본문을 `scrap_contents.content_text`에 Markdown으로 저장하고 `extraction_method="jina"`와 수집 시각을 기록합니다.
9. **추천 흐름:** 요청한 스크랩 조회 → 후보 자료 조합 → Ollama 비교 → 선택 ID·추천문·이유·근거·비교 ID 반환 순서입니다.
10. **내부 Place/Review:** 장소 정보와 리뷰, 주석의 근거 문장, 동행 정보를 사용합니다. 장소당 리뷰는 ID 순서의 최대 5개이며 합성 리뷰 여부를 표시합니다.
11. **외부 URL:** 저장된 본문이 있으면 재사용하고, 없으면 Jina 추출을 시도합니다. 모델에 넣는 후보 본문만 길이를 제한하며 DB의 전체 추출 본문은 보존합니다.
12. **Jina 실패:** 스크랩을 `failed`로 기록합니다. 잘못된 URL은 422, 키 누락은 503, 호출·응답 실패는 502로 처리합니다. 후보 하나가 실패해도 비교 대상이 몰래 바뀌지 않도록 요청 전체를 실패시킵니다.
13. **Ollama 실패:** 모델 설정 누락은 503, HTTP 호출 또는 응답 오류는 502입니다. 서버 시작과 `/health`는 Ollama 없이 가능합니다.
14. **Timeout:** Jina와 Ollama 각각 30초입니다.
15. **Retry:** 구현하지 않았습니다.
16. **실행:** 환경변수 설정 후 `uv run uvicorn travel_planner.main:app --reload`; 상태 확인은 `GET /api/v1/health`입니다.
17. **테스트 명령:** `uv run python -m unittest discover -s tests -v`입니다. 이번 환경에서는 `torch` 설치를 피하려고 백엔드 패키지만 지정한 `uv run --no-project` 임시 환경에서 실행했습니다.
18. **테스트 결과:** 11개 모두 통과했습니다. 본문 재사용·저장·실패 상태, Jina 오류, 근거 검증, `/health`, 추천 Route 응답을 확인했습니다. `git diff --check`도 통과했습니다.
19. **실제 Jina smoke test:** `JINA_API_KEY`가 없어 수행하지 못했습니다. 실제 DB 세션과 Ollama 연결도 접속 설정이 없어 검증하지 못했습니다.
20. **측정 로그:** Jina 성공·실패, 호출 시간, 본문 길이, Ollama 호출 시간, 추천 요청 전체 시간을 기록합니다. 키와 전체 본문은 로그에 남기지 않습니다.
21. **미구현:** 인증, HTTP 스크랩 생성 Route, 작업 큐, 학습·어댑터 적용은 이번 범위에 포함하지 않았습니다. 기존 스크랩 HTTP Route는 인증 부재로 계속 501입니다.
22. **Non-Jina 비교 조건:** API 경로·요청과 응답 형식·Prompt·`OLLAMA_MODEL`·DB 데이터·요구사항·스크랩 ID를 동일하게 맞춰야 합니다. 현재 Non-Jina 브랜치는 추천 Route가 501이므로 비교 전에 같은 추천 흐름을 적용해야 합니다.
23. **기술적 주의사항:** 현재 추천 Route에는 사용자 인증과 스크랩 소유권 검사가 없습니다. 신뢰된 실험 환경에서만 사용해야 합니다. 실제 DB·Jina·Ollama를 연결한 통합 검증은 남아 있습니다.

💡 근거 검증은 추천 응답의 근거 문장이 실제로 제공된 리뷰 또는 Jina 본문에 있는지 확인하는 절차입니다.
