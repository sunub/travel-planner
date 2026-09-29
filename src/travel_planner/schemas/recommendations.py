from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    scrap_ids: list[int] = Field(min_length=1, max_length=10)
    requirements: str = Field(min_length=1, max_length=2000)


class RecommendationEvidence(BaseModel):
    scrap_id: int
    text: str


class RecommendationResponse(BaseModel):
    selected_scrap_id: int
    recommendation: str
    reasoning: str
    evidence: list[RecommendationEvidence]
    compared_scrap_ids: list[int]


class RecommendationDecision(BaseModel):
    selected_scrap_id: int
    recommendation: str
    reasoning: str
    evidence: list[RecommendationEvidence]
