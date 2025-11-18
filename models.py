from pydantic import BaseModel
from typing import List, Optional

# Recommendation Models
class RecommendRequest(BaseModel):
    goal: str
    days_per_week: int
    level: str
    top_n: int = 5

class RecommendItem(BaseModel):
    title: str
    goal: str
    level_list: List[str]
    days_per_week: Optional[int]
    cosine_similarity: float
    description: str

class RecommendResponse(BaseModel):
    items: List[RecommendItem]

# Progress Models
class SessionItem(BaseModel):
    date: str
    effort: str

class IngestRequest(BaseModel):
    user_id: str
    planned_per_week: int
    sessions: List[SessionItem]

class IngestResponse(BaseModel):
    weeks_added: int

class ComputeProgressRequest(BaseModel):
    user_id: str
    decision_period_weeks: int = 4

class ComputeProgressResponse(BaseModel):
    final_level: str
    last_progress: float
    last_readiness: float
    weeks_observed: int

class TrajectoryRequest(BaseModel):
    user_id: str
    last_n_weeks: int = 12

class TrajectoryRow(BaseModel):
    week: str
    days_trained: int
    LevelScore: float
    Readiness: float
    cum_readiness_mean: float
    progress_bar: float
    is_decision_week: bool
    level_final: str
    level_progressive: str

class TrajectoryResponse(BaseModel):
    weeks: List[TrajectoryRow]
