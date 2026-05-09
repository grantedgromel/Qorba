"""Analysis create/result schemas. Sprint 1 covers the minimum-viable shape:
one fund, no benchmark, no peer group, returns one named metric."""

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, Field


class FundOut(BaseModel):
    id: UUID
    name: str
    inception: date
    last_observation: date
    n_observations: int


class MetricSelection(BaseModel):
    metric_ids: list[str] = Field(default_factory=lambda: ["sharpe"])


class AnalysisCreate(BaseModel):
    fund_id: UUID
    metrics: MetricSelection = Field(default_factory=MetricSelection)
    rf_annual: float = 0.0
    mar_annual: float = 0.0
    omega_threshold: float = 0.0


class AnalysisOut(BaseModel):
    id: UUID
    fund_id: UUID
    metrics: MetricSelection
    rf_annual: float
    mar_annual: float
    omega_threshold: float
    created_at: datetime


class MetricValue(BaseModel):
    metric_id: str
    value: float | None
    formatted: str


class AnalysisResult(BaseModel):
    analysis_id: UUID
    metrics: dict[str, MetricValue]
    computed_at: datetime
    version_hash: str
