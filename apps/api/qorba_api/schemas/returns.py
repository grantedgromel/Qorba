"""Return-series schemas shared by ingest and analysis."""

from datetime import date
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

Scale = Literal["percent", "decimal"]


class ReturnPoint(BaseModel):
    period: date
    value: float


class ReturnSeries(BaseModel):
    id: UUID
    name: str
    points: list[ReturnPoint]
    inception: date
    last_observation: date
    n_observations: int = Field(ge=0)
    source: Literal[
        "csv", "xlsx", "pdf_tier1", "pdf_tier2", "pdf_tier3", "paste", "library"
    ]
    checksum: str


class ExtractedReturns(BaseModel):
    """Result of an ingest, before user confirmation in the correction UI."""

    series: ReturnSeries
    confidence: float = Field(ge=0.0, le=1.0)
    tier_used: Literal[1, 2, 3, 4]
    detected_scale: Scale
    warnings: list[str] = []
    raw_table: list[list[str | None]] = []
