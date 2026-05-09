"""Ingest request/response schemas (Sprint 2).

Differs from `schemas.returns.ExtractedReturns` by being persisted with an
ID so the user can land on the correction UI, edit cells, toggle scale,
then POST /confirm.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from qorba_api.schemas.returns import ReturnPoint, Scale


class TableCell(BaseModel):
    """One row in the parsed (un-scaled) raw table — for the correction grid."""

    period: date
    raw: str | None = None
    parsed: float | None = None


class IngestionDraft(BaseModel):
    """A parsed-but-unconfirmed series — what /ingest/* returns."""

    id: UUID
    name: str
    detected_scale: Scale
    tier_used: Literal[1, 2, 3, 4]
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_components: dict[str, float] = Field(default_factory=dict)
    points: list[ReturnPoint]  # in detected scale (unscaled)
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime


class IngestPasteRequest(BaseModel):
    name: str = "fund"
    text: str


class CellEdit(BaseModel):
    period: date
    value: float | None  # null = delete that month


class IngestionConfirm(BaseModel):
    """User-confirmed correction to apply on /confirm.

    `scale` is what the *user* says the values are. The server applies it
    (decimal stays as-is, percent divides by 100) and persists a Fund.
    """

    scale: Scale
    name: str | None = None
    edits: list[CellEdit] = Field(default_factory=list)
