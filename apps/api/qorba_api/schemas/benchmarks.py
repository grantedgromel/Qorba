"""Benchmark library + ingest schemas."""

from __future__ import annotations

from datetime import date
from uuid import UUID

from pydantic import BaseModel

from qorba_api.schemas.returns import ReturnSeries


class BenchmarkLibraryItem(BaseModel):
    code: str
    name: str
    category: str
    provider: str  # "caissa" | "static" | "user"


class BenchmarkLibrary(BaseModel):
    items: list[BenchmarkLibraryItem]
    provider_active: str  # which provider is currently serving


class BenchmarkOut(BaseModel):
    id: UUID
    name: str
    code: str | None
    provider: str
    inception: date
    last_observation: date
    n_observations: int


class BenchmarkUploadFromIngest(BaseModel):
    """Promote a confirmed IngestionResult into a Benchmark."""

    name: str
    code: str | None = None
    series: ReturnSeries
