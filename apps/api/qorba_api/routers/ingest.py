"""Ingestion endpoints. Sprint 1: CSV only."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from qorba_api.auth import get_current_user
from qorba_api.core.ingestion.loader import apply_scale, load_returns
from qorba_api.db.models import User
from qorba_api.schemas.returns import ExtractedReturns
from qorba_api.services.series import series_to_dto

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/csv", response_model=ExtractedReturns)
async def ingest_csv(
    file: UploadFile = File(...),
    _user: User = Depends(get_current_user),
) -> ExtractedReturns:
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Missing filename")
    raw = await file.read()
    df, scale = load_returns(file.filename, raw)
    if df.empty:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Empty or unparseable CSV")
    numeric = df.select_dtypes("number")
    if numeric.empty:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "No numeric return column found")

    df_decimal = apply_scale(df, scale)
    s = df_decimal[numeric.columns[0]].dropna()
    s.name = numeric.columns[0]

    series = series_to_dto(s, name=str(numeric.columns[0]), source="csv")
    warnings: list[str] = []
    if len(s) < 12:
        warnings.append("Less than 12 months of data — some metrics may be unreliable.")

    return ExtractedReturns(
        series=series,
        confidence=1.0,
        tier_used=1,
        detected_scale=scale,
        warnings=warnings,
        raw_table=[],
    )
