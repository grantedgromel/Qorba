"""Analysis CRUD + compute."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorba_api.auth import get_current_user
from qorba_api.db.models import Analysis, Fund, User
from qorba_api.db.session import get_db
from qorba_api.schemas.analyses import (
    AnalysisCreate,
    AnalysisOut,
    AnalysisResult,
    MetricSelection,
)
from qorba_api.schemas.returns import ReturnSeries
from qorba_api.services.compute import compute_analysis

router = APIRouter(prefix="/analyses", tags=["analyses"])


def _to_out(a: Analysis) -> AnalysisOut:
    sel = a.selection or {}
    return AnalysisOut(
        id=a.id,
        fund_id=a.fund_id,
        metrics=MetricSelection(metric_ids=sel.get("metric_ids", ["sharpe"])),
        rf_annual=float(sel.get("rf_annual", 0.0)),
        mar_annual=float(sel.get("mar_annual", 0.0)),
        omega_threshold=float(sel.get("omega_threshold", 0.0)),
        created_at=a.created_at,
    )


@router.get("", response_model=list[AnalysisOut])
def list_analyses(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[AnalysisOut]:
    rows = db.scalars(
        select(Analysis)
        .where(Analysis.user_id == user.id)
        .order_by(Analysis.created_at.desc())
    ).all()
    return [_to_out(r) for r in rows]


@router.post("", response_model=AnalysisOut, status_code=status.HTTP_201_CREATED)
def create_analysis(
    body: AnalysisCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> AnalysisOut:
    fund = db.scalar(select(Fund).where(Fund.id == body.fund_id, Fund.user_id == user.id))
    if not fund:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Fund not found")
    analysis = Analysis(
        user_id=user.id,
        fund_id=fund.id,
        selection={
            "metric_ids": body.metrics.metric_ids,
            "rf_annual": body.rf_annual,
            "mar_annual": body.mar_annual,
            "omega_threshold": body.omega_threshold,
        },
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return _to_out(analysis)


@router.get("/{analysis_id}", response_model=AnalysisOut)
def get_analysis(
    analysis_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> AnalysisOut:
    a = db.scalar(
        select(Analysis).where(Analysis.id == analysis_id, Analysis.user_id == user.id)
    )
    if not a:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Analysis not found")
    return _to_out(a)


@router.post("/{analysis_id}/compute", response_model=AnalysisResult)
def compute(
    analysis_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> AnalysisResult:
    a = db.scalar(
        select(Analysis).where(Analysis.id == analysis_id, Analysis.user_id == user.id)
    )
    if not a:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Analysis not found")
    fund = db.get(Fund, a.fund_id)
    if not fund:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Fund not found")

    series = ReturnSeries.model_validate(fund.series)
    sel = a.selection or {}
    return compute_analysis(
        analysis_id=a.id,
        fund_series=series,
        metric_ids=sel.get("metric_ids", ["sharpe"]),
        rf_annual=float(sel.get("rf_annual", 0.0)),
        mar_annual=float(sel.get("mar_annual", 0.0)),
    )
