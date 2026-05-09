"""Fund persistence. Sprint 1: create from a confirmed ExtractedReturns; list/get."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorba_api.auth import get_current_user
from qorba_api.db.models import Fund, User
from qorba_api.db.session import get_db
from qorba_api.schemas.analyses import FundOut
from qorba_api.schemas.returns import ExtractedReturns

router = APIRouter(prefix="/funds", tags=["funds"])


@router.post("", response_model=FundOut, status_code=status.HTTP_201_CREATED)
def create_fund(
    body: ExtractedReturns,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> FundOut:
    fund = Fund(
        user_id=user.id,
        name=body.series.name,
        series=body.series.model_dump(mode="json"),
    )
    db.add(fund)
    db.commit()
    db.refresh(fund)
    return FundOut(
        id=fund.id,
        name=fund.name,
        inception=body.series.inception,
        last_observation=body.series.last_observation,
        n_observations=body.series.n_observations,
    )


@router.get("/{fund_id}", response_model=FundOut)
def get_fund(
    fund_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> FundOut:
    fund = db.scalar(select(Fund).where(Fund.id == fund_id, Fund.user_id == user.id))
    if not fund:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Fund not found")
    s = fund.series
    return FundOut(
        id=fund.id,
        name=fund.name,
        inception=s["inception"],
        last_observation=s["last_observation"],
        n_observations=s["n_observations"],
    )
