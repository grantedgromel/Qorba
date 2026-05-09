"""Ingestion endpoints. Each ingest produces an IngestionDraft persisted on
the server; the user reviews on the correction UI and POSTs /confirm to
mint a Fund."""

from __future__ import annotations

import re
from datetime import date
from typing import Literal
from uuid import UUID

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorba_api.auth import get_current_user
from qorba_api.core.ingestion.cascade import run_pdf_cascade
from qorba_api.core.ingestion.loader import (
    Scale,
    detect_scale,
    load_returns,
)
from qorba_api.db.models import Fund, IngestionResult, User
from qorba_api.db.session import get_db
from qorba_api.schemas.analyses import FundOut
from qorba_api.schemas.ingest import (
    CellEdit,
    IngestionConfirm,
    IngestionDraft,
    IngestPasteRequest,
)
from qorba_api.schemas.returns import ReturnPoint
from qorba_api.services.series import series_to_dto

router = APIRouter(prefix="/ingest", tags=["ingest"])


# ── Helpers ───────────────────────────────────────────────────────────


def _series_to_points(series: pd.Series) -> list[ReturnPoint]:
    return [
        ReturnPoint(period=pd.Timestamp(idx).date(), value=float(val))
        for idx, val in series.items()
    ]


def _persist_draft(
    db: Session,
    user: User,
    *,
    name: str,
    points: list[ReturnPoint],
    detected_scale: Scale,
    tier_used: Literal[1, 2, 3, 4],
    confidence: float,
    components: dict[str, float],
    warnings: list[str],
) -> IngestionDraft:
    payload = {
        "name": name,
        "detected_scale": detected_scale,
        "tier_used": tier_used,
        "confidence": confidence,
        "confidence_components": components,
        "points": [
            {"period": p.period.isoformat(), "value": p.value} for p in points
        ],
        "warnings": warnings,
    }
    record = IngestionResult(user_id=user.id, payload=payload)
    db.add(record)
    db.commit()
    db.refresh(record)
    return IngestionDraft(
        id=record.id,
        name=name,
        detected_scale=detected_scale,
        tier_used=tier_used,
        confidence=confidence,
        confidence_components=components,
        points=points,
        warnings=warnings,
        created_at=record.created_at,
    )


def _load_draft_or_404(
    db: Session, draft_id: UUID, user: User
) -> tuple[IngestionResult, dict]:
    row = db.scalar(
        select(IngestionResult).where(
            IngestionResult.id == draft_id, IngestionResult.user_id == user.id
        )
    )
    if not row:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Ingestion draft not found")
    if row.confirmed_fund_id is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Draft already confirmed")
    return row, row.payload


def _apply_edits(points: list[ReturnPoint], edits: list[CellEdit]) -> list[ReturnPoint]:
    edit_map: dict[date, CellEdit] = {e.period: e for e in edits}
    out: list[ReturnPoint] = []
    for p in points:
        if p.period in edit_map:
            new_val = edit_map[p.period].value
            edit_map.pop(p.period)
            if new_val is None:
                continue
            out.append(ReturnPoint(period=p.period, value=new_val))
        else:
            out.append(p)
    # Edits with periods not in the original draft = adds.
    for period, edit in edit_map.items():
        if edit.value is not None:
            out.append(ReturnPoint(period=period, value=edit.value))
    out.sort(key=lambda p: p.period)
    return out


# ── Tabular ingest (CSV/XLSX) ─────────────────────────────────────────


def _ingest_tabular(
    filename: str, raw: bytes, db: Session, user: User
) -> IngestionDraft:
    df, scale = load_returns(filename, raw)
    if df.empty:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "Empty or unparseable file"
        )
    numeric = df.select_dtypes("number")
    if numeric.empty:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "No numeric return column found"
        )
    series = numeric.iloc[:, 0].dropna()
    series.name = numeric.columns[0]
    points = _series_to_points(series)
    return _persist_draft(
        db,
        user,
        name=str(numeric.columns[0]),
        points=points,
        detected_scale=scale,
        tier_used=1,
        confidence=1.0,
        components={"coverage": 1.0, "month_grid": 1.0, "value_sanity": 1.0},
        warnings=(
            ["Less than 12 months of data — some metrics may be unreliable."]
            if len(series) < 12
            else []
        ),
    )


@router.post("/csv", response_model=IngestionDraft)
async def ingest_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> IngestionDraft:
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Missing filename")
    raw = await file.read()
    return _ingest_tabular(file.filename, raw, db, user)


@router.post("/xlsx", response_model=IngestionDraft)
async def ingest_xlsx(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> IngestionDraft:
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Missing filename")
    raw = await file.read()
    return _ingest_tabular(file.filename, raw, db, user)


# ── PDF ingest ────────────────────────────────────────────────────────


@router.post("/pdf", response_model=IngestionDraft)
async def ingest_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> IngestionDraft:
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Missing filename")
    raw = await file.read()

    result = run_pdf_cascade(raw)
    if result.series.empty:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Could not extract a return series from the PDF. "
            "Try uploading a CSV/Excel or paste the table directly.",
        )

    # Detect scale on a DataFrame so we reuse the heuristic.
    df = result.series.to_frame("v")
    scale = detect_scale(df)

    points = _series_to_points(result.series)
    return _persist_draft(
        db,
        user,
        name=file.filename.rsplit(".", 1)[0] or "fund",
        points=points,
        detected_scale=scale,
        tier_used=result.tier_used,  # type: ignore[arg-type]
        confidence=result.confidence,
        components=result.components,
        warnings=result.warnings,
    )


# ── Paste ingest (Tier 4) ─────────────────────────────────────────────


_DELIM_RE = re.compile(r"[,\t]| {2,}")


def _tokenize_paste(text: str) -> pd.DataFrame:
    rows: list[list[str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append([c.strip() for c in _DELIM_RE.split(line) if c.strip()])
    if not rows:
        return pd.DataFrame()
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    df = pd.DataFrame(rows[1:], columns=rows[0])
    return df


@router.post("/paste", response_model=IngestionDraft)
async def ingest_paste(
    body: IngestPasteRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> IngestionDraft:
    df = _tokenize_paste(body.text)
    if df.empty or df.shape[1] < 2:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Could not interpret the pasted text as a date+value table.",
        )

    # Use the same loader path: round-trip through CSV bytes so the date
    # detection + scale heuristic both apply.
    csv_bytes = df.to_csv(index=False).encode()
    df_loaded, scale = load_returns("paste.csv", csv_bytes)
    if df_loaded.empty:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "Could not parse pasted dates."
        )
    numeric = df_loaded.select_dtypes("number")
    if numeric.empty:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "No numeric column in pasted data."
        )
    series = numeric.iloc[:, 0].dropna()
    series.name = body.name
    points = _series_to_points(series)
    return _persist_draft(
        db,
        user,
        name=body.name,
        points=points,
        detected_scale=scale,
        tier_used=4,
        confidence=1.0,  # the user typed it themselves
        components={"coverage": 1.0, "month_grid": 1.0, "value_sanity": 1.0},
        warnings=[],
    )


# ── Get + Confirm ─────────────────────────────────────────────────────


@router.get("/drafts/{draft_id}", response_model=IngestionDraft)
def get_draft(
    draft_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> IngestionDraft:
    record, payload = _load_draft_or_404(db, draft_id, user)
    return IngestionDraft(
        id=record.id,
        name=payload["name"],
        detected_scale=payload["detected_scale"],
        tier_used=payload["tier_used"],
        confidence=payload["confidence"],
        confidence_components=payload.get("confidence_components", {}),
        points=[
            ReturnPoint(period=date.fromisoformat(p["period"]), value=p["value"])
            for p in payload["points"]
        ],
        warnings=payload.get("warnings", []),
        created_at=record.created_at,
    )


@router.post("/drafts/{draft_id}/confirm", response_model=FundOut, status_code=status.HTTP_201_CREATED)
def confirm_draft(
    draft_id: UUID,
    body: IngestionConfirm,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> FundOut:
    record, payload = _load_draft_or_404(db, draft_id, user)

    points = [
        ReturnPoint(period=date.fromisoformat(p["period"]), value=p["value"])
        for p in payload["points"]
    ]
    if body.edits:
        points = _apply_edits(points, body.edits)
    if not points:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "No data points after edits"
        )

    # Apply the user-confirmed scale.
    scale_factor = 100.0 if body.scale == "percent" else 1.0
    scaled_points = [
        ReturnPoint(period=p.period, value=p.value / scale_factor) for p in points
    ]

    pseries = pd.Series(
        [p.value for p in scaled_points],
        index=pd.DatetimeIndex([p.period for p in scaled_points]),
        name=body.name or payload["name"],
    )

    series = series_to_dto(
        pseries,
        name=body.name or payload["name"],
        source=_source_for_tier(payload["tier_used"]),
    )

    fund = Fund(
        user_id=user.id,
        name=series.name,
        series=series.model_dump(mode="json"),
    )
    db.add(fund)
    db.flush()  # populate fund.id before linking
    record.confirmed_fund_id = fund.id
    db.commit()
    db.refresh(fund)

    return FundOut(
        id=fund.id,
        name=fund.name,
        inception=series.inception,
        last_observation=series.last_observation,
        n_observations=series.n_observations,
    )


def _source_for_tier(tier: int) -> str:
    return {1: "pdf_tier1", 2: "pdf_tier2", 3: "pdf_tier3", 4: "paste"}.get(tier, "csv")
