"""Benchmark library + uploaded benchmarks."""

from __future__ import annotations

import hashlib
import uuid
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from qorba_api.auth import get_current_user
from qorba_api.db.models import Benchmark, User
from qorba_api.db.session import get_db
from qorba_api.schemas.benchmarks import (
    BenchmarkLibrary,
    BenchmarkLibraryItem,
    BenchmarkOut,
    BenchmarkUploadFromIngest,
)
from qorba_api.services.benchmark_provider import get_benchmark_provider

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


@router.get("/library", response_model=BenchmarkLibrary)
def library(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> BenchmarkLibrary:
    provider = get_benchmark_provider()
    items: list[BenchmarkLibraryItem] = [
        BenchmarkLibraryItem(code=b.code, name=b.name, category=b.category, provider=b.provider)
        for b in provider.list_benchmarks()
    ]
    user_benchmarks = db.scalars(select(Benchmark).where(Benchmark.user_id == user.id)).all()
    for ub in user_benchmarks:
        items.append(
            BenchmarkLibraryItem(
                code=ub.code or str(ub.id),
                name=ub.name,
                category="custom",
                provider="user",
            )
        )
    return BenchmarkLibrary(items=items, provider_active=provider.name)


@router.post("/upload", response_model=BenchmarkOut, status_code=status.HTTP_201_CREATED)
def upload(
    body: BenchmarkUploadFromIngest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> BenchmarkOut:
    bench = Benchmark(
        user_id=user.id,
        name=body.name,
        code=body.code,
        provider="user",
        is_user_uploaded=True,
        series=body.series.model_dump(mode="json"),
    )
    db.add(bench)
    db.commit()
    db.refresh(bench)
    return BenchmarkOut(
        id=bench.id,
        name=bench.name,
        code=bench.code,
        provider=bench.provider,
        inception=body.series.inception,
        last_observation=body.series.last_observation,
        n_observations=body.series.n_observations,
    )


@router.get("/{code}/returns")
def fetch_provider_returns(
    code: str,
    _user: User = Depends(get_current_user),
):
    """Fetch a series from the active BenchmarkProvider (not user uploads)."""
    provider = get_benchmark_provider()
    try:
        s = provider.fetch_returns(code)
    except KeyError as err:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"Unknown benchmark code: {code}"
        ) from err
    return {
        "code": s.info.code,
        "name": s.info.name,
        "category": s.info.category,
        "provider": s.info.provider,
        "points": [{"period": p.period.isoformat(), "value": p.value} for p in s.points],
    }


__all__ = ["router"]


# Helper used by ingest paths to checksum a series; kept here so that tests
# of benchmarks see the same canonical shape as funds.
def checksum_returns(points: list[tuple[date, float]]) -> str:
    h = hashlib.sha256()
    h.update(
        ",".join(f"{d.isoformat()}:{round(v, 12)}" for d, v in points).encode()
    )
    return h.hexdigest()


def _new_uuid() -> uuid.UUID:
    return uuid.uuid4()
