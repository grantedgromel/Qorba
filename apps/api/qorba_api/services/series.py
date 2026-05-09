"""Convert between ReturnSeries (API DTO) and pd.Series (analytics core)."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import date

import pandas as pd

from qorba_api.schemas.returns import ReturnPoint, ReturnSeries


def series_to_dto(series: pd.Series, *, name: str, source: str) -> ReturnSeries:
    if series.empty:
        raise ValueError("Cannot create a ReturnSeries from an empty pd.Series")
    points = [
        ReturnPoint(period=_to_date(idx), value=float(val))
        for idx, val in series.items()
    ]
    return ReturnSeries(
        id=uuid.uuid4(),
        name=name,
        points=points,
        inception=points[0].period,
        last_observation=points[-1].period,
        n_observations=len(points),
        source=source,  # type: ignore[arg-type]
        checksum=_checksum(points),
    )


def dto_to_series(series: ReturnSeries) -> pd.Series:
    idx = pd.DatetimeIndex([p.period for p in series.points])
    values = [p.value for p in series.points]
    return pd.Series(values, index=idx, name=series.name, dtype=float)


def _to_date(idx) -> date:
    if isinstance(idx, date) and not isinstance(idx, pd.Timestamp):
        return idx
    return pd.Timestamp(idx).date()


def _checksum(points: list[ReturnPoint]) -> str:
    h = hashlib.sha256()
    h.update(
        json.dumps(
            [(p.period.isoformat(), round(p.value, 12)) for p in points],
            separators=(",", ":"),
        ).encode()
    )
    return h.hexdigest()
