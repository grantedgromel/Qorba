"""Bridge between API selections and the Travers analytics core.

Sprint 2+: alongside the metric grid, the result also carries the full
monthly_returns and cumulative_growth time series, so the frontend can
render a hero chart + month strip without re-fetching the fund.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID

import pandas as pd

from qorba_api.core.analytics import absolute_return as ar
from qorba_api.core.analytics import absolute_risk as risk
from qorba_api.schemas.analyses import (
    AnalysisResult,
    MetricValue,
    TimeSeriesPoint,
)
from qorba_api.schemas.returns import ReturnSeries
from qorba_api.services.series import dto_to_series


def _fmt_pct(x: float | None, *, signed: bool = False) -> str:
    if x is None:
        return "—"
    val = x * 100
    sign = "+" if signed and val > 0 else ""
    return f"{sign}{val:.2f}%"


def _fmt_pct_signed(x: float | None) -> str:
    return _fmt_pct(x, signed=True)


def _fmt_ratio(x: float | None) -> str:
    return "—" if x is None else f"{x:.2f}"


METRIC_REGISTRY: dict[str, dict] = {
    "ann_return_geo": {
        "label": "Annualized Return (Geometric)",
        "fn": lambda s, ctx: ar.annualized_return_geometric(s),
        "format": _fmt_pct,
    },
    "ann_vol": {
        "label": "Annualized Volatility",
        "fn": lambda s, ctx: ar.annualized_volatility(s),
        "format": _fmt_pct,
    },
    "sharpe": {
        "label": "Sharpe Ratio",
        "fn": lambda s, ctx: ar.sharpe_ratio(s, ctx.get("rf_annual", 0.0)),
        "format": _fmt_ratio,
    },
    "sortino": {
        "label": "Sortino Ratio",
        "fn": lambda s, ctx: ar.sortino_ratio(s, ctx.get("mar_annual", 0.0)),
        "format": _fmt_ratio,
    },
    "max_dd": {
        "label": "Max Drawdown",
        "fn": lambda s, ctx: risk.max_drawdown(s),
        "format": _fmt_pct,
    },
    "win_rate": {
        "label": "Win Rate",
        "fn": lambda s, ctx: risk.win_rate(s),
        "format": _fmt_pct,
    },
    "best_month": {
        "label": "Best Month",
        "fn": lambda s, ctx: float(s.max()) if len(s) else None,
        "format": _fmt_pct_signed,
    },
    "worst_month": {
        "label": "Worst Month",
        "fn": lambda s, ctx: float(s.min()) if len(s) else None,
        "format": _fmt_pct_signed,
    },
    "avg_gain": {
        "label": "Average Gain",
        "fn": lambda s, ctx: risk.avg_gain(s),
        "format": _fmt_pct_signed,
    },
    "avg_loss": {
        "label": "Average Loss",
        "fn": lambda s, ctx: risk.avg_loss(s),
        "format": _fmt_pct_signed,
    },
}


def _cumulative_growth(series: pd.Series, base: float = 100.0) -> list[TimeSeriesPoint]:
    if series.empty:
        return []
    growth = (1 + series).cumprod() * base
    return [
        TimeSeriesPoint(period=pd.Timestamp(idx).date(), value=float(v))
        for idx, v in growth.items()
    ]


def compute_analysis(
    *,
    analysis_id: UUID,
    fund_series: ReturnSeries,
    metric_ids: list[str],
    rf_annual: float = 0.0,
    mar_annual: float = 0.0,
) -> AnalysisResult:
    s = dto_to_series(fund_series)
    ctx = {"rf_annual": rf_annual, "mar_annual": mar_annual}

    metrics: dict[str, MetricValue] = {}
    for mid in metric_ids:
        spec = METRIC_REGISTRY.get(mid)
        if spec is None:
            metrics[mid] = MetricValue(metric_id=mid, value=None, formatted="—")
            continue
        fn: Callable[[pd.Series, dict], float | None] = spec["fn"]
        try:
            value = fn(s, ctx)
            value = float(value) if value is not None else None
        except Exception:
            value = None
        metrics[mid] = MetricValue(
            metric_id=mid,
            value=value,
            formatted=spec["format"](value),
        )

    monthly = [
        TimeSeriesPoint(period=pd.Timestamp(idx).date(), value=float(v))
        for idx, v in s.items()
    ]
    growth = _cumulative_growth(s)

    payload = {
        "fund_checksum": fund_series.checksum,
        "metric_ids": sorted(metric_ids),
        "rf_annual": rf_annual,
        "mar_annual": mar_annual,
    }
    version_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return AnalysisResult(
        analysis_id=analysis_id,
        metrics=metrics,
        monthly_returns=monthly,
        cumulative_growth=growth,
        fund_name=fund_series.name,
        inception=fund_series.inception,
        last_observation=fund_series.last_observation,
        computed_at=datetime.now(UTC),
        version_hash=version_hash,
    )
