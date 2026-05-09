"""Bridge between API selections and the Travers analytics core.

Sprint 1 supports a small subset of metrics. The catalog grows in Sprint 3.
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
from qorba_api.schemas.analyses import AnalysisResult, MetricValue
from qorba_api.schemas.returns import ReturnSeries
from qorba_api.services.series import dto_to_series


def _fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.2f}%"


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
}


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
        computed_at=datetime.now(UTC),
        version_hash=version_hash,
    )
