"""Bridge between API selections and the Travers analytics core.

The compute service produces an AnalysisResult slice for a chosen period.
The full monthly_returns and cumulative_growth come back keyed to that
period, which the frontend renders directly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

import pandas as pd

from qorba_api.core.analytics import absolute_return as ar
from qorba_api.core.analytics import absolute_risk as risk
from qorba_api.core.analytics import regression as reg
from qorba_api.schemas.analyses import (
    AnalysisResult,
    MetricValue,
    TimeSeriesPoint,
)
from qorba_api.schemas.returns import ReturnSeries
from qorba_api.services.series import dto_to_series

Period = Literal["3M", "6M", "YTD", "1Y", "3Y", "5Y", "ALL"]
PERIODS: tuple[Period, ...] = ("3M", "6M", "YTD", "1Y", "3Y", "5Y", "ALL")


def slice_period(s: pd.Series, period: Period) -> pd.Series:
    """Slice a monthly return series to the requested trailing window."""
    if s.empty or period == "ALL":
        return s
    if period == "YTD":
        last = s.index[-1]
        start_year = pd.Timestamp(last).year
        return s[s.index.year == start_year]
    months = {"3M": 3, "6M": 6, "1Y": 12, "3Y": 36, "5Y": 60}[period]
    return s.iloc[-months:] if len(s) > months else s


# ── Metric formatters ─────────────────────────────────────────────────


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


# ── Metric registry ───────────────────────────────────────────────────
#
# Each entry carries a label, the compute fn, the formatter, and metadata
# (group + flags) that the /metrics/catalog endpoint surfaces to the
# frontend palette. New metrics drop in by adding a row.


METRIC_REGISTRY: dict[str, dict] = {
    # Returns
    "ann_return_geo": {
        "label": "Annualized Return (Geometric)",
        "group": "Returns",
        "fn": lambda s, ctx: ar.annualized_return_geometric(s),
        "format": _fmt_pct,
        "default": True,
    },
    "ann_return_arith": {
        "label": "Annualized Return (Arithmetic)",
        "group": "Returns",
        "fn": lambda s, ctx: ar.annualized_return_arithmetic(s),
        "format": _fmt_pct,
        "default": False,
    },
    "best_month": {
        "label": "Best Month",
        "group": "Returns",
        "fn": lambda s, ctx: float(s.max()) if len(s) else None,
        "format": _fmt_pct_signed,
        "default": False,
    },
    "worst_month": {
        "label": "Worst Month",
        "group": "Returns",
        "fn": lambda s, ctx: float(s.min()) if len(s) else None,
        "format": _fmt_pct_signed,
        "default": False,
    },
    "avg_gain": {
        "label": "Average Gain",
        "group": "Returns",
        "fn": lambda s, ctx: risk.avg_gain(s),
        "format": _fmt_pct_signed,
        "default": False,
    },
    "avg_loss": {
        "label": "Average Loss",
        "group": "Returns",
        "fn": lambda s, ctx: risk.avg_loss(s),
        "format": _fmt_pct_signed,
        "default": False,
    },
    # Risk
    "ann_vol": {
        "label": "Annualized Volatility",
        "group": "Risk",
        "fn": lambda s, ctx: ar.annualized_volatility(s),
        "format": _fmt_pct,
        "default": True,
    },
    "downside_dev": {
        "label": "Downside Deviation",
        "group": "Risk",
        "fn": lambda s, ctx: risk.downside_deviation(s, ctx.get("mar_annual", 0.0)),
        "format": _fmt_pct,
        "default": False,
    },
    "semideviation": {
        "label": "Semideviation",
        "group": "Risk",
        "fn": lambda s, ctx: risk.semideviation(s),
        "format": _fmt_pct,
        "default": False,
    },
    "skewness": {
        "label": "Skewness",
        "group": "Risk",
        "fn": lambda s, ctx: risk.skewness(s),
        "format": _fmt_ratio,
        "default": False,
    },
    "kurtosis": {
        "label": "Excess Kurtosis",
        "group": "Risk",
        "fn": lambda s, ctx: risk.excess_kurtosis(s),
        "format": _fmt_ratio,
        "default": False,
    },
    # Drawdown
    "max_dd": {
        "label": "Max Drawdown",
        "group": "Drawdown",
        "fn": lambda s, ctx: risk.max_drawdown(s),
        "format": _fmt_pct,
        "default": True,
    },
    "calmar": {
        "label": "Calmar Ratio",
        "group": "Drawdown",
        "fn": lambda s, ctx: ar.calmar_ratio(s),
        "format": _fmt_ratio,
        "default": False,
    },
    "sterling": {
        "label": "Sterling Ratio",
        "group": "Drawdown",
        "fn": lambda s, ctx: ar.sterling_ratio(s),
        "format": _fmt_ratio,
        "default": False,
    },
    "mar": {
        "label": "MAR Ratio",
        "group": "Drawdown",
        "fn": lambda s, ctx: ar.mar_ratio(s),
        "format": _fmt_ratio,
        "default": False,
    },
    # Risk-Adjusted
    "sharpe": {
        "label": "Sharpe Ratio",
        "group": "Risk-Adjusted",
        "fn": lambda s, ctx: ar.sharpe_ratio(s, ctx.get("rf_annual", 0.0)),
        "format": _fmt_ratio,
        "default": True,
    },
    "sortino": {
        "label": "Sortino Ratio",
        "group": "Risk-Adjusted",
        "fn": lambda s, ctx: ar.sortino_ratio(s, ctx.get("mar_annual", 0.0)),
        "format": _fmt_ratio,
        "default": True,
    },
    "omega": {
        "label": "Omega Ratio",
        "group": "Risk-Adjusted",
        "fn": lambda s, ctx: ar.omega_ratio(s, ctx.get("omega_threshold", 0.0)),
        "format": _fmt_ratio,
        "default": False,
    },
    # Distributional
    "win_rate": {
        "label": "Win Rate",
        "group": "Distributional",
        "fn": lambda s, ctx: risk.win_rate(s),
        "format": _fmt_pct,
        "default": True,
    },
    "gain_loss_ratio": {
        "label": "Gain/Loss Ratio",
        "group": "Distributional",
        "fn": lambda s, ctx: risk.gain_loss_ratio(s),
        "format": _fmt_ratio,
        "default": False,
    },
    "positive_months": {
        "label": "Positive Months",
        "group": "Distributional",
        "fn": lambda s, ctx: float((s > 0).sum()),
        "format": lambda v: "—" if v is None else f"{int(v)}",
        "default": False,
    },
    "negative_months": {
        "label": "Negative Months",
        "group": "Distributional",
        "fn": lambda s, ctx: float((s < 0).sum()),
        "format": lambda v: "—" if v is None else f"{int(v)}",
        "default": False,
    },
    # Benchmark-Relative — require a benchmark in ctx (Sprint 4)
    "beta": {
        "label": "Beta",
        "group": "Benchmark-Relative",
        "fn": lambda s, ctx: (
            reg.beta(s, ctx["benchmark"]) if ctx.get("benchmark") is not None else None
        ),
        "format": _fmt_ratio,
        "requires_benchmark": True,
        "default": True,
    },
    "alpha_ann": {
        "label": "Alpha (Annualized)",
        "group": "Benchmark-Relative",
        "fn": lambda s, ctx: (
            reg.alpha(s, ctx["benchmark"]) * 12 if ctx.get("benchmark") is not None else None
        ),
        "format": _fmt_pct,
        "requires_benchmark": True,
        "default": True,
    },
}


def metrics_catalog() -> list[dict]:
    """Surfaceable catalog for the frontend palette."""
    return [
        {
            "id": mid,
            "label": spec["label"],
            "group": spec["group"],
            "default": spec.get("default", False),
            "requires_benchmark": spec.get("requires_benchmark", False),
        }
        for mid, spec in METRIC_REGISTRY.items()
    ]


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
    period: Period = "ALL",
    rf_annual: float = 0.0,
    mar_annual: float = 0.0,
    omega_threshold: float = 0.0,
) -> AnalysisResult:
    s_full = dto_to_series(fund_series)
    s = slice_period(s_full, period)

    ctx = {
        "rf_annual": rf_annual,
        "mar_annual": mar_annual,
        "omega_threshold": omega_threshold,
        "benchmark": None,
    }

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
        "period": period,
        "rf_annual": rf_annual,
        "mar_annual": mar_annual,
        "omega_threshold": omega_threshold,
    }
    version_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    inception = monthly[0].period if monthly else fund_series.inception
    last_obs = monthly[-1].period if monthly else fund_series.last_observation

    return AnalysisResult(
        analysis_id=analysis_id,
        metrics=metrics,
        monthly_returns=monthly,
        cumulative_growth=growth,
        fund_name=fund_series.name,
        inception=inception,
        last_observation=last_obs,
        period=period,
        computed_at=datetime.now(UTC),
        version_hash=version_hash,
    )
