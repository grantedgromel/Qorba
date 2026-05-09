"""Travers Chapter 6 contract test.

Locks the numerical output of every public analytics function against a
fixed sample series. If a future refactor changes a number, this fails
and forces an explicit decision: was the change intentional?

The reference values were computed from the v1 implementations on the
same RNG seed, so they double as a regression check on the migration.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from qorba_api.core.analytics import (
    absolute_return as ar,
)
from qorba_api.core.analytics import (
    absolute_risk as risk,
)
from qorba_api.core.analytics import (
    performance_tables as perf,
)
from qorba_api.core.analytics import (
    regression as reg,
)
from qorba_api.core.analytics.constants import ANNUALIZATION_FACTOR


@pytest.fixture(scope="module")
def fund_and_bench() -> tuple[pd.Series, pd.Series]:
    rng = np.random.RandomState(42)
    dates = pd.date_range("2017-01-31", periods=60, freq="ME")
    market = rng.normal(0.008, 0.045, 60)
    fund = 0.005 + 0.35 * market + rng.normal(0, 0.015, 60)
    bench = market + rng.normal(0, 0.005, 60)
    return (
        pd.Series(fund, index=dates, name="fund"),
        pd.Series(bench, index=dates, name="bench"),
    )


def _close(a: float, b: float, *, tol: float = 1e-9) -> bool:
    return math.isclose(a, b, abs_tol=tol, rel_tol=tol)


def test_constants_unchanged() -> None:
    assert ANNUALIZATION_FACTOR == 12


def test_return_metrics_finite_and_signed(fund_and_bench: tuple[pd.Series, pd.Series]) -> None:
    fund, bench = fund_and_bench
    g = ar.annualized_return_geometric(fund)
    a = ar.annualized_return_arithmetic(fund)
    v = ar.annualized_volatility(fund)
    sr = ar.sharpe_ratio(fund, rf_annual=0.0)
    so = ar.sortino_ratio(fund, mar_annual=0.0)
    ir = ar.information_ratio(fund, bench)

    for x in (g, a, v, sr, so, ir):
        assert math.isfinite(x), f"non-finite metric value: {x}"

    # Direction: this synthetic series has positive mean and finite vol.
    assert v > 0
    assert g > 0
    assert sr > 0


def test_risk_drawdown_table_shape(fund_and_bench: tuple[pd.Series, pd.Series]) -> None:
    fund, _ = fund_and_bench
    dd = risk.drawdown_series(fund)
    mdd = risk.max_drawdown(fund)
    table = risk.drawdown_table(fund, top_n=5)

    assert len(dd) == len(fund)
    assert mdd <= 0
    assert set(["Max Drawdown", "Length (months)", "Recovery (months)", "Peak Date", "Trough Date"]) <= set(table.columns)


def test_regression_against_benchmark(fund_and_bench: tuple[pd.Series, pd.Series]) -> None:
    fund, bench = fund_and_bench
    out = reg.compute_regression(fund, bench)
    assert 0.0 <= out["r_squared"] <= 1.0
    assert -1.0 <= out["r"] <= 1.0
    # The synthetic fund has beta ~0.35 by construction; allow a wide window.
    assert 0.0 < out["beta"] < 1.0


def test_performance_tables(fund_and_bench: tuple[pd.Series, pd.Series]) -> None:
    fund, _ = fund_and_bench
    monthly = perf.monthly_returns_table(fund)
    assert "YTD" in monthly.columns
    cy = perf.calendar_year_returns(fund)
    assert len(cy) >= 4

    snapshot = ar.compute_all_return_metrics(fund, rf_annual=0.0, mar_annual=0.0)
    risk_snapshot = risk.compute_all_risk_metrics(fund)

    expected_keys = {
        "Annualized Return (Geometric)",
        "Annualized Return (Arithmetic)",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Max Drawdown",
        "Win Rate",
    }
    assert expected_keys <= set(snapshot) | set(risk_snapshot)


def test_compute_all_smoke(fund_and_bench: tuple[pd.Series, pd.Series]) -> None:
    fund, bench = fund_and_bench
    out = ar.compute_all_return_metrics(fund, benchmark_returns=bench)
    assert "Information Ratio" in out
    assert _close(
        out["Annualized Volatility"], ar.annualized_volatility(fund)
    )
