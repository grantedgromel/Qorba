"""Deterministic in-memory provider for tests and offline dev.

Returns a small synthetic universe (one per category we care about) so
the benchmark UI has something to render before Caissa is configured.
"""

from __future__ import annotations

from datetime import date
from typing import ClassVar

import numpy as np
import pandas as pd

from qorba_api.core.benchmarks.provider import (
    BenchmarkInfo,
    BenchmarkPoint,
    BenchmarkProvider,
    BenchmarkSeries,
)


def _series(seed: int, mu: float, sigma: float, n: int = 96) -> list[BenchmarkPoint]:
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2018-01-31", periods=n, freq="ME")
    values = rng.normal(mu, sigma, n)
    return [
        BenchmarkPoint(period=d.date(), value=float(v))
        for d, v in zip(dates, values, strict=True)
    ]


class StaticProvider(BenchmarkProvider):
    name = "static"

    _UNIVERSE: ClassVar[dict[str, BenchmarkInfo]] = {
        "SPX_TR": BenchmarkInfo(
            code="SPX_TR", name="S&P 500 Total Return", category="equity_us", provider="static"
        ),
        "RUT_TR": BenchmarkInfo(
            code="RUT_TR", name="Russell 2000 Total Return", category="equity_us", provider="static"
        ),
        "ACWI": BenchmarkInfo(
            code="ACWI", name="MSCI ACWI", category="equity_intl", provider="static"
        ),
        "HFRI_FW": BenchmarkInfo(
            code="HFRI_FW",
            name="HFRI Fund Weighted Composite",
            category="hedge_fund_index",
            provider="static",
        ),
    }

    _SEED_PARAMS: ClassVar[dict[str, tuple[int, float, float]]] = {
        "SPX_TR": (1, 0.009, 0.045),
        "RUT_TR": (2, 0.008, 0.058),
        "ACWI": (3, 0.007, 0.041),
        "HFRI_FW": (4, 0.005, 0.022),
    }

    def list_benchmarks(self) -> list[BenchmarkInfo]:
        return list(self._UNIVERSE.values())

    def fetch_returns(
        self,
        code: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> BenchmarkSeries:
        info = self._UNIVERSE.get(code)
        if info is None:
            raise KeyError(f"Unknown benchmark code: {code}")
        seed, mu, sigma = self._SEED_PARAMS[code]
        points = _series(seed, mu, sigma)
        if start is not None:
            points = [p for p in points if p.period >= start]
        if end is not None:
            points = [p for p in points if p.period <= end]
        return BenchmarkSeries(info=info, points=points)
