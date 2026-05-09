"""Provider protocol for benchmark data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol


@dataclass(frozen=True)
class BenchmarkInfo:
    code: str
    name: str
    category: str
    provider: str


@dataclass(frozen=True)
class BenchmarkPoint:
    period: date
    value: float  # decimal


@dataclass(frozen=True)
class BenchmarkSeries:
    info: BenchmarkInfo
    points: list[BenchmarkPoint]


class BenchmarkProvider(Protocol):
    """Anything that can list and fetch monthly benchmark return series."""

    name: str

    def list_benchmarks(self) -> list[BenchmarkInfo]: ...

    def fetch_returns(
        self,
        code: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> BenchmarkSeries: ...
