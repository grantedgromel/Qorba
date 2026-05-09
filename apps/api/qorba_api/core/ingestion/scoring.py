"""Confidence scoring for an extracted return series.

The cascade uses three components, each in [0, 1]:
  coverage_score    months extracted / months expected (inception → last)
  month_grid_score  penalize gaps in the year-month grid
  value_sanity_score penalize values where |r| > 0.5 (Travers-grade outliers)

Final confidence is the geometric mean of the three so a low score in any
single dimension drags the overall down.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import date


def _months_between(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month) + 1


def coverage_score(periods: list[date]) -> float:
    if len(periods) < 2:
        return 0.0
    expected = _months_between(min(periods), max(periods))
    actual = len(periods)
    return min(1.0, actual / expected)


def month_grid_score(periods: list[date]) -> float:
    if len(periods) < 2:
        return 0.0
    sorted_periods = sorted(periods)
    expected = _months_between(sorted_periods[0], sorted_periods[-1])
    seen = {(p.year, p.month) for p in sorted_periods}
    return len(seen) / expected if expected else 0.0


def value_sanity_score(values: Iterable[float]) -> float:
    """Penalize outliers. Heuristically normalize to decimal first so that
    percent-scale inputs aren't flagged en masse — the cascade runs before
    the user has confirmed scale."""
    vals = list(values)
    if not vals:
        return 0.0
    if any(abs(v) > 1.0 for v in vals):
        vals = [v / 100.0 for v in vals]
    outliers = sum(1 for v in vals if abs(v) > 0.5)
    return max(0.0, 1.0 - outliers / len(vals))


def confidence(
    periods: list[date], values: list[float]
) -> tuple[float, dict[str, float]]:
    cov = coverage_score(periods)
    grid = month_grid_score(periods)
    sanity = value_sanity_score(values)
    geo = math.pow(max(cov, 1e-9) * max(grid, 1e-9) * max(sanity, 1e-9), 1.0 / 3.0)
    return geo, {
        "coverage": cov,
        "month_grid": grid,
        "value_sanity": sanity,
    }
