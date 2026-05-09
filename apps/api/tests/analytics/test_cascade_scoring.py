from __future__ import annotations

from datetime import date

from qorba_api.core.ingestion.scoring import (
    confidence,
    coverage_score,
    month_grid_score,
    value_sanity_score,
)


def _months(start_year: int, start_month: int, n: int) -> list[date]:
    out: list[date] = []
    y, m = start_year, start_month
    for _ in range(n):
        out.append(date(y, m, 28))
        m += 1
        if m == 13:
            m = 1
            y += 1
    return out


def test_full_grid_max_scores() -> None:
    periods = _months(2020, 1, 36)
    values = [0.01] * 36
    c, parts = confidence(periods, values)
    assert parts["coverage"] == 1.0
    assert parts["month_grid"] == 1.0
    assert parts["value_sanity"] == 1.0
    assert c == 1.0


def test_gaps_drag_grid_score_down() -> None:
    periods = _months(2020, 1, 24)
    del periods[5]
    del periods[12]
    values = [0.01] * len(periods)
    _, parts = confidence(periods, values)
    assert parts["coverage"] < 1.0
    assert parts["month_grid"] < 1.0


def test_outliers_drag_sanity_score() -> None:
    # All decimal-scale; one absurd 80% month should be flagged.
    periods = _months(2020, 1, 12)
    values = [0.01] * 11 + [0.8]
    _, parts = confidence(periods, values)
    assert parts["value_sanity"] < 1.0
    assert value_sanity_score(values) < 1.0


def test_percent_scale_inputs_not_falsely_flagged() -> None:
    # Reasonable percent-scale returns (max ~6%) should not trip the sanity check.
    values = [1.2, -0.4, 2.1, 0.6, 5.9, -3.2, 0.0, 1.1, 2.4, -1.5, 0.8, 4.0]
    assert value_sanity_score(values) == 1.0


def test_empty_zero_scores() -> None:
    assert coverage_score([]) == 0.0
    assert month_grid_score([]) == 0.0
    assert value_sanity_score([]) == 0.0
