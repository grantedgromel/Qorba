"""PDF ingestion cascade.

Pipeline:
  Tier 1 PyMuPDF.find_tables   (fast, deterministic, free)
  Tier 2 pdfplumber             (line + text strategies)
  Tier 3 Claude Sonnet vision   (Sprint 6 — not in this cascade yet)
  Tier 4 Manual paste           (separate endpoint, not in this cascade)

Escalation triggers from Tier 1 → Tier 2:
  - find_tables returned no tables on every page
  - Best Tier 1 candidate had < 12 monthly observations
  - Confidence < ESCALATE_BELOW
The cascade keeps the best of (Tier 1, Tier 2) by length and confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import BytesIO

import pandas as pd

from qorba_api.core.ingestion import pdf_tier1, pdf_tier2
from qorba_api.core.ingestion.scoring import confidence

ESCALATE_BELOW = 0.6
MIN_OBSERVATIONS = 12


@dataclass
class CascadeResult:
    series: pd.Series
    tier_used: int
    confidence: float
    components: dict[str, float]
    warnings: list[str]


def _to_dates(idx: pd.DatetimeIndex) -> list[date]:
    return [pd.Timestamp(t).date() for t in idx]


def _score(series: pd.Series) -> tuple[float, dict[str, float]]:
    if series.empty:
        return 0.0, {"coverage": 0.0, "month_grid": 0.0, "value_sanity": 0.0}
    return confidence(_to_dates(series.index), [float(v) for v in series.values])


def _should_escalate(series: pd.Series, conf: float) -> bool:
    if series.empty:
        return True
    if len(series) < MIN_OBSERVATIONS:
        return True
    if conf < ESCALATE_BELOW:
        return True
    return False


def run_pdf_cascade(raw: bytes) -> CascadeResult:
    """Run Tier 1, escalate to Tier 2 if needed; return the best result."""
    warnings: list[str] = []

    # Tier 1
    s1 = pdf_tier1.extract_returns_from_pdf(BytesIO(raw))
    c1, comp1 = _score(s1)
    if not _should_escalate(s1, c1):
        return CascadeResult(s1, tier_used=1, confidence=c1, components=comp1, warnings=warnings)

    if s1.empty:
        warnings.append("Tier 1 (PyMuPDF) extracted no usable tables; escalating to Tier 2.")
    elif len(s1) < MIN_OBSERVATIONS:
        warnings.append(
            f"Tier 1 found only {len(s1)} observations (< {MIN_OBSERVATIONS}); "
            "escalating to Tier 2.",
        )
    else:
        warnings.append(
            f"Tier 1 confidence {c1:.2f} below threshold {ESCALATE_BELOW:.2f}; "
            "escalating to Tier 2.",
        )

    # Tier 2
    s2 = pdf_tier2.extract_returns_from_pdf(BytesIO(raw))
    c2, comp2 = _score(s2)

    # Pick the better of the two — primary on confidence, tie-break on length.
    candidates = [(s1, 1, c1, comp1), (s2, 2, c2, comp2)]
    candidates = [c for c in candidates if not c[0].empty]
    if not candidates:
        return CascadeResult(
            pd.Series(dtype=float, name="fund"),
            tier_used=2,
            confidence=0.0,
            components={"coverage": 0.0, "month_grid": 0.0, "value_sanity": 0.0},
            warnings=[*warnings, "No tables extracted from any tier."],
        )
    candidates.sort(key=lambda c: (c[2], len(c[0])), reverse=True)
    best_series, best_tier, best_conf, best_comp = candidates[0]
    return CascadeResult(
        series=best_series,
        tier_used=best_tier,
        confidence=best_conf,
        components=best_comp,
        warnings=warnings,
    )
