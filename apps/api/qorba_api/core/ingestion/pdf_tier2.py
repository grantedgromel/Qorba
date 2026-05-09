"""Tier 2 PDF parser using pdfplumber.

Falls back from Tier 1 (PyMuPDF find_tables) when:
  - find_tables found nothing usable
  - the best Tier 1 table had < 12 observations
  - confidence after Tier 1 was below threshold

pdfplumber gives us word-level geometry, so we can pull stream-style tables
that PyMuPDF misses. We try both `lines` (lattice) and `text` (stream)
strategies and keep the better result.

Camelot integration is gated behind QORBA_ENABLE_CAMELOT to avoid the
Ghostscript/system-dep tax in the default container; not implemented in
Sprint 2.
"""

from __future__ import annotations

import calendar
import re
from io import BytesIO

import pandas as pd
import pdfplumber

_MONTH_NAMES: dict[str, int] = {}
for i, m in enumerate(calendar.month_abbr):
    if m:
        _MONTH_NAMES[m.lower()] = i
for i, m in enumerate(calendar.month_name):
    if m:
        _MONTH_NAMES[m.lower()] = i


def _clean(cell: str | None) -> str | None:
    if cell is None:
        return None
    s = str(cell).strip().replace("–", "-").replace("—", "-")  # noqa: RUF001
    return s if s not in ("", "-", "N/A", "n/a", "--") else None


def _parse_value(text: str | None) -> float | None:
    if text is None:
        return None
    text = text.strip().rstrip("%").replace(",", "")
    m = re.match(r"^\(([0-9.]+)\)$", text)
    if m:
        text = "-" + m.group(1)
    try:
        return float(text)
    except ValueError:
        return None


def _is_year(val: str | None) -> int | None:
    if val is None:
        return None
    m = re.match(r"^(19|20)\d{2}$", val.strip())
    return int(val.strip()) if m else None


def _detect_calendar_table(table: list[list[str | None]]) -> bool:
    if not table or len(table) < 2:
        return False
    header = [_clean(c) for c in table[0]]
    matches = sum(1 for h in header if h and h.lower() in _MONTH_NAMES)
    return matches >= 6


def _extract_calendar(table: list[list[str | None]]) -> pd.Series:
    header = [_clean(c) for c in table[0]]
    col_to_month: dict[int, int] = {}
    for i, h in enumerate(header):
        if h and h.lower() in _MONTH_NAMES:
            col_to_month[i] = _MONTH_NAMES[h.lower()]
    dates: list[pd.Timestamp] = []
    values: list[float] = []
    for row in table[1:]:
        cleaned = [_clean(c) for c in row]
        year = next((_is_year(c) for c in cleaned if _is_year(c)), None)
        if year is None:
            continue
        for col_idx, month_num in col_to_month.items():
            if col_idx >= len(cleaned):
                continue
            v = _parse_value(cleaned[col_idx])
            if v is not None:
                dates.append(pd.Timestamp(year=year, month=month_num, day=1))
                values.append(v)
    if not dates:
        return pd.Series(dtype=float, name="fund")
    s = pd.Series(values, index=pd.DatetimeIndex(dates), name="fund").sort_index()
    return s[~s.index.duplicated(keep="first")]


def _detect_vertical_table(table: list[list[str | None]]) -> bool:
    if not table or len(table) < 3:
        return False
    hits = 0
    for row in table[1:6]:
        if not row:
            continue
        c = _clean(row[0])
        if not c:
            continue
        try:
            pd.to_datetime(c)
            hits += 1
        except (ValueError, TypeError):
            pass
    return hits >= 2


def _extract_vertical(table: list[list[str | None]]) -> pd.Series:
    dates: list[pd.Timestamp] = []
    values: list[float] = []
    for row in table[1:]:
        if len(row) < 2:
            continue
        d = _clean(row[0])
        v = _clean(row[1])
        if d is None:
            continue
        try:
            ts = pd.to_datetime(d)
        except (ValueError, TypeError):
            continue
        parsed = _parse_value(v)
        if parsed is not None:
            dates.append(ts)
            values.append(parsed)
    if not dates:
        return pd.Series(dtype=float, name="fund")
    s = pd.Series(values, index=pd.DatetimeIndex(dates), name="fund").sort_index()
    return s[~s.index.duplicated(keep="first")]


def _normalize(s: pd.Series) -> pd.Series:
    """No silent rescaling — that's now the user's choice. Just clean."""
    if s.empty:
        return s
    return s


def _try_extract(table: list[list[str | None]]) -> pd.Series:
    if not table:
        return pd.Series(dtype=float, name="fund")
    if _detect_calendar_table(table):
        return _extract_calendar(table)
    if _detect_vertical_table(table):
        return _extract_vertical(table)
    return pd.Series(dtype=float, name="fund")


def extract_returns_from_pdf(file_bytes: BytesIO) -> pd.Series:
    """Tier 2 extraction.

    Try lattice (lines) and stream (text) strategies; keep the longest
    series found across all pages.
    """
    raw = file_bytes.read()
    candidates: list[pd.Series] = []
    with pdfplumber.open(BytesIO(raw)) as pdf:
        for page in pdf.pages:
            for strategy in ("lines", "text"):
                try:
                    tables = page.extract_tables(
                        table_settings={
                            "vertical_strategy": strategy,
                            "horizontal_strategy": strategy,
                        }
                    )
                except Exception:
                    tables = []
                for tab in tables or []:
                    s = _try_extract(tab)
                    if not s.empty:
                        candidates.append(s)
    if not candidates:
        return pd.Series(dtype=float, name="fund")
    return _normalize(max(candidates, key=len))
