"""Extract monthly return series from PDF tearsheets using PyMuPDF (fitz)."""

import calendar
import re

import numpy as np
import pandas as pd
import pymupdf
from io import BytesIO


# Month name/abbreviation lookup (case-insensitive)
_MONTH_NAMES = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
_MONTH_NAMES.update({m.lower(): i for i, m in enumerate(calendar.month_name) if m})


def _clean_cell(cell: str | None) -> str | None:
    """Strip whitespace and normalize dashes."""
    if cell is None:
        return None
    cell = str(cell).strip().replace("\u2013", "-").replace("\u2014", "-")
    return cell if cell not in ("", "-", "N/A", "n/a", "--") else None


def _parse_return_value(text: str | None) -> float | None:
    """Parse a return string like '1.23%', '-0.45', '(1.23)' into a float."""
    if text is None:
        return None
    text = text.strip().rstrip("%")
    # Handle parenthetical negatives: (1.23) -> -1.23
    m = re.match(r"^\(([0-9.]+)\)$", text)
    if m:
        text = "-" + m.group(1)
    try:
        return float(text)
    except ValueError:
        return None


def _is_year(val: str | None) -> int | None:
    """Check if a string looks like a 4-digit year (1990-2099)."""
    if val is None:
        return None
    val = val.strip()
    m = re.match(r"^(19|20)\d{2}$", val)
    return int(val) if m else None


def _detect_calendar_table(table: list[list[str]]) -> bool:
    """Heuristic: does this table have month headers (Jan..Dec)?"""
    if not table or len(table) < 2:
        return False
    header = [_clean_cell(c) for c in table[0]]
    header_lower = [h.lower() if h else "" for h in header]
    month_matches = sum(1 for h in header_lower if h in _MONTH_NAMES)
    return month_matches >= 6  # At least half the months present


def _extract_from_calendar_table(table: list[list[str]]) -> pd.Series:
    """Extract returns from a calendar-style table (rows=years, cols=months).

    Typical layout:
        Year | Jan  | Feb  | ... | Dec  | YTD
        2020 | 1.23 | -0.5 | ... | 0.80 | 12.3
    """
    header = [_clean_cell(c) for c in table[0]]
    header_lower = [h.lower() if h else "" for h in header]

    # Map column index -> month number
    col_to_month = {}
    for i, h in enumerate(header_lower):
        if h in _MONTH_NAMES:
            col_to_month[i] = _MONTH_NAMES[h]

    dates = []
    values = []

    for row in table[1:]:
        cleaned = [_clean_cell(c) for c in row]
        # Find the year in this row (usually first column)
        year = None
        for c in cleaned:
            year = _is_year(c)
            if year:
                break
        if year is None:
            continue

        for col_idx, month_num in col_to_month.items():
            if col_idx >= len(cleaned):
                continue
            val = _parse_return_value(cleaned[col_idx])
            if val is not None:
                dt = pd.Timestamp(year=year, month=month_num, day=1)
                dates.append(dt)
                values.append(val)

    if not dates:
        return pd.Series(dtype=float)

    s = pd.Series(values, index=pd.DatetimeIndex(dates), name="fund")
    s = s.sort_index()
    # Remove duplicates (keep first)
    s = s[~s.index.duplicated(keep="first")]
    return s


def _detect_vertical_table(table: list[list[str]]) -> bool:
    """Heuristic: table with date column + return column (vertical layout)."""
    if not table or len(table) < 3:
        return False
    # Check if first column looks like dates
    date_count = 0
    for row in table[1:6]:  # Check first few data rows
        if not row:
            continue
        cell = _clean_cell(row[0])
        if cell:
            try:
                pd.to_datetime(cell)
                date_count += 1
            except (ValueError, TypeError):
                pass
    return date_count >= 2


def _extract_from_vertical_table(table: list[list[str]]) -> pd.Series:
    """Extract returns from a date-value vertical table."""
    dates = []
    values = []
    for row in table[1:]:
        if len(row) < 2:
            continue
        date_cell = _clean_cell(row[0])
        val_cell = _clean_cell(row[1])
        if date_cell is None:
            continue
        try:
            dt = pd.to_datetime(date_cell)
        except (ValueError, TypeError):
            continue
        val = _parse_return_value(val_cell)
        if val is not None:
            dates.append(dt)
            values.append(val)

    if not dates:
        return pd.Series(dtype=float)

    s = pd.Series(values, index=pd.DatetimeIndex(dates), name="fund")
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="first")]
    return s


def _normalize_returns(series: pd.Series) -> pd.Series:
    """Convert percentage returns to decimal if needed."""
    if series.empty:
        return series
    if series.abs().max() > 1.0:
        return series / 100.0
    return series


def _tables_from_page(page) -> list[list[list[str]]]:
    """Extract tables from a PyMuPDF page using find_tables()."""
    tables = []
    tab_finder = page.find_tables()
    for tab in tab_finder.tables:
        rows = tab.extract()
        if rows:
            tables.append(rows)
    return tables


def extract_returns_from_pdf(file_bytes: BytesIO) -> pd.Series:
    """Main entry point: extract a monthly return series from a PDF tearsheet.

    Tries two strategies per table found:
    1. Calendar-style table (Year | Jan | Feb | ... | Dec)
    2. Vertical date-value table

    Returns a pd.Series with DatetimeIndex and decimal returns.
    """
    all_series = []

    doc = pymupdf.open(stream=file_bytes.read(), filetype="pdf")
    try:
        for page in doc:
            tables = _tables_from_page(page)
            for table in tables:
                if not table:
                    continue

                if _detect_calendar_table(table):
                    s = _extract_from_calendar_table(table)
                    if not s.empty:
                        all_series.append(s)
                elif _detect_vertical_table(table):
                    s = _extract_from_vertical_table(table)
                    if not s.empty:
                        all_series.append(s)
    finally:
        doc.close()

    if not all_series:
        return pd.Series(dtype=float)

    # Pick the longest series (most likely the main return table)
    best = max(all_series, key=len)
    return _normalize_returns(best)
