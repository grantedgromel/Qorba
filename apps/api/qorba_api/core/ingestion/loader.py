"""CSV/Excel parsing, validation, and date alignment for fund return data.

The pct/decimal heuristic still runs here, but the v1 silent rescaling of
percentage values is removed: the raw scale decision is surfaced to the
caller so the UI can confirm it (Section 8 decision #3 in the plan).
"""

from io import BytesIO
from typing import Literal

import numpy as np
import pandas as pd

Scale = Literal["percent", "decimal"]


def _parse_dates(series: pd.Series) -> pd.DatetimeIndex:
    """Try multiple date formats and return a DatetimeIndex."""
    for fmt in [None, "%Y-%m-%d", "%m/%d/%Y", "%Y-%m", "%d/%m/%Y", "%Y%m%d"]:
        try:
            return pd.to_datetime(series, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.to_datetime(series, format="mixed", dayfirst=False)


def detect_scale(df: pd.DataFrame) -> Scale:
    """If any numeric column has |max| > 1.0 the values are percentages."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].abs().max() > 1.0:
            return "percent"
    return "decimal"


def apply_scale(df: pd.DataFrame, scale: Scale) -> pd.DataFrame:
    """Convert percent values to decimal. Decimal passes through."""
    if scale == "decimal":
        return df
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols] / 100.0
    return out


def _find_date_column(df: pd.DataFrame) -> str:
    date_names = ["date", "dates", "month", "period", "time", "timestamp"]
    for col in df.columns:
        if col.lower().strip() in date_names:
            return col
    return df.columns[0]


def load_returns(filename: str, raw: bytes) -> tuple[pd.DataFrame, Scale]:
    """Load returns from a CSV or Excel byte payload.

    Returns the raw (unscaled) DataFrame and the heuristic-detected scale.
    Scaling is applied separately by the caller after user confirmation.
    """
    if not raw:
        return pd.DataFrame(), "decimal"

    buf = BytesIO(raw)
    name = filename.lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(buf)
    else:
        df = pd.read_csv(buf)

    if df.empty:
        return df, "decimal"

    date_col = _find_date_column(df)
    df[date_col] = _parse_dates(df[date_col])
    df = df.set_index(date_col)
    df.index.name = "date"
    df = df.sort_index()
    df = df.dropna(how="all").dropna(axis=1, how="all")

    return df, detect_scale(df)


def load_fund_returns(filename: str, raw: bytes) -> tuple[pd.Series, Scale]:
    """Load a single-column fund return series."""
    df, scale = load_returns(filename, raw)
    if df.empty:
        return pd.Series(dtype=float, name="fund"), scale
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.Series(dtype=float, name="fund"), scale
    s = numeric.iloc[:, 0].dropna()
    s.name = "fund"
    return s, scale


def align_dates(*frames):
    """Align multiple DataFrames/Series to their common date intersection."""
    valid = [d for d in frames if d is not None and not d.empty]
    if len(valid) <= 1:
        return list(frames)

    common_idx = valid[0].index
    for d in valid[1:]:
        common_idx = common_idx.intersection(d.index)

    result = []
    for d in frames:
        if d is None or d.empty:
            result.append(d)
        else:
            result.append(d.loc[d.index.intersection(common_idx)])
    return result


def validate_data(series: pd.Series) -> list[str]:
    """Return validation warnings for a return series."""
    warnings: list[str] = []
    if series.empty:
        warnings.append("No data loaded.")
        return warnings
    if series.isna().sum() > 0:
        warnings.append(f"{series.isna().sum()} missing values found (dropped).")
    if len(series) < 12:
        warnings.append("Less than 12 months of data — some metrics may be unreliable.")
    if len(series) < 36:
        warnings.append("Less than 36 months — Calmar ratio unavailable.")
    if (series.abs() > 0.5).any():
        warnings.append("Some monthly returns exceed 50% — verify data is correct.")
    if series.index.duplicated().any():
        warnings.append("Duplicate dates detected — only first occurrence kept.")
    return warnings
