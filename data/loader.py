"""CSV/Excel parsing, validation, and date alignment for fund return data."""

import pandas as pd
import numpy as np
from io import BytesIO


def _parse_dates(series: pd.Series) -> pd.DatetimeIndex:
    """Try multiple date formats and return a DatetimeIndex."""
    for fmt in [None, "%Y-%m-%d", "%m/%d/%Y", "%Y-%m", "%d/%m/%Y", "%Y%m%d"]:
        try:
            return pd.to_datetime(series, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.to_datetime(series, format="mixed", dayfirst=False)


def _detect_and_normalize_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-detect if returns are in percentage or decimal form and normalize to decimal."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].abs().max() > 1.0:
            df[col] = df[col] / 100.0
    return df


def _find_date_column(df: pd.DataFrame) -> str:
    """Identify the date column by name or by type."""
    date_names = ["date", "dates", "month", "period", "time", "timestamp"]
    for col in df.columns:
        if col.lower().strip() in date_names:
            return col
    # Try first column
    return df.columns[0]


def load_returns(uploaded_file) -> pd.DataFrame:
    """Load fund, benchmark, or peer returns from a CSV or Excel upload.

    Returns a DataFrame with DatetimeIndex and decimal return columns.
    """
    if uploaded_file is None:
        return pd.DataFrame()

    name = uploaded_file.name.lower()
    raw = BytesIO(uploaded_file.read())

    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(raw)
    else:
        df = pd.read_csv(raw)

    if df.empty:
        return df

    # Find and set date index
    date_col = _find_date_column(df)
    df[date_col] = _parse_dates(df[date_col])
    df = df.set_index(date_col)
    df.index.name = "date"
    df = df.sort_index()

    # Drop fully-null rows/cols
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # Normalize returns to decimal
    df = _detect_and_normalize_returns(df)

    return df


def load_fund_returns(uploaded_file) -> pd.Series:
    """Load a single-column fund return series."""
    df = load_returns(uploaded_file)
    if df.empty:
        return pd.Series(dtype=float)
    # Take the first numeric column
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.Series(dtype=float)
    s = numeric.iloc[:, 0].dropna()
    s.name = "fund"
    return s


def align_dates(*dataframes: pd.DataFrame | pd.Series) -> list:
    """Align multiple DataFrames/Series to their common date intersection."""
    valid = [d for d in dataframes if d is not None and not d.empty]
    if len(valid) <= 1:
        return list(dataframes)

    # Find common dates
    common_idx = valid[0].index
    for d in valid[1:]:
        common_idx = common_idx.intersection(d.index)

    result = []
    for d in dataframes:
        if d is None or d.empty:
            result.append(d)
        else:
            result.append(d.loc[d.index.intersection(common_idx)])
    return result


def validate_data(series: pd.Series) -> list[str]:
    """Return a list of validation warnings for a return series."""
    warnings = []
    if series.empty:
        warnings.append("No data loaded.")
        return warnings

    if series.isna().sum() > 0:
        warnings.append(f"{series.isna().sum()} missing values found (dropped).")

    if len(series) < 12:
        warnings.append("Less than 12 months of data — some metrics may be unreliable.")

    if len(series) < 36:
        warnings.append("Less than 36 months — Calmar ratio unavailable.")

    # Check for suspicious values
    if (series.abs() > 0.5).any():
        warnings.append("Some monthly returns exceed 50% — verify data is correct.")

    # Check for duplicate dates
    if series.index.duplicated().any():
        warnings.append("Duplicate dates detected — only first occurrence kept.")

    return warnings
