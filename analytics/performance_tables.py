"""Calendar year performance tables and up/down capture analysis."""

import numpy as np
import pandas as pd
from config.settings import ANNUALIZATION_FACTOR


def monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """Pivot table: rows = years, columns = months (Jan-Dec), values = returns.
    Includes a YTD column.
    """
    df = returns.to_frame("return")
    df["year"] = df.index.year
    df["month"] = df.index.month

    pivot = df.pivot_table(values="return", index="year", columns="month", aggfunc="first")
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    pivot = pivot.rename(columns=month_names)

    # YTD = compound return for available months in each year
    ytd = {}
    for year in pivot.index:
        year_rets = pivot.loc[year].dropna()
        ytd[year] = (1 + year_rets).prod() - 1
    pivot["YTD"] = pd.Series(ytd)

    return pivot


def calendar_year_returns(returns: pd.Series) -> pd.Series:
    """Compound returns for each calendar year."""
    return returns.groupby(returns.index.year).apply(lambda x: (1 + x).prod() - 1)


def annualized_performance_table(returns: pd.Series,
                                 benchmarks: pd.DataFrame | None = None,
                                 factor: int = ANNUALIZATION_FACTOR) -> pd.DataFrame:
    """Annualized returns for 1yr, 3yr, 5yr, since inception."""
    periods = {"1Y": 12, "3Y": 36, "5Y": 60, "Since Inception": len(returns)}
    rows = {}

    for label, months in periods.items():
        if len(returns) < months and label != "Since Inception":
            continue
        chunk = returns.iloc[-months:]
        total = (1 + chunk).prod()
        ann = total ** (factor / len(chunk)) - 1 if total > 0 else 0.0
        rows[label] = {"Fund": ann}

        if benchmarks is not None:
            for col in benchmarks.columns:
                b_chunk = benchmarks[col].iloc[-months:]
                b_total = (1 + b_chunk).prod()
                b_ann = b_total ** (factor / len(b_chunk)) - 1 if b_total > 0 else 0.0
                rows[label][col] = b_ann

    return pd.DataFrame(rows).T


def up_down_capture(fund_returns: pd.Series,
                    benchmark_returns: pd.Series) -> dict:
    """Up capture and down capture ratios.

    Up capture = mean(fund | benchmark > 0) / mean(benchmark | benchmark > 0)
    Down capture = mean(fund | benchmark < 0) / mean(benchmark | benchmark < 0)
    """
    up_mask = benchmark_returns > 0
    down_mask = benchmark_returns < 0

    up_cap = (fund_returns[up_mask].mean() / benchmark_returns[up_mask].mean()
              if up_mask.sum() > 0 else 0.0)
    down_cap = (fund_returns[down_mask].mean() / benchmark_returns[down_mask].mean()
                if down_mask.sum() > 0 else 0.0)

    return {
        "Up Capture": up_cap,
        "Down Capture": down_cap,
        "Up months": int(up_mask.sum()),
        "Down months": int(down_mask.sum()),
    }


def best_worst_periods(returns: pd.Series) -> dict:
    """Best and worst month / quarter statistics."""
    from analytics.absolute_risk import quarterly_returns
    qtr = quarterly_returns(returns)
    return {
        "Best Month": returns.max(),
        "Best Month Date": returns.idxmax().strftime("%b %Y") if len(returns) > 0 else "",
        "Worst Month": returns.min(),
        "Worst Month Date": returns.idxmin().strftime("%b %Y") if len(returns) > 0 else "",
        "Best Quarter": qtr.max() if len(qtr) > 0 else 0.0,
        "Worst Quarter": qtr.min() if len(qtr) > 0 else 0.0,
        "Positive Months": int((returns > 0).sum()),
        "Negative Months": int((returns < 0).sum()),
        "Total Months": len(returns),
    }
