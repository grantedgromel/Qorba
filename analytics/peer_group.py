"""Peer group analysis from Travers Chapter 6."""

import numpy as np
import pandas as pd
from config.settings import ANNUALIZATION_FACTOR


def percentile_ranking(fund_return: float, peer_returns: pd.Series) -> float:
    """Rank fund among peers. 1 = best, 100 = worst (per Travers convention)."""
    if len(peer_returns) == 0:
        return 50.0
    all_returns = pd.concat([peer_returns, pd.Series([fund_return], index=["fund"])])
    ranked = all_returns.rank(ascending=False, method="min")
    pct = ranked["fund"] / len(all_returns) * 100
    return pct


def _annualized_return_for_period(returns: pd.Series, months: int,
                                  factor: int = ANNUALIZATION_FACTOR) -> float:
    """Compute annualized return for trailing N months."""
    if len(returns) < months:
        return float("nan")
    chunk = returns.iloc[-months:]
    total = (1 + chunk).prod()
    if total <= 0:
        return 0.0
    return total ** (factor / months) - 1


def quartile_distribution(fund_returns: pd.Series, peer_returns_df: pd.DataFrame,
                          periods: list[int] | None = None,
                          factor: int = ANNUALIZATION_FACTOR) -> pd.DataFrame:
    """Compute quartile boundaries and fund position for each period.

    Returns DataFrame with rows per period and columns:
    5th, 25th, 50th, 75th, 95th percentiles + Fund value + Fund percentile.
    """
    if periods is None:
        periods = [12, 24, 36, 48, 60]

    rows = []
    for months in periods:
        if len(fund_returns) < months:
            continue

        fund_ret = _annualized_return_for_period(fund_returns, months, factor)

        peer_rets = {}
        for col in peer_returns_df.columns:
            peer_rets[col] = _annualized_return_for_period(
                peer_returns_df[col].dropna(), months, factor
            )

        peer_series = pd.Series(peer_rets).dropna()
        if len(peer_series) == 0:
            continue

        pctiles = np.percentile(peer_series, [5, 25, 50, 75, 95])
        pct_rank = percentile_ranking(fund_ret, peer_series)

        label = f"{months // 12}Y" if months >= 12 else f"{months}M"
        rows.append({
            "Period": label,
            "5th": pctiles[0],
            "25th (Q1)": pctiles[1],
            "Median": pctiles[2],
            "75th (Q3)": pctiles[3],
            "95th": pctiles[4],
            "Fund": fund_ret,
            "Percentile": pct_rank,
        })

    return pd.DataFrame(rows)


def rolling_percentile(fund_returns: pd.Series, peer_returns_df: pd.DataFrame,
                       window_months: int = 12,
                       factor: int = ANNUALIZATION_FACTOR) -> pd.Series:
    """Rolling percentile ranking over time."""
    results = {}
    for i in range(window_months, len(fund_returns) + 1):
        fund_chunk = fund_returns.iloc[i - window_months:i]
        fund_ret = (1 + fund_chunk).prod() ** (factor / window_months) - 1

        peer_rets = {}
        for col in peer_returns_df.columns:
            peer_chunk = peer_returns_df[col].iloc[i - window_months:i].dropna()
            if len(peer_chunk) == window_months:
                peer_rets[col] = (1 + peer_chunk).prod() ** (factor / window_months) - 1

        if len(peer_rets) > 0:
            pct = percentile_ranking(fund_ret, pd.Series(peer_rets))
            results[fund_returns.index[i - 1]] = pct
        else:
            results[fund_returns.index[i - 1]] = float("nan")

    return pd.Series(results, dtype=float)
