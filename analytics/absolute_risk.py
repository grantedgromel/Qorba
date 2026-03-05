"""Absolute risk measures from Travers Chapter 6.

All functions take pd.Series of monthly decimal returns.
"""

import numpy as np
import pandas as pd
from config.settings import ANNUALIZATION_FACTOR


# ── Standard Deviation Variants ────────────────────────────────────────────

def standard_deviation(returns: pd.Series, factor: int = ANNUALIZATION_FACTOR) -> float:
    """Annualized standard deviation."""
    return returns.std(ddof=1) * np.sqrt(factor)


def gain_std(returns: pd.Series, factor: int = ANNUALIZATION_FACTOR) -> float:
    """Standard deviation of positive returns, annualized."""
    pos = returns[returns > 0]
    if len(pos) < 2:
        return 0.0
    return pos.std(ddof=1) * np.sqrt(factor)


def loss_std(returns: pd.Series, factor: int = ANNUALIZATION_FACTOR) -> float:
    """Standard deviation of negative returns, annualized."""
    neg = returns[returns < 0]
    if len(neg) < 2:
        return 0.0
    return neg.std(ddof=1) * np.sqrt(factor)


def downside_deviation(returns: pd.Series, mar_annual: float = 0.0,
                       factor: int = ANNUALIZATION_FACTOR) -> float:
    """Downside deviation below MAR (per Travers: n = count below MAR)."""
    mar_monthly = mar_annual / factor
    below = returns[returns < mar_monthly]
    n = len(below)
    if n == 0:
        return 0.0
    return np.sqrt(((below - mar_monthly) ** 2).sum() / n) * np.sqrt(factor)


def semideviation(returns: pd.Series, factor: int = ANNUALIZATION_FACTOR) -> float:
    """Standard deviation of returns below the mean (per Travers)."""
    mean_ret = returns.mean()
    below = returns[returns < mean_ret]
    n = len(below)
    if n == 0:
        return 0.0
    return np.sqrt(((below - mean_ret) ** 2).sum() / n) * np.sqrt(factor)


# ── Distribution Shape ─────────────────────────────────────────────────────

def skewness(returns: pd.Series) -> float:
    """Skewness = (1/n) * Sum((r - avg) / std)^3. Per Travers formula."""
    n = len(returns)
    if n < 3:
        return 0.0
    avg = returns.mean()
    std = returns.std(ddof=0)
    if std == 0:
        return 0.0
    return ((((returns - avg) / std) ** 3).sum()) / n


def kurtosis(returns: pd.Series) -> float:
    """Kurtosis = (1/n) * Sum((r - avg) / std)^4. Per Travers formula."""
    n = len(returns)
    if n < 4:
        return 0.0
    avg = returns.mean()
    std = returns.std(ddof=0)
    if std == 0:
        return 0.0
    return ((((returns - avg) / std) ** 4).sum()) / n


def excess_kurtosis(returns: pd.Series) -> float:
    """Excess Kurtosis = Kurtosis - 3."""
    return kurtosis(returns) - 3


def quarterly_returns(returns: pd.Series) -> pd.Series:
    """Compound monthly returns into calendar quarters."""
    # Group by year-quarter
    grouped = returns.groupby(returns.index.to_period("Q"))
    quarterly = grouped.apply(lambda x: (1 + x).prod() - 1)
    quarterly.index = quarterly.index.to_timestamp()
    return quarterly


# ── Drawdown Analysis ──────────────────────────────────────────────────────

def drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute the running drawdown series from peak."""
    wealth = (1 + returns).cumprod()
    running_max = wealth.cummax()
    dd = (wealth - running_max) / running_max
    dd.name = "drawdown"
    return dd


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown (negative value)."""
    dd = drawdown_series(returns)
    return dd.min() if len(dd) > 0 else 0.0


def drawdown_table(returns: pd.Series, top_n: int = 10) -> pd.DataFrame:
    """Build a drawdown episodes table (matching Table 6.3 from Travers).

    Returns columns: Max Drawdown, Length (months), Recovery (months),
                    Peak Date, Trough Date
    """
    wealth = (1 + returns).cumprod()
    running_max = wealth.cummax()
    dd = (wealth - running_max) / running_max

    # Identify drawdown episodes
    in_drawdown = dd < 0
    episodes = []
    start = None

    for i, (dt, val) in enumerate(dd.items()):
        if val < 0 and start is None:
            start = i
        elif val >= 0 and start is not None:
            episodes.append((start, i - 1, i))
            start = None

    # Handle ongoing drawdown
    if start is not None:
        episodes.append((start, len(dd) - 1, None))

    if not episodes:
        return pd.DataFrame(columns=[
            "Max Drawdown", "Length (months)", "Recovery (months)",
            "Peak Date", "Trough Date"
        ])

    rows = []
    for ep_start, ep_trough_search_end, recovery_idx in episodes:
        # The peak is the date just before the drawdown starts
        peak_idx = max(0, ep_start - 1) if ep_start > 0 else 0
        peak_date = dd.index[peak_idx]

        # Find the trough within this episode
        episode_dd = dd.iloc[ep_start:ep_trough_search_end + 1]
        trough_pos = episode_dd.idxmin()
        trough_val = episode_dd.min()
        trough_date = trough_pos

        # Length = months from peak to trough
        trough_idx = dd.index.get_loc(trough_date)
        length = trough_idx - peak_idx

        # Recovery = months from trough to full recovery
        if recovery_idx is not None:
            recovery = recovery_idx - trough_idx
        else:
            recovery = None  # still in drawdown

        rows.append({
            "Max Drawdown": trough_val,
            "Length (months)": length,
            "Recovery (months)": recovery if recovery is not None else "Ongoing",
            "Peak Date": peak_date.strftime("%b-%y"),
            "Trough Date": trough_date.strftime("%b-%y"),
        })

    table = pd.DataFrame(rows)
    table = table.sort_values("Max Drawdown").head(top_n).reset_index(drop=True)
    return table


# ── Gain / Loss Ratio ──────────────────────────────────────────────────────

def gain_loss_ratio(returns: pd.Series) -> float:
    """Average gain / |average loss|."""
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    if len(losses) == 0 or losses.mean() == 0:
        return float("inf") if len(gains) > 0 else 0.0
    return gains.mean() / abs(losses.mean())


def avg_gain(returns: pd.Series) -> float:
    pos = returns[returns > 0]
    return pos.mean() if len(pos) > 0 else 0.0


def avg_loss(returns: pd.Series) -> float:
    neg = returns[returns < 0]
    return neg.mean() if len(neg) > 0 else 0.0


def win_rate(returns: pd.Series) -> float:
    """Percentage of positive months."""
    if len(returns) == 0:
        return 0.0
    return (returns > 0).sum() / len(returns)


# ── Summary ────────────────────────────────────────────────────────────────

def compute_all_risk_metrics(returns: pd.Series, mar_annual: float = 0.0,
                             factor: int = ANNUALIZATION_FACTOR) -> dict:
    """Compute all absolute risk metrics in one call."""
    qtr = quarterly_returns(returns)
    return {
        "Standard Deviation": standard_deviation(returns, factor),
        "Gain Std Dev": gain_std(returns, factor),
        "Loss Std Dev": loss_std(returns, factor),
        "Downside Deviation": downside_deviation(returns, mar_annual, factor),
        "Semideviation": semideviation(returns, factor),
        "Skewness (Monthly)": skewness(returns),
        "Skewness (Quarterly)": skewness(qtr),
        "Kurtosis (Monthly)": kurtosis(returns),
        "Kurtosis (Quarterly)": kurtosis(qtr),
        "Excess Kurtosis (Monthly)": excess_kurtosis(returns),
        "Excess Kurtosis (Quarterly)": excess_kurtosis(qtr),
        "Max Drawdown": max_drawdown(returns),
        "Gain/Loss Ratio": gain_loss_ratio(returns),
        "Average Gain": avg_gain(returns),
        "Average Loss": avg_loss(returns),
        "Win Rate": win_rate(returns),
    }
