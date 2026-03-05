"""Absolute return measures from Travers Chapter 6.

All functions take pd.Series of monthly decimal returns.
"""

import numpy as np
import pandas as pd
from config.settings import ANNUALIZATION_FACTOR, STERLING_PENALTY, CALMAR_MONTHS


def annualized_return_geometric(returns: pd.Series, factor: int = ANNUALIZATION_FACTOR) -> float:
    """Compound (geometric) annualized return."""
    total = (1 + returns).prod()
    n = len(returns)
    if n == 0 or total <= 0:
        return 0.0
    return total ** (factor / n) - 1


def annualized_return_arithmetic(returns: pd.Series, factor: int = ANNUALIZATION_FACTOR) -> float:
    """Arithmetic annualized return (used by Sharpe per Travers)."""
    return returns.mean() * factor


def annualized_volatility(returns: pd.Series, factor: int = ANNUALIZATION_FACTOR) -> float:
    """Annualized standard deviation."""
    return returns.std(ddof=1) * np.sqrt(factor)


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0,
                 factor: int = ANNUALIZATION_FACTOR) -> float:
    """Sharpe = (r - rf) / v. Book uses arithmetic annualized mean."""
    vol = annualized_volatility(returns, factor)
    if vol == 0:
        return 0.0
    ann_ret = annualized_return_arithmetic(returns, factor)
    return (ann_ret - rf_annual) / vol


def m2_ratio(returns: pd.Series, benchmark_returns: pd.Series,
             rf_annual: float = 0.0, factor: int = ANNUALIZATION_FACTOR) -> float:
    """M2 = Sharpe × benchmark_vol + rf."""
    sr = sharpe_ratio(returns, rf_annual, factor)
    bench_vol = annualized_volatility(benchmark_returns, factor)
    return sr * bench_vol + rf_annual


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series,
                      factor: int = ANNUALIZATION_FACTOR) -> float:
    """Information Ratio = Premium / Tracking Error."""
    excess = returns - benchmark_returns
    tracking_error = excess.std(ddof=1) * np.sqrt(factor)
    if tracking_error == 0:
        return 0.0
    premium = annualized_return_arithmetic(returns, factor) - annualized_return_arithmetic(benchmark_returns, factor)
    return premium / tracking_error


def _max_drawdown(returns: pd.Series) -> float:
    """Compute maximum drawdown (negative number)."""
    wealth = (1 + returns).cumprod()
    running_max = wealth.cummax()
    drawdown = (wealth - running_max) / running_max
    return drawdown.min()


def mar_ratio(returns: pd.Series, factor: int = ANNUALIZATION_FACTOR) -> float:
    """MAR = annualized_return / |max_drawdown| (since inception)."""
    mdd = _max_drawdown(returns)
    if mdd == 0:
        return 0.0
    ann_ret = annualized_return_geometric(returns, factor)
    return ann_ret / abs(mdd)


def calmar_ratio(returns: pd.Series, months: int = CALMAR_MONTHS,
                 factor: int = ANNUALIZATION_FACTOR) -> float | None:
    """Calmar = annualized_return (3yr) / |max_drawdown (3yr)|."""
    if len(returns) < months:
        return None
    recent = returns.iloc[-months:]
    mdd = _max_drawdown(recent)
    if mdd == 0:
        return 0.0
    ann_ret = annualized_return_geometric(recent, factor)
    return ann_ret / abs(mdd)


def sterling_ratio(returns: pd.Series, penalty: float = STERLING_PENALTY,
                   factor: int = ANNUALIZATION_FACTOR) -> float:
    """Sterling = annualized_return / (|max_drawdown| + 10%)."""
    mdd = _max_drawdown(returns)
    denom = abs(mdd) + penalty
    if denom == 0:
        return 0.0
    ann_ret = annualized_return_geometric(returns, factor)
    return ann_ret / denom


def sortino_ratio(returns: pd.Series, mar_annual: float = 0.0,
                  factor: int = ANNUALIZATION_FACTOR) -> float:
    """Sortino = (annualized_return - MAR) / downside_deviation.

    Per Travers: downside deviation uses only returns below MAR,
    n = count of returns below MAR.
    """
    mar_monthly = mar_annual / factor
    below = returns[returns < mar_monthly]
    if len(below) == 0:
        return float("inf") if returns.mean() > mar_monthly else 0.0
    dd = np.sqrt(((below - mar_monthly) ** 2).sum() / len(below)) * np.sqrt(factor)
    if dd == 0:
        return 0.0
    ann_ret = annualized_return_geometric(returns, factor)
    return (ann_ret - mar_annual) / dd


def omega_ratio(returns: pd.Series, threshold_monthly: float = 0.0) -> float:
    """Omega = sum(r_i - L for r_i > L) / sum(L - r_i for r_i < L).

    Discrete approximation of the probability-weighted gains/losses ratio.
    """
    gains = returns[returns > threshold_monthly] - threshold_monthly
    losses = threshold_monthly - returns[returns < threshold_monthly]
    if losses.sum() == 0:
        return float("inf")
    return gains.sum() / losses.sum()


def compute_all_return_metrics(returns: pd.Series, rf_annual: float = 0.0,
                               mar_annual: float = 0.0,
                               omega_threshold: float = 0.0,
                               benchmark_returns: pd.Series | None = None,
                               factor: int = ANNUALIZATION_FACTOR) -> dict:
    """Compute all absolute return metrics in one call."""
    result = {
        "Annualized Return (Geometric)": annualized_return_geometric(returns, factor),
        "Annualized Return (Arithmetic)": annualized_return_arithmetic(returns, factor),
        "Annualized Volatility": annualized_volatility(returns, factor),
        "Sharpe Ratio": sharpe_ratio(returns, rf_annual, factor),
        "MAR Ratio": mar_ratio(returns, factor),
        "Calmar Ratio": calmar_ratio(returns, CALMAR_MONTHS, factor),
        "Sterling Ratio": sterling_ratio(returns, STERLING_PENALTY, factor),
        "Sortino Ratio": sortino_ratio(returns, mar_annual, factor),
        "Omega Ratio": omega_ratio(returns, omega_threshold / factor if omega_threshold else 0.0),
    }

    if benchmark_returns is not None and not benchmark_returns.empty:
        result["M2 Ratio"] = m2_ratio(returns, benchmark_returns, rf_annual, factor)
        result["Information Ratio"] = information_ratio(returns, benchmark_returns, factor)

    return result
