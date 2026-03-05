"""Rolling metrics — 15 time-series views to avoid endpoint sensitivity.

Each function returns a pd.Series indexed by date with the rolling metric value.
"""

import numpy as np
import pandas as pd
from config.settings import ANNUALIZATION_FACTOR


def _rolling_apply(returns: pd.Series, window: int, func, **kwargs) -> pd.Series:
    """Generic rolling window application. Returns NaN for insufficient data."""
    results = {}
    for i in range(window, len(returns) + 1):
        chunk = returns.iloc[i - window:i]
        results[returns.index[i - 1]] = func(chunk, **kwargs)
    return pd.Series(results, dtype=float)


# ── Individual Rolling Metric Functions ────────────────────────────────────

def rolling_annualized_return(returns: pd.Series, window: int = 12,
                              factor: int = ANNUALIZATION_FACTOR) -> pd.Series:
    """Rolling compound annualized return."""
    def _calc(chunk, factor=factor):
        total = (1 + chunk).prod()
        if total <= 0:
            return 0.0
        return total ** (factor / len(chunk)) - 1
    return _rolling_apply(returns, window, _calc)


def rolling_annualized_volatility(returns: pd.Series, window: int = 12,
                                  factor: int = ANNUALIZATION_FACTOR) -> pd.Series:
    """Rolling annualized standard deviation."""
    def _calc(chunk, factor=factor):
        return chunk.std(ddof=1) * np.sqrt(factor)
    return _rolling_apply(returns, window, _calc)


def rolling_sharpe(returns: pd.Series, window: int = 12,
                   rf_annual: float = 0.0,
                   factor: int = ANNUALIZATION_FACTOR) -> pd.Series:
    """Rolling Sharpe ratio (arithmetic mean)."""
    def _calc(chunk, rf_annual=rf_annual, factor=factor):
        vol = chunk.std(ddof=1) * np.sqrt(factor)
        if vol == 0:
            return 0.0
        ann_ret = chunk.mean() * factor
        return (ann_ret - rf_annual) / vol
    return _rolling_apply(returns, window, _calc)


def rolling_sortino(returns: pd.Series, window: int = 12,
                    mar_annual: float = 0.0,
                    factor: int = ANNUALIZATION_FACTOR) -> pd.Series:
    """Rolling Sortino ratio."""
    def _calc(chunk, mar_annual=mar_annual, factor=factor):
        mar_m = mar_annual / factor
        below = chunk[chunk < mar_m]
        if len(below) == 0:
            return float("nan")
        dd = np.sqrt(((below - mar_m) ** 2).sum() / len(below)) * np.sqrt(factor)
        if dd == 0:
            return 0.0
        ann_ret = (1 + chunk).prod() ** (factor / len(chunk)) - 1
        return (ann_ret - mar_annual) / dd
    return _rolling_apply(returns, window, _calc)


def rolling_beta(fund_returns: pd.Series, benchmark_returns: pd.Series,
                 window: int = 12) -> pd.Series:
    """Rolling beta relative to a benchmark."""
    results = {}
    for i in range(window, len(fund_returns) + 1):
        f = fund_returns.iloc[i - window:i]
        b = benchmark_returns.iloc[i - window:i]
        cov = np.cov(f, b, ddof=1)
        var_b = cov[1, 1]
        if var_b == 0:
            results[fund_returns.index[i - 1]] = 0.0
        else:
            results[fund_returns.index[i - 1]] = cov[0, 1] / var_b
    return pd.Series(results, dtype=float)


def rolling_alpha(fund_returns: pd.Series, benchmark_returns: pd.Series,
                  window: int = 12,
                  factor: int = ANNUALIZATION_FACTOR) -> pd.Series:
    """Rolling alpha (annualized)."""
    results = {}
    for i in range(window, len(fund_returns) + 1):
        f = fund_returns.iloc[i - window:i]
        b = benchmark_returns.iloc[i - window:i]
        cov = np.cov(f, b, ddof=1)
        var_b = cov[1, 1]
        bt = cov[0, 1] / var_b if var_b != 0 else 0.0
        a = (f.mean() - bt * b.mean()) * factor
        results[fund_returns.index[i - 1]] = a
    return pd.Series(results, dtype=float)


def rolling_correlation(fund_returns: pd.Series, benchmark_returns: pd.Series,
                        window: int = 12) -> pd.Series:
    """Rolling Pearson correlation."""
    results = {}
    for i in range(window, len(fund_returns) + 1):
        f = fund_returns.iloc[i - window:i]
        b = benchmark_returns.iloc[i - window:i]
        r = np.corrcoef(f, b)[0, 1]
        results[fund_returns.index[i - 1]] = r
    return pd.Series(results, dtype=float)


def rolling_information_ratio(fund_returns: pd.Series, benchmark_returns: pd.Series,
                              window: int = 12,
                              factor: int = ANNUALIZATION_FACTOR) -> pd.Series:
    """Rolling information ratio."""
    results = {}
    for i in range(window, len(fund_returns) + 1):
        f = fund_returns.iloc[i - window:i]
        b = benchmark_returns.iloc[i - window:i]
        excess = f - b
        te = excess.std(ddof=1) * np.sqrt(factor)
        if te == 0:
            results[fund_returns.index[i - 1]] = 0.0
        else:
            prem = (f.mean() - b.mean()) * factor
            results[fund_returns.index[i - 1]] = prem / te
    return pd.Series(results, dtype=float)


def rolling_max_drawdown(returns: pd.Series, window: int = 12) -> pd.Series:
    """Rolling maximum drawdown (worst peak-to-trough in trailing window)."""
    def _calc(chunk):
        wealth = (1 + chunk).cumprod()
        running_max = wealth.cummax()
        dd = (wealth - running_max) / running_max
        return dd.min()
    return _rolling_apply(returns, window, _calc)


def rolling_downside_deviation(returns: pd.Series, window: int = 12,
                               mar_annual: float = 0.0,
                               factor: int = ANNUALIZATION_FACTOR) -> pd.Series:
    """Rolling downside deviation."""
    def _calc(chunk, mar_annual=mar_annual, factor=factor):
        mar_m = mar_annual / factor
        below = chunk[chunk < mar_m]
        if len(below) == 0:
            return 0.0
        return np.sqrt(((below - mar_m) ** 2).sum() / len(below)) * np.sqrt(factor)
    return _rolling_apply(returns, window, _calc)


def rolling_skewness(returns: pd.Series, window: int = 12) -> pd.Series:
    """Rolling skewness."""
    def _calc(chunk):
        n = len(chunk)
        if n < 3:
            return 0.0
        avg = chunk.mean()
        std = chunk.std(ddof=0)
        if std == 0:
            return 0.0
        return ((((chunk - avg) / std) ** 3).sum()) / n
    return _rolling_apply(returns, window, _calc)


def rolling_kurtosis(returns: pd.Series, window: int = 12) -> pd.Series:
    """Rolling kurtosis."""
    def _calc(chunk):
        n = len(chunk)
        if n < 4:
            return 0.0
        avg = chunk.mean()
        std = chunk.std(ddof=0)
        if std == 0:
            return 0.0
        return ((((chunk - avg) / std) ** 4).sum()) / n
    return _rolling_apply(returns, window, _calc)


def rolling_up_capture(fund_returns: pd.Series, benchmark_returns: pd.Series,
                       window: int = 12) -> pd.Series:
    """Rolling up-capture ratio."""
    results = {}
    for i in range(window, len(fund_returns) + 1):
        f = fund_returns.iloc[i - window:i]
        b = benchmark_returns.iloc[i - window:i]
        up_mask = b > 0
        if up_mask.sum() == 0:
            results[fund_returns.index[i - 1]] = float("nan")
        else:
            results[fund_returns.index[i - 1]] = f[up_mask].mean() / b[up_mask].mean()
    return pd.Series(results, dtype=float)


def rolling_down_capture(fund_returns: pd.Series, benchmark_returns: pd.Series,
                         window: int = 12) -> pd.Series:
    """Rolling down-capture ratio."""
    results = {}
    for i in range(window, len(fund_returns) + 1):
        f = fund_returns.iloc[i - window:i]
        b = benchmark_returns.iloc[i - window:i]
        down_mask = b < 0
        if down_mask.sum() == 0:
            results[fund_returns.index[i - 1]] = float("nan")
        else:
            results[fund_returns.index[i - 1]] = f[down_mask].mean() / b[down_mask].mean()
    return pd.Series(results, dtype=float)


def rolling_win_rate(returns: pd.Series, window: int = 12) -> pd.Series:
    """Rolling percentage of positive months."""
    def _calc(chunk):
        return (chunk > 0).sum() / len(chunk)
    return _rolling_apply(returns, window, _calc)


def rolling_gain_loss_ratio(returns: pd.Series, window: int = 12) -> pd.Series:
    """Rolling gain/loss ratio."""
    def _calc(chunk):
        gains = chunk[chunk > 0]
        losses = chunk[chunk < 0]
        if len(losses) == 0 or losses.mean() == 0:
            return float("nan")
        return gains.mean() / abs(losses.mean())
    return _rolling_apply(returns, window, _calc)
