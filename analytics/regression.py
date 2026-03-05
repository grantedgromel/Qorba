"""Regression-based statistics from Travers Chapter 6."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from config.settings import ANNUALIZATION_FACTOR
from analytics.absolute_return import annualized_return_arithmetic, annualized_volatility


def compute_regression(fund_returns: pd.Series,
                       benchmark_returns: pd.Series) -> dict:
    """Full OLS regression: fund = alpha + beta * benchmark + epsilon.

    Returns dict with alpha, beta, t_stat, r, r_squared, beta_std_error, residuals.
    Uses statsmodels for proper standard errors.
    """
    y = fund_returns.values
    x = sm.add_constant(benchmark_returns.values)

    model = sm.OLS(y, x).fit()

    alpha = model.params[0]
    beta = model.params[1]
    t_stat = model.tvalues[1]
    beta_se = model.bse[1]
    r_squared = model.rsquared

    # Correlation
    r = np.corrcoef(fund_returns.values, benchmark_returns.values)[0, 1]

    return {
        "alpha": alpha,
        "beta": beta,
        "t_stat": t_stat,
        "r": r,
        "r_squared": r_squared,
        "beta_std_error": beta_se,
        "residuals": model.resid,
    }


def beta(fund_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Beta = Correlation * (std_fund / std_benchmark). Per Travers."""
    r = np.corrcoef(fund_returns, benchmark_returns)[0, 1]
    std_f = fund_returns.std(ddof=1)
    std_b = benchmark_returns.std(ddof=1)
    if std_b == 0:
        return 0.0
    return r * (std_f / std_b)


def alpha(fund_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Alpha = avg_fund - Beta * avg_benchmark. Per Travers."""
    b = beta(fund_returns, benchmark_returns)
    return fund_returns.mean() - b * benchmark_returns.mean()


def t_stat(fund_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """T-Stat = Beta / Beta_Standard_Error."""
    result = compute_regression(fund_returns, benchmark_returns)
    return result["t_stat"]


def correlation(fund_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Pearson correlation coefficient."""
    return np.corrcoef(fund_returns, benchmark_returns)[0, 1]


def r_squared(fund_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """R-squared = correlation^2."""
    r = correlation(fund_returns, benchmark_returns)
    return r ** 2


def treynor_ratio(fund_returns: pd.Series, benchmark_returns: pd.Series,
                  rf_annual: float = 0.0,
                  factor: int = ANNUALIZATION_FACTOR) -> float:
    """Treynor = (annualized_return - rf) / Beta. Per Travers."""
    b = beta(fund_returns, benchmark_returns)
    if b == 0:
        return 0.0
    ann_ret = annualized_return_arithmetic(fund_returns, factor)
    return (ann_ret - rf_annual) / b


def compute_all_regression_metrics(fund_returns: pd.Series,
                                   benchmark_returns: pd.Series,
                                   rf_annual: float = 0.0,
                                   factor: int = ANNUALIZATION_FACTOR) -> dict:
    """Compute all regression-based metrics for one benchmark."""
    reg = compute_regression(fund_returns, benchmark_returns)
    return {
        "Alpha (monthly)": reg["alpha"],
        "Alpha (annualized)": reg["alpha"] * factor,
        "Beta": reg["beta"],
        "T-Stat": reg["t_stat"],
        "Correlation (R)": reg["r"],
        "R-Squared": reg["r_squared"],
        "Beta Std Error": reg["beta_std_error"],
        "Treynor Ratio": treynor_ratio(fund_returns, benchmark_returns, rf_annual, factor),
    }


def correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix for all columns in a DataFrame."""
    return returns_df.corr()
