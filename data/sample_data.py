"""Embedded sample dataset for demo mode — synthetic hedge fund returns."""

import pandas as pd
import numpy as np


def generate_sample_data() -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Generate synthetic fund, benchmark, and peer return data.

    Returns (fund_returns, benchmark_returns, peer_returns).
    """
    rng = np.random.RandomState(42)
    dates = pd.date_range("2017-01-31", periods=60, freq="ME")

    # Market factor
    market = rng.normal(0.008, 0.045, 60)

    # Fund: low-beta, positive alpha, slight positive skew
    fund = 0.005 + 0.35 * market + rng.normal(0, 0.015, 60)
    fund = pd.Series(fund, index=dates, name="fund")

    # Benchmarks
    sp500 = market + rng.normal(0, 0.005, 60)
    russell = 1.15 * market + rng.normal(0.001, 0.008, 60)
    hfri = 0.6 * market + rng.normal(0.002, 0.012, 60)

    benchmarks = pd.DataFrame({
        "S&P 500": sp500,
        "Russell 2000": russell,
        "HFRI Equity Hedge": hfri,
    }, index=dates)

    # Peer funds
    peers = pd.DataFrame(index=dates)
    for i in range(8):
        beta = rng.uniform(0.2, 0.9)
        alpha = rng.normal(0.002, 0.003)
        noise = rng.normal(0, rng.uniform(0.01, 0.03), 60)
        peers[f"Peer {i+1}"] = alpha + beta * market + noise

    return fund, benchmarks, peers
