"""Regression scatter plot with line of best fit.

Shows fund vs benchmark monthly returns with OLS regression line
and alpha/beta annotation.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config.settings import COLORS
from charts.theme import get_layout, get_color_sequence


def regression_scatter(fund_returns: pd.Series,
                       benchmark_returns: pd.Series,
                       name: str = "Benchmark") -> go.Figure:
    """Scatter plot of fund vs benchmark returns with regression line.

    Parameters
    ----------
    fund_returns : pd.Series
        Monthly decimal returns for the fund.
    benchmark_returns : pd.Series
        Monthly decimal returns for the benchmark.
    name : str
        Label for the benchmark axis.

    Returns
    -------
    go.Figure
    """
    colors = get_color_sequence()

    # Align series
    aligned = pd.concat([fund_returns, benchmark_returns], axis=1).dropna()
    fund_vals = aligned.iloc[:, 0].values * 100
    bench_vals = aligned.iloc[:, 1].values * 100

    # OLS fit: fund = alpha + beta * benchmark
    coeffs = np.polyfit(bench_vals, fund_vals, 1)
    beta = coeffs[0]
    alpha = coeffs[1]

    # Correlation / R-squared
    r = np.corrcoef(bench_vals, fund_vals)[0, 1]
    r_sq = r ** 2

    # Regression line x range
    x_min = bench_vals.min() - 1
    x_max = bench_vals.max() + 1
    x_line = np.array([x_min, x_max])
    y_line = alpha + beta * x_line

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=bench_vals,
        y=fund_vals,
        mode="markers",
        name="Monthly Returns",
        marker=dict(
            size=7,
            color=colors[0],
            opacity=0.65,
            line=dict(width=1, color=COLORS["navy_900"]),
        ),
        hovertemplate=(
            f"{name}: " + "%{x:.2f}%<br>"
            "Fund: %{y:.2f}%<extra></extra>"
        ),
    ))

    # Best-fit line
    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode="lines",
        name="Best Fit",
        line=dict(color=COLORS["red"], width=2, dash="dash"),
        hoverinfo="skip",
    ))

    # Annotation with alpha, beta, R-squared
    annotation_text = (
        f"<b>Alpha:</b> {alpha:.2f}% / mo"
        f"<br><b>Beta:</b> {beta:.2f}"
        f"<br><b>R\u00b2:</b> {r_sq:.3f}"
    )

    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text=annotation_text,
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor=COLORS["navy_900"],
        borderwidth=1.5,
        borderpad=8,
        font=dict(
            family="Inter, sans-serif",
            size=12,
            color=COLORS["navy_900"],
        ),
    )

    fig.update_layout(
        **get_layout(
            title={"text": f"Regression: Fund vs {name}"},
            xaxis_title=f"{name} Return (%)",
            yaxis_title="Fund Return (%)",
            hovermode="closest",
        )
    )

    return fig
