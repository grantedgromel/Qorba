"""Cumulative wealth chart (growth of $100).

Fund shown as a thick solid line; benchmarks as thinner dashed lines.
"""

import pandas as pd
import plotly.graph_objects as go

from charts.theme import get_layout, get_color_sequence


def cumulative_chart(fund_returns: pd.Series,
                     benchmark_df: pd.DataFrame | None = None) -> go.Figure:
    """Cumulative wealth chart starting at 100.

    Parameters
    ----------
    fund_returns : pd.Series
        Monthly decimal returns for the fund.
    benchmark_df : pd.DataFrame, optional
        DataFrame where each column is a benchmark's monthly returns.
        Columns are used as benchmark names.

    Returns
    -------
    go.Figure
    """
    colors = get_color_sequence()

    fig = go.Figure()

    # Fund cumulative wealth
    fund_wealth = (1 + fund_returns).cumprod() * 100

    fig.add_trace(go.Scatter(
        x=fund_wealth.index,
        y=fund_wealth.values,
        mode="lines",
        name="Fund",
        line=dict(color=colors[0], width=3),
        hovertemplate="%{x|%b %Y}<br>Value: %{y:.1f}<extra>Fund</extra>",
    ))

    # Benchmark lines
    if benchmark_df is not None and not benchmark_df.empty:
        dash_styles = ["dash", "dot", "dashdot", "longdash", "longdashdot"]
        for i, col in enumerate(benchmark_df.columns):
            bm_returns = benchmark_df[col].dropna()
            bm_wealth = (1 + bm_returns).cumprod() * 100
            color_idx = (i + 1) % len(colors)
            dash_idx = i % len(dash_styles)

            fig.add_trace(go.Scatter(
                x=bm_wealth.index,
                y=bm_wealth.values,
                mode="lines",
                name=col,
                line=dict(
                    color=colors[color_idx],
                    width=1.8,
                    dash=dash_styles[dash_idx],
                ),
                hovertemplate="%{x|%b %Y}<br>Value: %{y:.1f}<extra>" + col + "</extra>",
            ))

    # Starting reference line at 100
    fig.add_hline(
        y=100, line_width=1, line_dash="dot",
        line_color="#425466", opacity=0.4,
    )

    fig.update_layout(
        **get_layout(
            title={"text": "Cumulative Wealth (Growth of 100)"},
            xaxis_title="",
            yaxis_title="Value",
            hovermode="x unified",
        )
    )

    return fig
