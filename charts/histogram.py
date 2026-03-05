"""Monthly return distribution histogram.

Overlays fund vs benchmark distributions with semi-transparent bars
in 1% buckets for clean visual comparison.
"""

import pandas as pd
import plotly.graph_objects as go

from charts.theme import get_layout, get_color_sequence


def return_histogram(returns: pd.Series,
                     benchmark_returns: pd.Series | None = None) -> go.Figure:
    """Monthly return distribution histogram in 1-percent buckets.

    Parameters
    ----------
    returns : pd.Series
        Monthly decimal returns for the fund.
    benchmark_returns : pd.Series, optional
        Monthly decimal returns for a benchmark to overlay.

    Returns
    -------
    go.Figure
    """
    colors = get_color_sequence()

    # Convert to percentage for display
    fund_pct = returns * 100

    fig = go.Figure()

    # Fund histogram
    fig.add_trace(go.Histogram(
        x=fund_pct,
        xbins=dict(size=1),
        name="Fund",
        marker_color=colors[0],
        opacity=0.70,
        hovertemplate="Return: %{x:.0f}%<br>Count: %{y}<extra></extra>",
    ))

    # Benchmark overlay
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        bench_pct = benchmark_returns * 100
        fig.add_trace(go.Histogram(
            x=bench_pct,
            xbins=dict(size=1),
            name="Benchmark",
            marker_color=colors[1],
            opacity=0.50,
            hovertemplate="Return: %{x:.0f}%<br>Count: %{y}<extra></extra>",
        ))

    fig.update_layout(
        **get_layout(
            title={"text": "Monthly Return Distribution"},
            xaxis_title="Monthly Return (%)",
            yaxis_title="Frequency",
            barmode="overlay",
            bargap=0.05,
            hovermode="x",
        )
    )

    # Add vertical line at zero
    fig.add_vline(
        x=0, line_width=1.5, line_dash="dash",
        line_color="#425466", opacity=0.6,
    )

    return fig
