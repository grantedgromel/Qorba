"""Risk-return scatter plot.

Plots annualized return vs annualized volatility for multiple
funds/benchmarks, with each point labeled.
"""

import plotly.graph_objects as go

from config.settings import COLORS
from charts.theme import get_layout, get_color_sequence


def risk_return_scatter(metrics_dict: dict) -> go.Figure:
    """Risk/return scatter plot.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary of {name: {"return": float, "vol": float}}.
        Both values should be annualized decimals (e.g. 0.08 for 8%).

    Returns
    -------
    go.Figure
    """
    colors = get_color_sequence()

    names = list(metrics_dict.keys())
    rets = [metrics_dict[n]["return"] * 100 for n in names]
    vols = [metrics_dict[n]["vol"] * 100 for n in names]

    fig = go.Figure()

    for i, name in enumerate(names):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=[vols[i]],
            y=[rets[i]],
            mode="markers+text",
            name=name,
            text=[name],
            textposition="top center",
            textfont=dict(size=11, color=COLORS["navy_900"], weight=600),
            marker=dict(
                size=14,
                color=color,
                line=dict(width=2, color=COLORS["navy_900"]),
            ),
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Volatility: %{x:.1f}%<br>"
                "Return: %{y:.1f}%<extra></extra>"
            ),
        ))

    fig.update_layout(
        **get_layout(
            title={"text": "Risk-Return Scatter"},
            xaxis_title="Annualized Volatility (%)",
            yaxis_title="Annualized Return (%)",
            hovermode="closest",
            showlegend=False,
        )
    )

    return fig
