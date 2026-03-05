"""Drawdown / underwater equity charts.

Two views:
1. drawdown_chart       - filled area below zero showing the drawdown series
2. drawdown_highlight_chart - cumulative wealth with top N drawdowns shaded
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config.settings import COLORS
from charts.theme import get_layout, get_color_sequence


# ── Helpers ───────────────────────────────────────────────────────────────

def _drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute drawdown series from monthly returns."""
    wealth = (1 + returns).cumprod()
    running_max = wealth.cummax()
    drawdown = (wealth - running_max) / running_max
    return drawdown


def _top_drawdowns(returns: pd.Series, top_n: int = 5) -> list[dict]:
    """Identify the top N drawdowns with start, trough, and recovery dates.

    Returns a list of dicts sorted by depth (worst first):
        {start, trough, end, depth, length_months}
    """
    dd = _drawdown_series(returns)
    drawdowns: list[dict] = []

    in_drawdown = False
    start = None

    for i in range(len(dd)):
        if dd.iloc[i] < 0 and not in_drawdown:
            # Drawdown begins (previous index was peak)
            start = dd.index[i - 1] if i > 0 else dd.index[i]
            in_drawdown = True
        elif dd.iloc[i] >= 0 and in_drawdown:
            # Drawdown recovered
            trough_idx = dd.iloc[dd.index.get_indexer([start])[0]:i + 1].idxmin()
            drawdowns.append({
                "start": start,
                "trough": trough_idx,
                "end": dd.index[i],
                "depth": dd.loc[trough_idx],
            })
            in_drawdown = False

    # Handle ongoing drawdown at end of series
    if in_drawdown and start is not None:
        trough_idx = dd.iloc[dd.index.get_indexer([start])[0]:].idxmin()
        drawdowns.append({
            "start": start,
            "trough": trough_idx,
            "end": dd.index[-1],
            "depth": dd.loc[trough_idx],
        })

    # Sort by depth (most negative first) and take top N
    drawdowns.sort(key=lambda d: d["depth"])
    return drawdowns[:top_n]


# ── Main Charts ───────────────────────────────────────────────────────────

def drawdown_chart(returns: pd.Series) -> go.Figure:
    """Underwater equity chart: drawdown series as filled red area below zero.

    Parameters
    ----------
    returns : pd.Series
        Monthly decimal returns.

    Returns
    -------
    go.Figure
    """
    dd = _drawdown_series(returns) * 100  # percent

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dd.index,
        y=dd.values,
        mode="lines",
        fill="tozeroy",
        name="Drawdown",
        line=dict(color=COLORS["red"], width=1.5),
        fillcolor="rgba(255, 92, 92, 0.35)",
        hovertemplate="%{x|%b %Y}<br>Drawdown: %{y:.1f}%<extra></extra>",
    ))

    # Zero line
    fig.add_hline(y=0, line_width=1.5, line_color=COLORS["navy_700"], opacity=0.5)

    fig.update_layout(
        **get_layout(
            title={"text": "Underwater Equity (Drawdown)"},
            xaxis_title="",
            yaxis_title="Drawdown (%)",
            hovermode="x unified",
            showlegend=False,
        )
    )

    return fig


def drawdown_highlight_chart(returns: pd.Series, top_n: int = 5) -> go.Figure:
    """Cumulative wealth chart with the top N drawdowns shaded.

    Parameters
    ----------
    returns : pd.Series
        Monthly decimal returns.
    top_n : int
        Number of worst drawdowns to highlight.

    Returns
    -------
    go.Figure
    """
    colors = get_color_sequence()
    wealth = (1 + returns).cumprod() * 100

    fig = go.Figure()

    # Shaded drawdown regions
    top_dds = _top_drawdowns(returns, top_n)
    shade_colors = [
        "rgba(255, 92, 92, 0.20)",
        "rgba(255, 159, 67, 0.18)",
        "rgba(255, 217, 61, 0.18)",
        "rgba(122, 115, 255, 0.15)",
        "rgba(66, 84, 102, 0.12)",
    ]

    for i, dd in enumerate(top_dds):
        shade = shade_colors[i % len(shade_colors)]
        fig.add_vrect(
            x0=dd["start"], x1=dd["end"],
            fillcolor=shade,
            line_width=0,
            annotation_text=f"#{i + 1} ({dd['depth']:.1%})",
            annotation_position="top left",
            annotation_font=dict(size=10, color=COLORS["navy_700"]),
        )

    # Wealth line
    fig.add_trace(go.Scatter(
        x=wealth.index,
        y=wealth.values,
        mode="lines",
        name="Fund",
        line=dict(color=colors[0], width=2.5),
        hovertemplate="%{x|%b %Y}<br>Value: %{y:.1f}<extra>Fund</extra>",
    ))

    fig.update_layout(
        **get_layout(
            title={"text": f"Cumulative Wealth with Top {top_n} Drawdowns"},
            xaxis_title="",
            yaxis_title="Value (100 base)",
            hovermode="x unified",
        )
    )

    return fig
