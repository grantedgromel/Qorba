"""Peer group charts.

1. quartile_box_chart       - box-like display of quartile distribution
                              with the fund's position marked
2. rolling_percentile_chart - line chart of rolling percentile over time
"""

import pandas as pd
import plotly.graph_objects as go

from config.settings import COLORS
from charts.theme import get_layout, get_color_sequence


def quartile_box_chart(quartile_df: pd.DataFrame) -> go.Figure:
    """Box-like chart showing peer quartile distribution with fund position.

    Expects a DataFrame from analytics.peer_group.quartile_distribution
    with columns: Period, 5th, 25th (Q1), Median, 75th (Q3), 95th,
    Fund, Percentile.

    Each period is rendered as a vertical range bar (5th-95th) with
    the interquartile range (Q1-Q3) highlighted, and the fund marked
    as a prominent dot.

    Parameters
    ----------
    quartile_df : pd.DataFrame
        Output of quartile_distribution().

    Returns
    -------
    go.Figure
    """
    colors = get_color_sequence()

    fig = go.Figure()

    periods = quartile_df["Period"].tolist()

    # 5th to 95th range (thin bar)
    fig.add_trace(go.Bar(
        x=periods,
        y=(quartile_df["95th"] - quartile_df["5th"]) * 100,
        base=quartile_df["5th"].values * 100,
        name="5th-95th",
        marker_color="rgba(66, 84, 102, 0.12)",
        marker_line=dict(width=1, color=COLORS["gray_200"]),
        width=0.6,
        hovertemplate=(
            "%{x}<br>"
            "5th: %{base:.1f}%<br>"
            "95th: %{customdata:.1f}%<extra></extra>"
        ),
        customdata=quartile_df["95th"].values * 100,
    ))

    # Q1 to Q3 range (darker bar)
    fig.add_trace(go.Bar(
        x=periods,
        y=(quartile_df["75th (Q3)"] - quartile_df["25th (Q1)"]) * 100,
        base=quartile_df["25th (Q1)"].values * 100,
        name="Q1-Q3",
        marker_color="rgba(99, 91, 255, 0.20)",
        marker_line=dict(width=1.5, color=colors[0]),
        width=0.6,
        hovertemplate=(
            "%{x}<br>"
            "Q1: %{base:.1f}%<br>"
            "Q3: %{customdata:.1f}%<extra></extra>"
        ),
        customdata=quartile_df["75th (Q3)"].values * 100,
    ))

    # Median line (markers across each bar)
    fig.add_trace(go.Scatter(
        x=periods,
        y=quartile_df["Median"].values * 100,
        mode="markers",
        name="Median",
        marker=dict(
            symbol="line-ew-open",
            size=18,
            color=COLORS["navy_700"],
            line=dict(width=2.5),
        ),
        hovertemplate="%{x}<br>Median: %{y:.1f}%<extra></extra>",
    ))

    # Fund position (prominent dot)
    fig.add_trace(go.Scatter(
        x=periods,
        y=quartile_df["Fund"].values * 100,
        mode="markers+text",
        name="Fund",
        text=[f"{v:.1f}%" for v in quartile_df["Fund"].values * 100],
        textposition="top center",
        textfont=dict(size=11, color=COLORS["navy_900"], weight=700),
        marker=dict(
            size=14,
            color=colors[0],
            line=dict(width=2.5, color=COLORS["navy_900"]),
        ),
        hovertemplate=(
            "%{x}<br>"
            "Fund: %{y:.1f}%<br>"
            "Percentile: %{customdata:.0f}<extra></extra>"
        ),
        customdata=quartile_df["Percentile"].values,
    ))

    fig.update_layout(
        **get_layout(
            title={"text": "Peer Group Quartile Distribution"},
            xaxis_title="Period",
            yaxis_title="Annualized Return (%)",
            barmode="overlay",
            hovermode="x",
        )
    )

    return fig


def rolling_percentile_chart(rolling_pct_dict: dict) -> go.Figure:
    """Line chart of rolling percentile ranking over time.

    Lower percentile = better ranking (1 = best).

    Parameters
    ----------
    rolling_pct_dict : dict
        Dictionary of {window_label: pd.Series} where each Series is
        a rolling percentile over time.

    Returns
    -------
    go.Figure
    """
    colors = get_color_sequence()

    fig = go.Figure()

    for i, (label, series) in enumerate(rolling_pct_dict.items()):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=label,
            line=dict(color=color, width=2),
            hovertemplate="%{x|%b %Y}<br>Percentile: %{y:.0f}<extra>" + label + "</extra>",
        ))

    # Quartile reference bands
    quartile_bands = [
        (0, 25, "rgba(0, 212, 170, 0.08)", "Q1 (Top)"),
        (25, 50, "rgba(99, 91, 255, 0.06)", "Q2"),
        (50, 75, "rgba(255, 159, 67, 0.06)", "Q3"),
        (75, 100, "rgba(255, 92, 92, 0.08)", "Q4 (Bottom)"),
    ]

    for y0, y1, fillcolor, band_label in quartile_bands:
        fig.add_hrect(
            y0=y0, y1=y1,
            fillcolor=fillcolor, line_width=0,
        )

    # 50th percentile reference
    fig.add_hline(
        y=50, line_width=1, line_dash="dash",
        line_color=COLORS["navy_700"], opacity=0.4,
        annotation_text="Median",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color=COLORS["navy_700"]),
    )

    fig.update_layout(
        **get_layout(
            title={"text": "Rolling Percentile Ranking"},
            xaxis_title="",
            yaxis_title="Percentile (lower = better)",
            hovermode="x unified",
            yaxis=dict(
                showgrid=False, zeroline=False,
                showline=True, linewidth=1.5,
                linecolor=COLORS["gray_200"],
                autorange="reversed",  # 1 (best) at top
                range=[100, 0],
                tickfont=dict(size=11, color=COLORS["navy_700"]),
                title_font=dict(size=12, color=COLORS["navy_700"]),
            ),
        )
    )

    return fig
