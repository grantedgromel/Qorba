"""Generic rolling metric line chart.

Plots one or more rolling window series overlaid, with an optional
horizontal reference line and latest-value marker dots.
"""

import pandas as pd
import plotly.graph_objects as go

from config.settings import COLORS
from charts.theme import get_layout, get_color_sequence


def rolling_metric_chart(series_dict: dict,
                         title: str,
                         y_label: str,
                         reference_value: float | None = None) -> go.Figure:
    """Overlaid rolling metric chart with latest-value dots.

    Parameters
    ----------
    series_dict : dict
        Dictionary of {window_label: pd.Series} where each Series
        contains the rolling metric indexed by date.
        Example: {"12M": rolling_sharpe_12, "36M": rolling_sharpe_36}
    title : str
        Chart title.
    y_label : str
        Y-axis label.
    reference_value : float, optional
        If provided, draw a horizontal dashed reference line at this value.

    Returns
    -------
    go.Figure
    """
    colors = get_color_sequence()

    fig = go.Figure()

    for i, (label, series) in enumerate(series_dict.items()):
        series = series.dropna()
        if len(series) == 0:
            continue

        color = colors[i % len(colors)]

        # Main line
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=label,
            line=dict(color=color, width=2.2),
            hovertemplate=(
                "%{x|%b %Y}<br>"
                + y_label + ": %{y:.3f}"
                + f"<extra>{label}</extra>"
            ),
        ))

        # Latest value dot
        last_date = series.index[-1]
        last_val = series.iloc[-1]
        fig.add_trace(go.Scatter(
            x=[last_date],
            y=[last_val],
            mode="markers+text",
            name=f"{label} (latest)",
            text=[f"{last_val:.2f}"],
            textposition="top right",
            textfont=dict(size=10, color=color, weight=600),
            marker=dict(
                size=9,
                color=color,
                line=dict(width=2, color=COLORS["navy_900"]),
            ),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Reference line
    if reference_value is not None:
        fig.add_hline(
            y=reference_value,
            line_width=1.5,
            line_dash="dash",
            line_color=COLORS["navy_700"],
            opacity=0.5,
            annotation_text=f"{reference_value:.2f}",
            annotation_position="bottom right",
            annotation_font=dict(size=10, color=COLORS["navy_700"]),
        )

    fig.update_layout(
        **get_layout(
            title={"text": title},
            xaxis_title="",
            yaxis_title=y_label,
            hovermode="x unified",
        )
    )

    return fig
