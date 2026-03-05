"""Correlation visualizations.

1. correlation_heatmap      - Plotly heatmap for a correlation matrix
2. rolling_correlation_chart - overlaid rolling correlation series
"""

import pandas as pd
import plotly.graph_objects as go

from config.settings import COLORS
from charts.theme import get_layout, get_color_sequence


def correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Plotly heatmap for a correlation matrix.

    Uses a blue-white-red diverging colorscale centered at zero.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Square correlation matrix (e.g. from returns_df.corr()).

    Returns
    -------
    go.Figure
    """
    labels = list(corr_matrix.columns)

    # Blue-white-red scale: negative = blue, zero = white, positive = red
    colorscale = [
        [0.0, "#635BFF"],   # strong negative -> primary blue
        [0.5, "#FFFFFF"],   # zero -> white
        [1.0, "#FF5C5C"],   # strong positive -> red
    ]

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=corr_matrix.values,
        x=labels,
        y=labels,
        colorscale=colorscale,
        zmin=-1, zmax=1,
        text=corr_matrix.values.round(2),
        texttemplate="%{text:.2f}",
        textfont=dict(size=12, color=COLORS["navy_900"]),
        hovertemplate="%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>",
        colorbar=dict(
            title="Corr",
            titlefont=dict(size=12, color=COLORS["navy_700"]),
            tickfont=dict(size=10, color=COLORS["navy_700"]),
            thickness=12,
            len=0.75,
        ),
    ))

    fig.update_layout(
        **get_layout(
            title={"text": "Correlation Matrix"},
            xaxis=dict(
                showgrid=False, zeroline=False, showline=False,
                tickfont=dict(size=11, color=COLORS["navy_700"]),
                side="bottom",
            ),
            yaxis=dict(
                showgrid=False, zeroline=False, showline=False,
                tickfont=dict(size=11, color=COLORS["navy_700"]),
                autorange="reversed",
            ),
            hovermode="closest",
        )
    )

    return fig


def rolling_correlation_chart(rolling_series_dict: dict) -> go.Figure:
    """Overlaid rolling correlation time series.

    Parameters
    ----------
    rolling_series_dict : dict
        Dictionary of {label: pd.Series} where each Series is a
        rolling correlation over time.

    Returns
    -------
    go.Figure
    """
    colors = get_color_sequence()

    fig = go.Figure()

    for i, (label, series) in enumerate(rolling_series_dict.items()):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=label,
            line=dict(color=color, width=2),
            hovertemplate="%{x|%b %Y}<br>Corr: %{y:.3f}<extra>" + label + "</extra>",
        ))

    # Reference lines at key levels
    for level in [0.0]:
        fig.add_hline(
            y=level, line_width=1, line_dash="dot",
            line_color=COLORS["navy_700"], opacity=0.4,
        )

    fig.update_layout(
        **get_layout(
            title={"text": "Rolling Correlation"},
            xaxis_title="",
            yaxis_title="Correlation",
            hovermode="x unified",
            yaxis=dict(
                showgrid=False, zeroline=False,
                showline=True, linewidth=1.5,
                linecolor=COLORS["gray_200"],
                range=[-1.05, 1.05],
                tickfont=dict(size=11, color=COLORS["navy_700"]),
                title_font=dict(size=12, color=COLORS["navy_700"]),
            ),
        )
    )

    return fig
