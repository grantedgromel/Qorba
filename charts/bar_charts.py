"""Generic horizontal bar chart for comparing metrics across funds/benchmarks.

Colors bars sequentially using the neobrutalist chart palette.
"""

import plotly.graph_objects as go

from config.settings import COLORS
from charts.theme import get_layout, get_color_sequence


def comparison_bar(names: list[str],
                   values: list[float],
                   title: str,
                   format_str: str = ".2f") -> go.Figure:
    """Horizontal bar chart comparing a single metric across entities.

    Parameters
    ----------
    names : list[str]
        Labels for each bar (fund/benchmark names).
    values : list[float]
        Metric values corresponding to each name.
    title : str
        Chart title.
    format_str : str
        Python format string for the value annotations (e.g. ".2f", ".1%").

    Returns
    -------
    go.Figure
    """
    colors = get_color_sequence()

    # Assign colors from the sequence
    bar_colors = [colors[i % len(colors)] for i in range(len(names))]

    # Format text annotations
    text_vals = [f"{v:{format_str}}" for v in values]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=names,
        x=values,
        orientation="h",
        marker_color=bar_colors,
        marker_line=dict(width=1.5, color=COLORS["navy_900"]),
        text=text_vals,
        textposition="outside",
        textfont=dict(
            family="JetBrains Mono, monospace",
            size=12,
            color=COLORS["navy_900"],
            weight=600,
        ),
        hovertemplate="%{y}: %{x" + f":{format_str}" + "}<extra></extra>",
    ))

    fig.update_layout(
        **get_layout(
            title={"text": title},
            xaxis_title="",
            yaxis_title="",
            yaxis=dict(
                showgrid=False, zeroline=False,
                showline=False,
                autorange="reversed",  # first item at top
                tickfont=dict(size=12, color=COLORS["navy_900"], weight=600),
            ),
            xaxis=dict(
                showgrid=False, zeroline=False,
                showline=True, linewidth=1.5,
                linecolor=COLORS["gray_200"],
                tickfont=dict(size=11, color=COLORS["navy_700"]),
            ),
            margin={"l": 150, "r": 80, "t": 70, "b": 40},
            showlegend=False,
        )
    )

    return fig
