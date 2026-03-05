"""Up-capture / down-capture grouped bar chart.

Side-by-side bars showing how much of the benchmark's upside and
downside the fund captures, per benchmark.
"""

import plotly.graph_objects as go

from config.settings import COLORS
from charts.theme import get_layout, get_color_sequence


def capture_chart(capture_dict: dict) -> go.Figure:
    """Grouped bar chart of up-capture and down-capture ratios.

    Parameters
    ----------
    capture_dict : dict
        Dictionary of {benchmark_name: {"up": float, "down": float}}.
        Values are ratios (e.g. 0.85 means 85% capture). They will
        be displayed as percentages.

    Returns
    -------
    go.Figure
    """
    colors = get_color_sequence()

    names = list(capture_dict.keys())
    up_vals = [capture_dict[n]["up"] * 100 for n in names]
    down_vals = [capture_dict[n]["down"] * 100 for n in names]

    fig = go.Figure()

    # Up-capture bars (green)
    fig.add_trace(go.Bar(
        x=names,
        y=up_vals,
        name="Up Capture",
        marker_color=COLORS["green"],
        marker_line=dict(width=1.5, color=COLORS["navy_900"]),
        text=[f"{v:.0f}%" for v in up_vals],
        textposition="outside",
        textfont=dict(size=12, color=COLORS["navy_900"], weight=600),
        hovertemplate="%{x}<br>Up Capture: %{y:.1f}%<extra></extra>",
    ))

    # Down-capture bars (red)
    fig.add_trace(go.Bar(
        x=names,
        y=down_vals,
        name="Down Capture",
        marker_color=COLORS["red"],
        marker_line=dict(width=1.5, color=COLORS["navy_900"]),
        text=[f"{v:.0f}%" for v in down_vals],
        textposition="outside",
        textfont=dict(size=12, color=COLORS["navy_900"], weight=600),
        hovertemplate="%{x}<br>Down Capture: %{y:.1f}%<extra></extra>",
    ))

    # 100% reference line
    fig.add_hline(
        y=100, line_width=1.5, line_dash="dash",
        line_color=COLORS["navy_700"], opacity=0.5,
        annotation_text="100%",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color=COLORS["navy_700"]),
    )

    fig.update_layout(
        **get_layout(
            title={"text": "Up / Down Capture Ratios"},
            xaxis_title="",
            yaxis_title="Capture (%)",
            barmode="group",
            bargap=0.25,
            bargroupgap=0.1,
        )
    )

    return fig
