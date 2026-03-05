"""Drawdown Analysis page -- episodes table, underwater chart, and highlight chart."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config.settings import CHART_COLORS, COLORS
from analytics.absolute_risk import drawdown_series, drawdown_table

st.title("Drawdown Analysis")

# ── Guard ─────────────────────────────────────────────────────────────────────
fund_returns: pd.Series = st.session_state.get("fund_returns", pd.Series(dtype=float))

if fund_returns.empty:
    st.info("Please upload fund return data or enable Sample Data in the sidebar to get started.")
    st.stop()

# ── Drawdown episodes table (Travers Table 6.3) ──────────────────────────────
st.subheader("Drawdown Episodes")

dd_tbl = drawdown_table(fund_returns, top_n=10)

if dd_tbl.empty:
    st.success("No drawdown episodes found -- the fund has never drawn down from its peak.")
else:
    display_tbl = dd_tbl.copy()
    display_tbl["Max Drawdown"] = display_tbl["Max Drawdown"].apply(lambda x: f"{x:.2%}")
    display_tbl.index = range(1, len(display_tbl) + 1)
    display_tbl.index.name = "#"
    st.dataframe(display_tbl, use_container_width=True)

st.divider()

# ── Full underwater chart ─────────────────────────────────────────────────────
st.subheader("Underwater Chart")

dd = drawdown_series(fund_returns)

fig_uw = go.Figure()
fig_uw.add_trace(go.Scatter(
    x=dd.index,
    y=dd.values,
    fill="tozeroy",
    mode="lines",
    line=dict(color=COLORS["red"], width=1.5),
    fillcolor="rgba(255, 92, 92, 0.3)",
    name="Drawdown",
    hovertemplate="%{x|%b %Y}: %{y:.2%}<extra></extra>",
))
fig_uw.update_layout(
    yaxis_title="Drawdown from Peak",
    xaxis_title="",
    template="plotly_white",
    height=380,
    margin=dict(l=40, r=20, t=20, b=40),
    yaxis=dict(tickformat=".1%"),
    showlegend=False,
    hovermode="x unified",
)
st.plotly_chart(fig_uw, use_container_width=True)

st.divider()

# ── Top 5 drawdown highlight chart ───────────────────────────────────────────
st.subheader("Top 5 Drawdowns on Cumulative Performance")

cumulative = (1 + fund_returns).cumprod()

fig_hl = go.Figure()

# Cumulative line
fig_hl.add_trace(go.Scatter(
    x=cumulative.index,
    y=cumulative.values,
    mode="lines",
    line=dict(color=CHART_COLORS[0], width=2.5),
    name="Cumulative Return",
))

# Highlight the top 5 drawdown periods
if not dd_tbl.empty:
    # Re-derive drawdown episodes to get date ranges
    wealth = (1 + fund_returns).cumprod()
    running_max = wealth.cummax()
    dd_raw = (wealth - running_max) / running_max

    # Find drawdown episode boundaries
    in_dd = dd_raw < 0
    episodes = []
    start = None
    for i, val in enumerate(in_dd.values):
        if val and start is None:
            start = i
        elif not val and start is not None:
            episodes.append((start, i - 1))
            start = None
    if start is not None:
        episodes.append((start, len(in_dd) - 1))

    # Sort by severity and take top 5
    episode_depths = []
    for s, e in episodes:
        depth = dd_raw.iloc[s:e + 1].min()
        episode_depths.append((s, e, depth))
    episode_depths.sort(key=lambda x: x[2])
    top_5 = episode_depths[:5]

    highlight_colors = [
        "rgba(255, 92, 92, 0.25)",
        "rgba(255, 159, 67, 0.25)",
        "rgba(255, 217, 61, 0.25)",
        "rgba(0, 212, 170, 0.20)",
        "rgba(122, 115, 255, 0.20)",
    ]

    for idx, (s, e, depth) in enumerate(top_5):
        start_date = fund_returns.index[max(0, s - 1)]
        end_date = fund_returns.index[min(e, len(fund_returns) - 1)]
        fig_hl.add_vrect(
            x0=start_date,
            x1=end_date,
            fillcolor=highlight_colors[idx % len(highlight_colors)],
            layer="below",
            line_width=0,
            annotation_text=f"#{idx+1} ({depth:.1%})",
            annotation_position="top left",
            annotation_font_size=10,
        )

fig_hl.update_layout(
    yaxis_title="Growth of $1",
    xaxis_title="",
    template="plotly_white",
    height=450,
    margin=dict(l=40, r=20, t=30, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hovermode="x unified",
)
st.plotly_chart(fig_hl, use_container_width=True)
