"""Peer Group Analysis page -- quartile distribution and rolling percentile ranking."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config.settings import CHART_COLORS, COLORS
from analytics.peer_group import quartile_distribution, rolling_percentile

st.title("Peer Group Analysis")

# -- Guard ─────────────────────────────────────────────────────────────────────
fund_returns: pd.Series = st.session_state.get("fund_returns", pd.Series(dtype=float))
peer_returns: pd.DataFrame = st.session_state.get("peer_returns", pd.DataFrame())

if fund_returns.empty:
    st.info("Please upload fund return data or enable Sample Data in the sidebar to get started.")
    st.stop()

if peer_returns.empty:
    st.info("Peer group analysis requires peer returns data. Please upload a peer returns file "
            "or enable Sample Data.")
    st.stop()

# -- Quartile Distribution ────────────────────────────────────────────────────
st.subheader("Quartile Distribution Across Periods")

periods = [12, 24, 36, 48, 60]
available_periods = [p for p in periods if len(fund_returns) >= p]

if not available_periods:
    st.warning("Not enough data for any standard period (minimum 12 months required).")
    st.stop()

qd = quartile_distribution(fund_returns, peer_returns, periods=available_periods)

if qd.empty:
    st.warning("Unable to compute quartile distribution. Check that peer data aligns with fund data.")
    st.stop()

# Display table
display_qd = qd.copy()
for col in ["5th", "25th (Q1)", "Median", "75th (Q3)", "95th", "Fund"]:
    if col in display_qd.columns:
        display_qd[col] = display_qd[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
if "Percentile" in display_qd.columns:
    display_qd["Percentile"] = display_qd["Percentile"].apply(
        lambda x: f"{x:.0f}th" if pd.notna(x) else "N/A"
    )

st.dataframe(display_qd.set_index("Period"), use_container_width=True)

# Quartile box-style chart
fig_q = go.Figure()

for _, row in qd.iterrows():
    period = row["Period"]
    # Box range: 25th to 75th
    fig_q.add_trace(go.Bar(
        x=[period],
        y=[row["75th (Q3)"] - row["25th (Q1)"]],
        base=[row["25th (Q1)"]],
        marker_color="rgba(129, 140, 248, 0.2)",
        marker_line_color=COLORS["border"],
        marker_line_width=0.5,
        name="Q1-Q3 Range" if _ == 0 else None,
        showlegend=(_ == 0),
        width=0.6,
    ))

    # Median line
    fig_q.add_trace(go.Scatter(
        x=[period],
        y=[row["Median"]],
        mode="markers",
        marker=dict(color=COLORS["text_secondary"], size=12, symbol="line-ew-open", line_width=3),
        name="Median" if _ == 0 else None,
        showlegend=(_ == 0),
    ))

    # Fund marker
    fig_q.add_trace(go.Scatter(
        x=[period],
        y=[row["Fund"]],
        mode="markers",
        marker=dict(color=CHART_COLORS[0], size=14, symbol="diamond",
                    line=dict(color=COLORS["border"], width=0.5)),
        name="Fund" if _ == 0 else None,
        showlegend=(_ == 0),
    ))

fig_q.update_layout(
    yaxis_title="Annualized Return",
    xaxis_title="Period",
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    height=420,
    margin=dict(l=40, r=20, t=30, b=40),
    yaxis=dict(tickformat=".1%"),
    barmode="overlay",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    font=dict(color=COLORS["text_secondary"]),
)
st.plotly_chart(fig_q, use_container_width=True)

st.divider()

# -- Rolling Percentile ───────────────────────────────────────────────────────
st.subheader("Rolling Percentile Ranking")

roll_window = st.selectbox("Rolling Window (months)", [6, 12, 24, 36], index=1)
roll_pct = rolling_percentile(fund_returns, peer_returns, window_months=roll_window)

if roll_pct.empty or roll_pct.isna().all():
    st.warning("Not enough overlapping data to compute rolling percentile for this window.")
else:
    fig_roll = go.Figure()

    fig_roll.add_trace(go.Scatter(
        x=roll_pct.index,
        y=roll_pct.values,
        mode="lines",
        line=dict(color=CHART_COLORS[0], width=2.5),
        name="Fund Percentile",
    ))

    # Quartile reference lines
    fig_roll.add_hline(y=25, line_dash="dash", line_color=COLORS["green"],
                       annotation_text="Top Quartile", annotation_position="top left",
                       annotation_font_color=COLORS["text_secondary"])
    fig_roll.add_hline(y=50, line_dash="dash", line_color=COLORS["text_muted"],
                       annotation_text="Median",
                       annotation_font_color=COLORS["text_secondary"])
    fig_roll.add_hline(y=75, line_dash="dash", line_color=COLORS["red"],
                       annotation_text="Bottom Quartile", annotation_position="bottom left",
                       annotation_font_color=COLORS["text_secondary"])

    fig_roll.update_layout(
        yaxis_title="Percentile (1=best, 100=worst)",
        xaxis_title="",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=400,
        margin=dict(l=40, r=20, t=30, b=40),
        yaxis=dict(autorange="reversed", range=[0, 100]),
        hovermode="x unified",
        font=dict(color=COLORS["text_secondary"]),
    )
    st.plotly_chart(fig_roll, use_container_width=True)
