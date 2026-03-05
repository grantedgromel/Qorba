"""Overview page -- high-level fund performance snapshot."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config.settings import CHART_COLORS, COLORS
from analytics.absolute_return import (
    annualized_return_geometric,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
)
from analytics.absolute_risk import max_drawdown, drawdown_series

st.title("Overview")

# ── Guard: data must be loaded ────────────────────────────────────────────────
fund_returns: pd.Series = st.session_state.get("fund_returns", pd.Series(dtype=float))
benchmark_returns: pd.DataFrame = st.session_state.get("benchmark_returns", pd.DataFrame())
rf_annual: float = st.session_state.get("rf_annual", 0.0)
mar_annual: float = st.session_state.get("mar_annual", 0.0)

if fund_returns.empty:
    st.info("Please upload fund return data or enable Sample Data in the sidebar to get started.")
    st.stop()

# ── Key metrics row ───────────────────────────────────────────────────────────
ann_ret = annualized_return_geometric(fund_returns)
ann_vol = annualized_volatility(fund_returns)
sr = sharpe_ratio(fund_returns, rf_annual)
mdd = max_drawdown(fund_returns)
sort_r = sortino_ratio(fund_returns, mar_annual)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Annualized Return", f"{ann_ret:.2%}")
c2.metric("Sharpe Ratio", f"{sr:.2f}")
c3.metric("Max Drawdown", f"{mdd:.2%}")
c4.metric("Sortino Ratio", f"{sort_r:.2f}")

st.divider()

# ── Cumulative performance chart ──────────────────────────────────────────────
st.subheader("Cumulative Performance")

cumulative_fund = (1 + fund_returns).cumprod()

fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(
    x=cumulative_fund.index,
    y=cumulative_fund.values,
    mode="lines",
    name="Fund",
    line=dict(color=CHART_COLORS[0], width=3),
))

if not benchmark_returns.empty:
    for i, col in enumerate(benchmark_returns.columns):
        cum_bm = (1 + benchmark_returns[col]).cumprod()
        fig_cum.add_trace(go.Scatter(
            x=cum_bm.index,
            y=cum_bm.values,
            mode="lines",
            name=col,
            line=dict(color=CHART_COLORS[(i + 1) % len(CHART_COLORS)], width=2, dash="dash"),
        ))

fig_cum.update_layout(
    yaxis_title="Growth of $1",
    xaxis_title="",
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=40, r=20, t=30, b=40),
    height=420,
)
st.plotly_chart(fig_cum, use_container_width=True)

st.divider()

# ── Two-column section: Histogram + Drawdown ──────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Return Distribution")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=fund_returns.values,
        nbinsx=30,
        marker_color=CHART_COLORS[0],
        marker_line_color=COLORS["navy_900"],
        marker_line_width=1,
        opacity=0.85,
        name="Fund",
    ))
    fig_hist.update_layout(
        xaxis_title="Monthly Return",
        yaxis_title="Frequency",
        template="plotly_white",
        showlegend=False,
        margin=dict(l=40, r=20, t=20, b=40),
        height=350,
        xaxis=dict(tickformat=".1%"),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col_right:
    st.subheader("Underwater Chart")
    dd = drawdown_series(fund_returns)
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd.index,
        y=dd.values,
        fill="tozeroy",
        mode="lines",
        line=dict(color=COLORS["red"], width=1.5),
        fillcolor="rgba(255, 92, 92, 0.3)",
        name="Drawdown",
    ))
    fig_dd.update_layout(
        yaxis_title="Drawdown",
        xaxis_title="",
        template="plotly_white",
        showlegend=False,
        margin=dict(l=40, r=20, t=20, b=40),
        height=350,
        yaxis=dict(tickformat=".1%"),
    )
    st.plotly_chart(fig_dd, use_container_width=True)

st.divider()

# ── Summary statistics table ──────────────────────────────────────────────────
st.subheader("Summary Statistics")

stats = {
    "Annualized Return": f"{ann_ret:.2%}",
    "Annualized Volatility": f"{ann_vol:.2%}",
    "Sharpe Ratio": f"{sr:.2f}",
    "Sortino Ratio": f"{sort_r:.2f}",
    "Max Drawdown": f"{mdd:.2%}",
    "Best Month": f"{fund_returns.max():.2%}",
    "Worst Month": f"{fund_returns.min():.2%}",
    "Positive Months": f"{(fund_returns > 0).sum()} / {len(fund_returns)}",
    "Average Monthly Return": f"{fund_returns.mean():.2%}",
    "Monthly Std Dev": f"{fund_returns.std():.2%}",
}

stats_df = pd.DataFrame.from_dict(stats, orient="index", columns=["Fund"])

if not benchmark_returns.empty:
    for col in benchmark_returns.columns:
        bm = benchmark_returns[col]
        bm_ann_ret = annualized_return_geometric(bm)
        bm_vol = annualized_volatility(bm)
        bm_sr = sharpe_ratio(bm, rf_annual)
        bm_sort = sortino_ratio(bm, mar_annual)
        bm_mdd = max_drawdown(bm)
        stats_df[col] = [
            f"{bm_ann_ret:.2%}",
            f"{bm_vol:.2%}",
            f"{bm_sr:.2f}",
            f"{bm_sort:.2f}",
            f"{bm_mdd:.2%}",
            f"{bm.max():.2%}",
            f"{bm.min():.2%}",
            f"{(bm > 0).sum()} / {len(bm)}",
            f"{bm.mean():.2%}",
            f"{bm.std():.2%}",
        ]

st.dataframe(stats_df, use_container_width=True)
