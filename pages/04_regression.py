"""Regression Analysis page -- OLS regression and correlation from Travers Chapter 6."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from config.settings import CHART_COLORS, COLORS
from analytics.regression import (
    compute_regression,
    compute_all_regression_metrics,
    correlation_matrix,
)

st.title("Regression Analysis")

# -- Guard ─────────────────────────────────────────────────────────────────────
fund_returns: pd.Series = st.session_state.get("fund_returns", pd.Series(dtype=float))
benchmark_returns: pd.DataFrame = st.session_state.get("benchmark_returns", pd.DataFrame())
rf_annual: float = st.session_state.get("rf_annual", 0.0)

if fund_returns.empty:
    st.info("Please upload fund return data or enable Sample Data in the sidebar to get started.")
    st.stop()

if benchmark_returns.empty:
    st.warning("Regression analysis requires benchmark data. Please upload benchmark returns "
               "or enable Sample Data.")
    st.stop()

# -- Per-benchmark regression ──────────────────────────────────────────────────
for i, col in enumerate(benchmark_returns.columns):
    bm = benchmark_returns[col]
    st.subheader(f"Regression vs {col}")

    reg = compute_regression(fund_returns, bm)
    metrics = compute_all_regression_metrics(fund_returns, bm, rf_annual)

    col_chart, col_stats = st.columns([2, 1])

    with col_chart:
        # Scatter with regression line
        x_vals = bm.values
        y_vals = fund_returns.values
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_line = reg["alpha"] + reg["beta"] * x_line

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker=dict(
                color=CHART_COLORS[0],
                size=7,
                line=dict(color=COLORS["border"], width=0.5),
                opacity=0.7,
            ),
            name="Monthly Returns",
        ))
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            line=dict(color=COLORS["red"], width=2.5),
            name=f"OLS: y = {reg['alpha']:.4f} + {reg['beta']:.2f}x",
        ))
        fig.update_layout(
            xaxis_title=f"{col} Return",
            yaxis_title="Fund Return",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=380,
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis=dict(tickformat=".1%"),
            yaxis=dict(tickformat=".1%"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            font=dict(color=COLORS["text_secondary"]),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_stats:
        stats_df = pd.DataFrame({
            "Metric": list(metrics.keys()),
            "Value": [f"{v:.4f}" for v in metrics.values()],
        })
        st.dataframe(stats_df.set_index("Metric"), use_container_width=True)

    st.divider()

# -- Correlation Heatmap ───────────────────────────────────────────────────────
st.subheader("Correlation Matrix")

combined = pd.DataFrame({"Fund": fund_returns})
for col in benchmark_returns.columns:
    combined[col] = benchmark_returns[col]

corr = correlation_matrix(combined)

fig_heat = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns.tolist(),
    y=corr.index.tolist(),
    colorscale=[
        [0.0, COLORS["red"]],
        [0.5, COLORS["bg_card"]],
        [1.0, CHART_COLORS[0]],
    ],
    zmin=-1,
    zmax=1,
    text=[[f"{v:.2f}" for v in row] for row in corr.values],
    texttemplate="%{text}",
    textfont=dict(size=14, color=COLORS["text_primary"]),
    hovertemplate="(%{x}, %{y}): %{z:.3f}<extra></extra>",
))
fig_heat.update_layout(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    height=400,
    margin=dict(l=60, r=20, t=20, b=60),
    xaxis=dict(side="bottom"),
    font=dict(color=COLORS["text_secondary"]),
)
st.plotly_chart(fig_heat, use_container_width=True)
