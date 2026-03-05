"""Return Measures page -- absolute return ratios from Travers Chapter 6."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config.settings import CHART_COLORS, COLORS
from analytics.absolute_return import (
    annualized_return_geometric,
    sharpe_ratio,
    m2_ratio,
    information_ratio,
    mar_ratio,
    calmar_ratio,
    sterling_ratio,
    sortino_ratio,
    omega_ratio,
    compute_all_return_metrics,
)

st.title("Return Measures")

# ── Guard ─────────────────────────────────────────────────────────────────────
fund_returns: pd.Series = st.session_state.get("fund_returns", pd.Series(dtype=float))
benchmark_returns: pd.DataFrame = st.session_state.get("benchmark_returns", pd.DataFrame())
rf_annual: float = st.session_state.get("rf_annual", 0.0)
mar_annual: float = st.session_state.get("mar_annual", 0.0)
omega_threshold: float = st.session_state.get("omega_threshold", 0.0)

if fund_returns.empty:
    st.info("Please upload fund return data or enable Sample Data in the sidebar to get started.")
    st.stop()

has_benchmarks = not benchmark_returns.empty

# ── Compute fund metrics ──────────────────────────────────────────────────────
first_bm = benchmark_returns.iloc[:, 0] if has_benchmarks else None
fund_metrics = compute_all_return_metrics(
    fund_returns, rf_annual, mar_annual, omega_threshold, first_bm
)


def _comparison_bar(metric_name: str, fund_val: float, is_pct: bool = False):
    """Bar chart comparing fund vs benchmarks for a single metric."""
    names = ["Fund"]
    values = [fund_val]
    colors = [CHART_COLORS[0]]

    if has_benchmarks:
        for i, col in enumerate(benchmark_returns.columns):
            bm = benchmark_returns[col]
            bm_metrics = compute_all_return_metrics(
                bm, rf_annual, mar_annual, omega_threshold, None
            )
            if metric_name in bm_metrics:
                names.append(col)
                values.append(bm_metrics[metric_name])
                colors.append(CHART_COLORS[(i + 1) % len(CHART_COLORS)])

    fig = go.Figure(go.Bar(
        x=names,
        y=values,
        marker_color=colors,
        marker_line_color=COLORS["navy_900"],
        marker_line_width=1.5,
        text=[f"{v:.2%}" if is_pct else f"{v:.2f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_white",
        height=280,
        margin=dict(l=40, r=20, t=20, b=40),
        yaxis_title=metric_name,
        yaxis=dict(tickformat=".2%" if is_pct else ".2f"),
    )
    return fig


# ── Metric definitions with formulas ─────────────────────────────────────────
metric_configs = [
    {
        "name": "Sharpe Ratio",
        "key": "Sharpe Ratio",
        "formula": "Sharpe = (Annualized Return - Risk-Free Rate) / Annualized Volatility",
        "pct": False,
    },
    {
        "name": "M2 Ratio",
        "key": "M2 Ratio",
        "formula": "M2 = Sharpe x Benchmark Volatility + Risk-Free Rate. Requires benchmark.",
        "pct": True,
        "needs_benchmark": True,
    },
    {
        "name": "Information Ratio",
        "key": "Information Ratio",
        "formula": "IR = (Return - Benchmark Return) / Tracking Error. Requires benchmark.",
        "pct": False,
        "needs_benchmark": True,
    },
    {
        "name": "MAR Ratio",
        "key": "MAR Ratio",
        "formula": "MAR = Annualized Return / |Max Drawdown| (since inception).",
        "pct": False,
    },
    {
        "name": "Calmar Ratio",
        "key": "Calmar Ratio",
        "formula": "Calmar = Annualized Return (3yr) / |Max Drawdown (3yr)|. Requires 36 months.",
        "pct": False,
    },
    {
        "name": "Sterling Ratio",
        "key": "Sterling Ratio",
        "formula": "Sterling = Annualized Return / (|Max Drawdown| + 10%).",
        "pct": False,
    },
    {
        "name": "Sortino Ratio",
        "key": "Sortino Ratio",
        "formula": "Sortino = (Annualized Return - MAR) / Downside Deviation.",
        "pct": False,
    },
    {
        "name": "Omega Ratio",
        "key": "Omega Ratio",
        "formula": "Omega = Sum of gains above threshold / Sum of losses below threshold.",
        "pct": False,
    },
]

# ── Render each metric ────────────────────────────────────────────────────────
for cfg in metric_configs:
    needs_bm = cfg.get("needs_benchmark", False)
    if needs_bm and not has_benchmarks:
        continue

    val = fund_metrics.get(cfg["key"])
    if val is None:
        continue

    st.subheader(cfg["name"])

    col_metric, col_chart = st.columns([1, 2])

    with col_metric:
        if cfg["pct"]:
            st.metric(cfg["name"], f"{val:.2%}")
        else:
            st.metric(cfg["name"], f"{val:.2f}")

    with col_chart:
        if has_benchmarks:
            fig = _comparison_bar(cfg["key"], val, is_pct=cfg["pct"])
            st.plotly_chart(fig, use_container_width=True)

    with st.expander(f"Formula: {cfg['name']}"):
        st.markdown(cfg["formula"])

    st.divider()

# ── Full metrics table ────────────────────────────────────────────────────────
st.subheader("All Return Metrics")

rows = {}
for k, v in fund_metrics.items():
    if v is None:
        rows[k] = "N/A"
    elif "Ratio" not in k and "Volatility" in k or "Return" in k:
        rows[k] = f"{v:.2%}"
    else:
        rows[k] = f"{v:.4f}"

table_df = pd.DataFrame.from_dict(rows, orient="index", columns=["Fund"])

if has_benchmarks:
    for col in benchmark_returns.columns:
        bm = benchmark_returns[col]
        bm_metrics = compute_all_return_metrics(
            bm, rf_annual, mar_annual, omega_threshold, None
        )
        col_vals = []
        for k in fund_metrics.keys():
            bm_val = bm_metrics.get(k)
            if bm_val is None:
                col_vals.append("N/A")
            elif "Ratio" not in k and ("Volatility" in k or "Return" in k):
                col_vals.append(f"{bm_val:.2%}")
            else:
                col_vals.append(f"{bm_val:.4f}")
        table_df[col] = col_vals

st.dataframe(table_df, use_container_width=True)
