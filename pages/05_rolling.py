"""Rolling Analytics page -- rolling windows to avoid endpoint sensitivity."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config.settings import CHART_COLORS, COLORS
from analytics.rolling import (
    rolling_annualized_return,
    rolling_annualized_volatility,
    rolling_sharpe,
    rolling_sortino,
    rolling_max_drawdown,
    rolling_downside_deviation,
    rolling_skewness,
    rolling_kurtosis,
    rolling_beta,
    rolling_alpha,
    rolling_correlation,
    rolling_information_ratio,
    rolling_up_capture,
    rolling_down_capture,
    rolling_win_rate,
    rolling_gain_loss_ratio,
)

st.title("Rolling Analytics")

# -- Guard ─────────────────────────────────────────────────────────────────────
fund_returns: pd.Series = st.session_state.get("fund_returns", pd.Series(dtype=float))
benchmark_returns: pd.DataFrame = st.session_state.get("benchmark_returns", pd.DataFrame())
rf_annual: float = st.session_state.get("rf_annual", 0.0)
mar_annual: float = st.session_state.get("mar_annual", 0.0)

if fund_returns.empty:
    st.info("Please upload fund return data or enable Sample Data in the sidebar to get started.")
    st.stop()

has_benchmarks = not benchmark_returns.empty

# -- Window controls ───────────────────────────────────────────────────────────
st.subheader("Window Settings")

wc1, wc2, wc3 = st.columns(3)
with wc1:
    win1 = st.number_input("Window 1 (months)", min_value=3, value=12, step=1)
with wc2:
    win2 = st.number_input("Window 2 (months)", min_value=3, value=24, step=1)
with wc3:
    win3 = st.number_input("Window 3 (months)", min_value=3, value=36, step=1)

# Preset buttons
st.caption("Quick presets:")
p1, p2, p3, p4 = st.columns(4)
with p1:
    if st.button("6 months"):
        win1 = 6
with p2:
    if st.button("12 months"):
        win1 = 12
with p3:
    if st.button("24 months"):
        win1 = 24
with p4:
    if st.button("36 months"):
        win1 = 36

windows = [win1, win2, win3]
window_labels = [f"{w}M" for w in windows]


def _rolling_metric_chart(title: str, series_dict: dict, y_format: str = ".2f",
                           y_title: str = "") -> go.Figure:
    """Build a line chart with multiple rolling series."""
    fig = go.Figure()
    color_idx = 0
    for label, s in series_dict.items():
        if s is not None and not s.empty:
            fig.add_trace(go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                name=label,
                line=dict(color=CHART_COLORS[color_idx % len(CHART_COLORS)], width=2),
            ))
            color_idx += 1
    fig.update_layout(
        title=title,
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
        yaxis=dict(tickformat=y_format),
        yaxis_title=y_title,
        xaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        font=dict(color=COLORS["text_secondary"]),
    )
    return fig


st.divider()

# -- Performance group ─────────────────────────────────────────────────────────
st.subheader("Performance")

# Rolling Return
series = {f"Return {wl}": rolling_annualized_return(fund_returns, w) for w, wl in zip(windows, window_labels)}
st.plotly_chart(_rolling_metric_chart("Rolling Annualized Return", series, y_format=".1%"), use_container_width=True)

# Rolling Volatility
series = {f"Vol {wl}": rolling_annualized_volatility(fund_returns, w) for w, wl in zip(windows, window_labels)}
st.plotly_chart(_rolling_metric_chart("Rolling Annualized Volatility", series, y_format=".1%"), use_container_width=True)

# Rolling Sharpe
series = {f"Sharpe {wl}": rolling_sharpe(fund_returns, w, rf_annual) for w, wl in zip(windows, window_labels)}
st.plotly_chart(_rolling_metric_chart("Rolling Sharpe Ratio", series), use_container_width=True)

# Rolling Sortino
series = {f"Sortino {wl}": rolling_sortino(fund_returns, w, mar_annual) for w, wl in zip(windows, window_labels)}
st.plotly_chart(_rolling_metric_chart("Rolling Sortino Ratio", series), use_container_width=True)

st.divider()

# -- Risk group ────────────────────────────────────────────────────────────────
st.subheader("Risk")

# Rolling Max DD
series = {f"Max DD {wl}": rolling_max_drawdown(fund_returns, w) for w, wl in zip(windows, window_labels)}
st.plotly_chart(_rolling_metric_chart("Rolling Max Drawdown", series, y_format=".1%"), use_container_width=True)

# Rolling Downside Dev
series = {f"DD {wl}": rolling_downside_deviation(fund_returns, w, mar_annual) for w, wl in zip(windows, window_labels)}
st.plotly_chart(_rolling_metric_chart("Rolling Downside Deviation", series, y_format=".1%"), use_container_width=True)

# Rolling Skewness
series = {f"Skew {wl}": rolling_skewness(fund_returns, w) for w, wl in zip(windows, window_labels)}
st.plotly_chart(_rolling_metric_chart("Rolling Skewness", series, y_format=".2f"), use_container_width=True)

# Rolling Kurtosis
series = {f"Kurt {wl}": rolling_kurtosis(fund_returns, w) for w, wl in zip(windows, window_labels)}
st.plotly_chart(_rolling_metric_chart("Rolling Kurtosis", series, y_format=".2f"), use_container_width=True)

st.divider()

# -- Benchmark-relative group ─────────────────────────────────────────────────
if has_benchmarks:
    st.subheader("Benchmark-Relative")

    # Use first benchmark for rolling benchmark-relative metrics
    first_bm_name = benchmark_returns.columns[0]
    first_bm = benchmark_returns.iloc[:, 0]

    # Rolling Beta
    series = {f"Beta {wl}": rolling_beta(fund_returns, first_bm, w) for w, wl in zip(windows, window_labels)}
    st.plotly_chart(
        _rolling_metric_chart(f"Rolling Beta vs {first_bm_name}", series, y_format=".2f"),
        use_container_width=True,
    )

    # Rolling Alpha
    series = {f"Alpha {wl}": rolling_alpha(fund_returns, first_bm, w) for w, wl in zip(windows, window_labels)}
    st.plotly_chart(
        _rolling_metric_chart(f"Rolling Alpha vs {first_bm_name}", series, y_format=".2%"),
        use_container_width=True,
    )

    # Rolling Correlation
    series = {f"Corr {wl}": rolling_correlation(fund_returns, first_bm, w) for w, wl in zip(windows, window_labels)}
    st.plotly_chart(
        _rolling_metric_chart(f"Rolling Correlation vs {first_bm_name}", series, y_format=".2f"),
        use_container_width=True,
    )

    # Rolling Information Ratio
    series = {f"IR {wl}": rolling_information_ratio(fund_returns, first_bm, w) for w, wl in zip(windows, window_labels)}
    st.plotly_chart(
        _rolling_metric_chart(f"Rolling Information Ratio vs {first_bm_name}", series, y_format=".2f"),
        use_container_width=True,
    )

    # Rolling Up/Down Capture
    for w, wl in zip(windows, window_labels):
        up_cap = rolling_up_capture(fund_returns, first_bm, w)
        dn_cap = rolling_down_capture(fund_returns, first_bm, w)
        series = {f"Up Capture {wl}": up_cap, f"Down Capture {wl}": dn_cap}
        st.plotly_chart(
            _rolling_metric_chart(f"Rolling Up/Down Capture ({wl}) vs {first_bm_name}", series, y_format=".1%"),
            use_container_width=True,
        )
        break  # only show for the first window to avoid too many charts

    st.divider()

# -- Style group ───────────────────────────────────────────────────────────────
st.subheader("Style")

# Rolling Win Rate
series = {f"Win Rate {wl}": rolling_win_rate(fund_returns, w) for w, wl in zip(windows, window_labels)}
st.plotly_chart(_rolling_metric_chart("Rolling Win Rate", series, y_format=".0%"), use_container_width=True)

# Rolling Gain/Loss Ratio
series = {f"G/L {wl}": rolling_gain_loss_ratio(fund_returns, w) for w, wl in zip(windows, window_labels)}
st.plotly_chart(_rolling_metric_chart("Rolling Gain/Loss Ratio", series, y_format=".2f"), use_container_width=True)
