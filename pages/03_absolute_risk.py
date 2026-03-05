"""Risk Measures page -- absolute risk metrics from Travers Chapter 6."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config.settings import CHART_COLORS, COLORS
from analytics.absolute_risk import (
    standard_deviation,
    gain_std,
    loss_std,
    downside_deviation,
    semideviation,
    skewness,
    kurtosis,
    excess_kurtosis,
    quarterly_returns,
    max_drawdown,
    gain_loss_ratio,
    avg_gain,
    avg_loss,
    win_rate,
)

st.title("Risk Measures")

# ── Guard ─────────────────────────────────────────────────────────────────────
fund_returns: pd.Series = st.session_state.get("fund_returns", pd.Series(dtype=float))
mar_annual: float = st.session_state.get("mar_annual", 0.0)

if fund_returns.empty:
    st.info("Please upload fund return data or enable Sample Data in the sidebar to get started.")
    st.stop()

qtr = quarterly_returns(fund_returns)

# ── Standard Deviation section ────────────────────────────────────────────────
st.subheader("Standard Deviation Measures")

sd = standard_deviation(fund_returns)
g_std = gain_std(fund_returns)
l_std = loss_std(fund_returns)

c1, c2, c3 = st.columns(3)
c1.metric("Annualized Std Dev", f"{sd:.2%}")
c2.metric("Gain Std Dev", f"{g_std:.2%}")
c3.metric("Loss Std Dev", f"{l_std:.2%}")

# Bar chart for std dev breakdown
fig_std = go.Figure(go.Bar(
    x=["Annualized Std Dev", "Gain Std Dev", "Loss Std Dev"],
    y=[sd, g_std, l_std],
    marker_color=[CHART_COLORS[0], COLORS["green"], COLORS["red"]],
    marker_line_color=COLORS["navy_900"],
    marker_line_width=1.5,
    text=[f"{sd:.2%}", f"{g_std:.2%}", f"{l_std:.2%}"],
    textposition="outside",
))
fig_std.update_layout(
    template="plotly_white",
    height=300,
    margin=dict(l=40, r=20, t=20, b=40),
    yaxis=dict(tickformat=".2%"),
)
st.plotly_chart(fig_std, use_container_width=True)

st.divider()

# ── Downside Deviation & Semideviation ────────────────────────────────────────
st.subheader("Downside Risk")

dd_val = downside_deviation(fund_returns, mar_annual)
semi_val = semideviation(fund_returns)

c1, c2 = st.columns(2)
c1.metric("Downside Deviation", f"{dd_val:.2%}")
c2.metric("Semideviation", f"{semi_val:.2%}")

st.divider()

# ── Skewness ──────────────────────────────────────────────────────────────────
st.subheader("Skewness")

skew_m = skewness(fund_returns)
skew_q = skewness(qtr)

c1, c2 = st.columns(2)
c1.metric("Skewness (Monthly)", f"{skew_m:.4f}")
c2.metric("Skewness (Quarterly)", f"{skew_q:.4f}")

if skew_m > 0:
    st.success("Positive skew: return distribution has a longer right tail -- more extreme "
               "gains than losses. This is generally favorable for hedge funds.")
elif skew_m < 0:
    st.warning("Negative skew: return distribution has a longer left tail -- more extreme "
               "losses than gains. Common in short-volatility or insurance-selling strategies.")
else:
    st.info("Skewness near zero suggests a roughly symmetric return distribution.")

st.divider()

# ── Kurtosis ──────────────────────────────────────────────────────────────────
st.subheader("Kurtosis")

kurt_m = kurtosis(fund_returns)
kurt_q = kurtosis(qtr)
excess_m = excess_kurtosis(fund_returns)
excess_q = excess_kurtosis(qtr)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Kurtosis (Monthly)", f"{kurt_m:.4f}")
c2.metric("Kurtosis (Quarterly)", f"{kurt_q:.4f}")
c3.metric("Excess Kurt (Monthly)", f"{excess_m:.4f}")
c4.metric("Excess Kurt (Quarterly)", f"{excess_q:.4f}")

if excess_m > 0:
    st.warning("Leptokurtic (excess kurtosis > 0): fatter tails than a normal distribution. "
               "Higher probability of extreme returns (both positive and negative).")
elif excess_m < 0:
    st.success("Platykurtic (excess kurtosis < 0): thinner tails than normal. "
               "Lower probability of extreme returns.")
else:
    st.info("Excess kurtosis near zero is consistent with a normal distribution.")

st.divider()

# ── Gain / Loss Ratio ────────────────────────────────────────────────────────
st.subheader("Gain / Loss Analysis")

gl_ratio = gain_loss_ratio(fund_returns)
a_gain = avg_gain(fund_returns)
a_loss = avg_loss(fund_returns)
wr = win_rate(fund_returns)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Gain/Loss Ratio", f"{gl_ratio:.2f}")
c2.metric("Avg Gain", f"{a_gain:.2%}")
c3.metric("Avg Loss", f"{a_loss:.2%}")
c4.metric("Win Rate", f"{wr:.2%}")

# Summary table
gl_table = pd.DataFrame({
    "Metric": ["Average Gain", "Average Loss", "Gain/Loss Ratio", "Win Rate",
               "Positive Months", "Negative Months"],
    "Value": [
        f"{a_gain:.2%}",
        f"{a_loss:.2%}",
        f"{gl_ratio:.2f}",
        f"{wr:.2%}",
        str(int((fund_returns > 0).sum())),
        str(int((fund_returns < 0).sum())),
    ],
})
st.dataframe(gl_table.set_index("Metric"), use_container_width=True)
