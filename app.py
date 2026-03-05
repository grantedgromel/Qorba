"""Qorba -- Hedge Fund Analytics Dashboard (Streamlit entry point)."""

import streamlit as st
import pandas as pd

from config.settings import CUSTOM_CSS, FUND_TYPES, LOGO_SVG
from data.loader import load_fund_returns, load_returns, align_dates, validate_data
from data.sample_data import generate_sample_data

# -- Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QORBA | Hedge Fund Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject dark mode CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -- Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(LOGO_SVG, unsafe_allow_html=True)
    st.divider()

    # Fund type selector
    fund_type = st.selectbox("Fund Type", FUND_TYPES, index=0)
    st.session_state["fund_type"] = fund_type

    st.divider()

    # File uploaders
    fund_file = st.file_uploader("Fund Returns (required)", type=["csv", "xlsx", "xls"])
    bench_file = st.file_uploader("Benchmark Returns (optional)", type=["csv", "xlsx", "xls"])
    peer_file = st.file_uploader("Peer Group Returns (optional)", type=["csv", "xlsx", "xls"])

    st.divider()

    # Sample data toggle
    use_sample = st.checkbox("Use Sample Data", value=False)

    st.divider()

    # Parameter inputs
    rf_pct = st.number_input("Risk-Free Rate (%)", value=0.0, step=0.25, format="%.2f")
    mar_pct = st.number_input("MAR (%)", value=0.0, step=0.25, format="%.2f")
    omega_pct = st.number_input("Omega Threshold (%)", value=0.0, step=0.25, format="%.2f")

    # Convert to decimal for analytics
    st.session_state["rf_annual"] = rf_pct / 100.0
    st.session_state["mar_annual"] = mar_pct / 100.0
    st.session_state["omega_threshold"] = omega_pct / 100.0

# -- Load data ─────────────────────────────────────────────────────────────────
fund_returns = pd.Series(dtype=float)
benchmark_returns = pd.DataFrame()
peer_returns = pd.DataFrame()

if use_sample:
    fund_returns, benchmark_returns, peer_returns = generate_sample_data()
else:
    if fund_file is not None:
        fund_returns = load_fund_returns(fund_file)
    if bench_file is not None:
        benchmark_returns = load_returns(bench_file)
    if peer_file is not None:
        peer_returns = load_returns(peer_file)

# Align dates if we have fund + benchmark data
if not fund_returns.empty and not benchmark_returns.empty:
    aligned = align_dates(fund_returns, benchmark_returns)
    fund_returns = aligned[0]
    benchmark_returns = aligned[1]

if not fund_returns.empty and not peer_returns.empty:
    aligned = align_dates(fund_returns, peer_returns)
    fund_returns = aligned[0]
    peer_returns = aligned[1]

# Store in session state
st.session_state["fund_returns"] = fund_returns
st.session_state["benchmark_returns"] = benchmark_returns
st.session_state["peer_returns"] = peer_returns
st.session_state["data_loaded"] = not fund_returns.empty

# -- Data summary in sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.markdown("### Data Summary")
    if not fund_returns.empty:
        st.markdown(f"**Period:** {fund_returns.index[0].strftime('%b %Y')} -- "
                    f"{fund_returns.index[-1].strftime('%b %Y')}")
        st.markdown(f"**Observations:** {len(fund_returns)}")
        if not benchmark_returns.empty:
            st.markdown(f"**Benchmarks:** {', '.join(benchmark_returns.columns)}")
        else:
            st.markdown("**Benchmarks:** None")
        if not peer_returns.empty:
            st.markdown(f"**Peers loaded:** {len(peer_returns.columns)}")
        else:
            st.markdown("**Peers loaded:** 0")

        # Validation warnings
        warnings = validate_data(fund_returns)
        if warnings:
            for w in warnings:
                st.warning(w, icon=None)
    else:
        st.info("Upload fund returns or enable Sample Data to begin.")

# -- Navigation ────────────────────────────────────────────────────────────────
pages = [
    st.Page("pages/01_overview.py", title="Overview", icon=None),
    st.Page("pages/02_absolute_return.py", title="Return Measures", icon=None),
    st.Page("pages/03_absolute_risk.py", title="Risk Measures", icon=None),
    st.Page("pages/04_regression.py", title="Regression Analysis", icon=None),
    st.Page("pages/05_rolling.py", title="Rolling Analytics", icon=None),
    st.Page("pages/06_peer_analysis.py", title="Peer Group Analysis", icon=None),
    st.Page("pages/07_drawdown.py", title="Drawdown Analysis", icon=None),
    st.Page("pages/08_calendar.py", title="Calendar Performance", icon=None),
]

nav = st.navigation(pages)
nav.run()
