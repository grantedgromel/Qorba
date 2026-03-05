"""Calendar Performance page -- monthly heatmap, calendar year returns, capture ratios."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from config.settings import CHART_COLORS, COLORS
from analytics.performance_tables import (
    monthly_returns_table,
    calendar_year_returns,
    annualized_performance_table,
    up_down_capture,
    best_worst_periods,
)

st.title("Calendar Performance")

# -- Guard ─────────────────────────────────────────────────────────────────────
fund_returns: pd.Series = st.session_state.get("fund_returns", pd.Series(dtype=float))
benchmark_returns: pd.DataFrame = st.session_state.get("benchmark_returns", pd.DataFrame())

if fund_returns.empty:
    st.info("Please upload fund return data or enable Sample Data in the sidebar to get started.")
    st.stop()

has_benchmarks = not benchmark_returns.empty

# -- Monthly returns heatmap ───────────────────────────────────────────────────
st.subheader("Monthly Returns Heatmap")

monthly_tbl = monthly_returns_table(fund_returns)

# Build annotated heatmap
months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "YTD"]
cols_present = [m for m in months_order if m in monthly_tbl.columns]
z_data = monthly_tbl[cols_present].values
years = monthly_tbl.index.tolist()

# Dark-mode diverging colorscale: red -> dark bg -> green
colorscale = [
    [0.0, COLORS["red"]],
    [0.5, COLORS["bg_card"]],
    [1.0, COLORS["green"]],
]

# Determine symmetric range
max_abs = np.nanmax(np.abs(z_data)) if not np.all(np.isnan(z_data)) else 0.1

# Annotations
annotations_text = []
for row in z_data:
    row_text = []
    for val in row:
        if pd.isna(val):
            row_text.append("")
        else:
            row_text.append(f"{val:.1%}")
    annotations_text.append(row_text)

fig_heat = go.Figure(data=go.Heatmap(
    z=z_data,
    x=cols_present,
    y=[str(y) for y in years],
    colorscale=colorscale,
    zmid=0,
    zmin=-max_abs,
    zmax=max_abs,
    text=annotations_text,
    texttemplate="%{text}",
    textfont=dict(size=11, color=COLORS["text_primary"]),
    hovertemplate="Year %{y}, %{x}: %{text}<extra></extra>",
    showscale=True,
    colorbar=dict(title="Return", tickformat=".0%", tickfont=dict(color=COLORS["text_secondary"])),
))
fig_heat.update_layout(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    height=max(300, 50 * len(years) + 80),
    margin=dict(l=60, r=20, t=20, b=40),
    yaxis=dict(autorange="reversed"),
    font=dict(color=COLORS["text_secondary"]),
)
st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# -- Calendar year returns bar chart ───────────────────────────────────────────
st.subheader("Calendar Year Returns")

fund_cy = calendar_year_returns(fund_returns)

fig_cy = go.Figure()
fig_cy.add_trace(go.Bar(
    x=[str(y) for y in fund_cy.index],
    y=fund_cy.values,
    name="Fund",
    marker_color=CHART_COLORS[0],
    marker_line_color=COLORS["border"],
    marker_line_width=0.5,
    text=[f"{v:.1%}" for v in fund_cy.values],
    textposition="outside",
    textfont=dict(color=COLORS["text_secondary"]),
))

if has_benchmarks:
    for i, col in enumerate(benchmark_returns.columns):
        bm_cy = calendar_year_returns(benchmark_returns[col])
        # Align years
        common_years = fund_cy.index.intersection(bm_cy.index)
        fig_cy.add_trace(go.Bar(
            x=[str(y) for y in common_years],
            y=bm_cy.loc[common_years].values,
            name=col,
            marker_color=CHART_COLORS[(i + 1) % len(CHART_COLORS)],
            marker_line_color=COLORS["border"],
            marker_line_width=0.5,
            text=[f"{v:.1%}" for v in bm_cy.loc[common_years].values],
            textposition="outside",
            textfont=dict(color=COLORS["text_secondary"]),
        ))

fig_cy.update_layout(
    barmode="group",
    yaxis_title="Return",
    xaxis_title="Year",
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    height=400,
    margin=dict(l=40, r=20, t=30, b=40),
    yaxis=dict(tickformat=".0%"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    font=dict(color=COLORS["text_secondary"]),
)
st.plotly_chart(fig_cy, use_container_width=True)

st.divider()

# -- Annualized performance table ──────────────────────────────────────────────
st.subheader("Annualized Performance")

ann_tbl = annualized_performance_table(
    fund_returns,
    benchmark_returns if has_benchmarks else None,
)

if not ann_tbl.empty:
    display_ann = ann_tbl.copy()
    for col in display_ann.columns:
        display_ann[col] = display_ann[col].apply(
            lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
        )
    st.dataframe(display_ann, use_container_width=True)
else:
    st.info("Not enough data for annualized performance calculations.")

st.divider()

# -- Up/Down Capture Ratios ───────────────────────────────────────────────────
if has_benchmarks:
    st.subheader("Up/Down Capture Ratios")

    capture_rows = []
    for col in benchmark_returns.columns:
        cap = up_down_capture(fund_returns, benchmark_returns[col])
        capture_rows.append({
            "Benchmark": col,
            "Up Capture": cap["Up Capture"],
            "Down Capture": cap["Down Capture"],
            "Up Months": cap["Up months"],
            "Down Months": cap["Down months"],
        })
    cap_df = pd.DataFrame(capture_rows)

    # Display table
    display_cap = cap_df.copy()
    display_cap["Up Capture"] = display_cap["Up Capture"].apply(lambda x: f"{x:.2%}")
    display_cap["Down Capture"] = display_cap["Down Capture"].apply(lambda x: f"{x:.2%}")
    st.dataframe(display_cap.set_index("Benchmark"), use_container_width=True)

    # Bar chart
    fig_cap = go.Figure()
    fig_cap.add_trace(go.Bar(
        x=cap_df["Benchmark"],
        y=cap_df["Up Capture"],
        name="Up Capture",
        marker_color=COLORS["green"],
        marker_line_color=COLORS["border"],
        marker_line_width=0.5,
        text=[f"{v:.0%}" for v in cap_df["Up Capture"]],
        textposition="outside",
        textfont=dict(color=COLORS["text_secondary"]),
    ))
    fig_cap.add_trace(go.Bar(
        x=cap_df["Benchmark"],
        y=cap_df["Down Capture"],
        name="Down Capture",
        marker_color=COLORS["red"],
        marker_line_color=COLORS["border"],
        marker_line_width=0.5,
        text=[f"{v:.0%}" for v in cap_df["Down Capture"]],
        textposition="outside",
        textfont=dict(color=COLORS["text_secondary"]),
    ))
    fig_cap.update_layout(
        barmode="group",
        yaxis_title="Capture Ratio",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(l=40, r=20, t=30, b=40),
        yaxis=dict(tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(color=COLORS["text_secondary"]),
    )
    st.plotly_chart(fig_cap, use_container_width=True)

    st.divider()

# -- Best / Worst Period Stats ─────────────────────────────────────────────────
st.subheader("Best / Worst Periods")

bw = best_worst_periods(fund_returns)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Best Month", f"{bw['Best Month']:.2%}", help=bw["Best Month Date"])
c2.metric("Worst Month", f"{bw['Worst Month']:.2%}", help=bw["Worst Month Date"])
c3.metric("Best Quarter", f"{bw['Best Quarter']:.2%}")
c4.metric("Worst Quarter", f"{bw['Worst Quarter']:.2%}")

c5, c6, c7 = st.columns(3)
c5.metric("Positive Months", str(bw["Positive Months"]))
c6.metric("Negative Months", str(bw["Negative Months"]))
c7.metric("Total Months", str(bw["Total Months"]))
