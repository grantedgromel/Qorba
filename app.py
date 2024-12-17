import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import plotly.subplots as sp
from typing import List, Tuple, Dict

class QorbaAnalyzer:
    def __init__(self, rf_rate: float = 0.02):
        self.rf_rate = rf_rate
        
    def calculate_calendar_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate calendar year returns."""
        data.index = pd.to_datetime(data.index)
        annual_returns = data.groupby(data.index.year).apply(
            lambda x: (1 + x).prod() - 1
        )
        return annual_returns
    
    def calculate_trailing_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trailing period returns."""
        current_date = data.index[-1]
        periods = {
            "1Y": 12,
            "3Y": 36,
            "5Y": 60,
            "10Y": 120,
            "15Y": 180,
            "ITD": len(data)
        }
        
        trailing_returns = {}
        for period, months in periods.items():
            if len(data) >= months:
                returns = (1 + data.iloc[-months:]).prod() ** (12/months) - 1
                trailing_returns[period] = returns
                
        return pd.DataFrame(trailing_returns)
    
    def calculate_rolling_metrics(self, data: pd.DataFrame, window: int, 
                                benchmark_col: str) -> Dict[str, pd.DataFrame]:
        """Calculate rolling performance metrics."""
        rolling_returns = data.rolling(window).apply(
            lambda x: (1 + x).prod() ** (12/window) - 1
        )
        
        rolling_vol = data.rolling(window).std() * np.sqrt(12)
        
        # Calculate rolling beta and alpha
        rolling_beta = data.rolling(window).cov()[benchmark_col] / \
                      data[benchmark_col].rolling(window).var()
        
        rolling_alpha = rolling_returns - (self.rf_rate + 
            rolling_beta * (rolling_returns[benchmark_col] - self.rf_rate))
        
        # Calculate rolling Sharpe ratio
        rolling_sharpe = (rolling_returns - self.rf_rate) / rolling_vol
        
        # Calculate rolling correlation
        rolling_corr = data.rolling(window).corr()[benchmark_col]
        
        # Calculate rolling up/down capture
        benchmark_returns = data[benchmark_col]
        up_months = benchmark_returns > 0
        down_months = benchmark_returns < 0
        
        def calculate_capture(x, condition):
            if len(x[condition]) > 0:
                return x[condition].mean() / benchmark_returns[condition].mean()
            return np.nan
        
        rolling_up_capture = data.rolling(window).apply(
            lambda x: calculate_capture(x, up_months[-window:])
        )
        
        rolling_down_capture = data.rolling(window).apply(
            lambda x: calculate_capture(x, down_months[-window:])
        )
        
        return {
            "returns": rolling_returns,
            "volatility": rolling_vol,
            "alpha": rolling_alpha,
            "beta": rolling_beta,
            "sharpe": rolling_sharpe,
            "correlation": rolling_corr,
            "up_capture": rolling_up_capture,
            "down_capture": rolling_down_capture
        }
    
    def plot_cumulative_returns(self, data: pd.DataFrame) -> go.Figure:
        """Plot cumulative returns with percentage y-axis."""
        cum_returns = (1 + data).cumprod()
        
        fig = go.Figure()
        colors = px.colors.qualitative.Set1
        
        for i, column in enumerate(cum_returns.columns):
            fig.add_trace(
                go.Scatter(
                    x=cum_returns.index,
                    y=cum_returns[column] * 100 - 100,  # Convert to percentage
                    name=column,
                    mode='lines',
                    line=dict(color=colors[i % len(colors)])
                )
            )
            
        fig.update_layout(
            title='Cumulative Returns',
            xaxis_title='Date',
            yaxis_title='Cumulative Return (%)',
            yaxis_tickformat=',.1f',
            height=600,
            showlegend=True
        )
        
        return fig

    def plot_monthly_returns(self, data: pd.DataFrame) -> go.Figure:
        """Plot monthly returns as a bar chart."""
        fig = go.Figure()
        colors = px.colors.qualitative.Set1
        
        for i, column in enumerate(data.columns):
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data[column] * 100,  # Convert to percentage
                    name=column,
                    marker_color=colors[i % len(colors)]
                )
            )
            
        fig.update_layout(
            title='Monthly Returns',
            xaxis_title='Date',
            yaxis_title='Return (%)',
            yaxis_tickformat=',.1f',
            height=600,
            barmode='group'
        )
        
        return fig

    def plot_risk_return(self, data: pd.DataFrame, periods: List[str]) -> go.Figure:
        """Plot risk-return scatter plot for multiple periods."""
        fig = go.Figure()
        colors = px.colors.qualitative.Set1
        
        for period in periods:
            if period == "ITD":
                period_data = data
            else:
                months = int(period[0]) * 12
                if len(data) >= months:
                    period_data = data.iloc[-months:]
                else:
                    continue
                    
            returns = (1 + period_data).prod() ** (12/len(period_data)) - 1
            vol = period_data.std() * np.sqrt(12)
            
            for i, column in enumerate(data.columns):
                fig.add_trace(
                    go.Scatter(
                        x=[vol[column] * 100],
                        y=[returns[column] * 100],
                        name=f"{column} ({period})",
                        mode='markers',
                        marker=dict(
                            size=12,
                            color=colors[i % len(colors)]
                        )
                    )
                )
                
        fig.update_layout(
            title='Risk-Return Analysis',
            xaxis_title='Annualized Volatility (%)',
            yaxis_title='Annualized Return (%)',
            height=600,
            showlegend=True
        )
        
        return fig

    def plot_correlation_matrix(self, data: pd.DataFrame) -> go.Figure:
        """Plot correlation matrix with numbers."""
        corr_matrix = data.corr()
        
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                text=np.round(corr_matrix * 100, 1),
                texttemplate='%{text:.1f}%',
                textfont={"size": 10},
                colorscale='RdBu_r',  # Reversed RdBu to make red=high, blue=low
                zmin=-1,
                zmax=1
            )
        )
        
        fig.update_layout(
            title='Correlation Matrix',
            height=600
        )
        
        return fig

    def plot_drawdowns(self, data: pd.DataFrame, benchmark_col: str) -> Tuple[go.Figure, go.Figure]:
        """Plot absolute and relative drawdowns."""
        # Absolute drawdowns
        cum_returns = (1 + data).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        
        fig_abs = go.Figure()
        colors = px.colors.qualitative.Set1
        
        for i, column in enumerate(drawdowns.columns):
            fig_abs.add_trace(
                go.Scatter(
                    x=drawdowns.index,
                    y=drawdowns[column] * 100,
                    name=column,
                    mode='lines',
                    line=dict(color=colors[i % len(colors)])
                )
            )
            
        fig_abs.update_layout(
            title='Absolute Drawdowns',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            height=400
        )
        
        # Relative drawdowns vs primary benchmark
        relative_cum_returns = cum_returns.div(cum_returns[benchmark_col], axis=0)
        relative_rolling_max = relative_cum_returns.expanding().max()
        relative_drawdowns = (relative_cum_returns - relative_rolling_max) / relative_rolling_max
        
        fig_rel = go.Figure()
        
        for i, column in enumerate(relative_drawdowns.columns):
            if column != benchmark_col:
                fig_rel.add_trace(
                    go.Scatter(
                        x=relative_drawdowns.index,
                        y=relative_drawdowns[column] * 100,
                        name=column,
                        mode='lines',
                        line=dict(color=colors[i % len(colors)])
                    )
                )
                
        fig_rel.update_layout(
            title=f'Relative Drawdowns vs {benchmark_col}',
            xaxis_title='Date',
            yaxis_title='Relative Drawdown (%)',
            height=400
        )
        
        return fig_abs, fig_rel

def main():
    st.set_page_config(page_title="Qorba", layout="wide")
    
    st.title("Qorba")
    st.markdown("""
    ### A Quantitative Return Analysis Tool
    
    Qorba (قربة) is a sophisticated return analysis tool designed to provide comprehensive performance analytics 
    for investment managers. The name pays homage to the traditional Middle Eastern soup dish, reflecting the 
    tool's ability to blend various performance metrics into a cohesive analytical "soup" - while also being 
    a play on QoR (Quality of Returns) Analysis.
    
    ### Instructions
    1. Prepare your CSV file with monthly returns data:
        - Column 1: Manager returns
        - Column 2: Primary benchmark
        - Column 3: Secondary benchmark
        - Column 4+: Additional peers or indexes
    2. All returns should be in decimal form (e.g., 0.05 for 5%)
    3. Dates should be in the first column in YYYY-MM-DD format
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load and validate data
            data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
            
            # Display analysis date and benchmarks
            st.markdown(f"""
            **Analysis as of:** {data.index[-1].strftime('%B %Y')}  
            **Primary Benchmark:** {data.columns[1]}  
            **Secondary Benchmark:** {data.columns[2]}
            """)
            
            # Initialize analyzer
            analyzer = QorbaAnalyzer()
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs([
                "Returns Analysis", 
                "Risk Metrics", 
                "Rolling Analysis",
                "Drawdown Analysis"
            ])
            
            with tab1:
                # Calendar year returns
                st.subheader("Calendar Year Returns")
                calendar_returns = analyzer.calculate_calendar_returns(data)
                st.dataframe(calendar_returns.style.format("{:.2%}"))
                
                # Trailing returns
                st.subheader("Trailing Returns")
                trailing_returns = analyzer.calculate_trailing_returns(data)
                st.dataframe(trailing_returns.style.format("{:.2%}"))
                
                # Monthly returns chart
                st.plotly_chart(analyzer.plot_monthly_returns(data), use_container_width=True)
                
                # Cumulative returns chart
                st.plotly_chart(analyzer.plot_cumulative_returns(data), use_container_width=True)
            
            with tab2:
                # Risk-return plot
                periods = ["1Y", "3Y", "5Y", "ITD"]
                st.plotly_chart(
                    analyzer.plot_risk_return(data, periods),
                    use_container_width=True
                )
                
                # Correlation matrix
                st.plotly_chart(
                    analyzer.plot_correlation_matrix(data),
                    use_container_width=True
                )
            
            with tab3:
                # Rolling period selection
                rolling_window = st.selectbox(
                    "Select rolling period",
                    ["1-Year", "3-Year", "5-Year"],
                    index=1
                )
                window = int(rolling_window[0]) * 12
                
                # Calculate rolling metrics
                rolling_metrics = analyzer.calculate_rolling_metrics(
                    data, window, data.columns[1]
                )
                
                # Display rolling metrics charts
                metrics_to_plot = [
                    ("returns", "Rolling Returns"),
                    ("volatility", "Rolling Volatility"),
                    ("alpha", "Rolling Alpha"),
                    ("beta", "Rolling Beta"),
                    ("sharpe", "Rolling Sharpe Ratio"),
                    ("correlation", "Rolling Correlation"),
                    ("up_capture", "Rolling Up Capture"),
                    ("down_capture", "Rolling Down Capture")
                ]
                
                for metric, title in metrics_to_plot:
                    fig = go.Figure()
                    for column in data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=rolling_metrics[metric].index,
                                y=rolling_metrics[metric][column],
                                name=column,
                                mode='lines'
                            )
                        )
                    fig.update_layout(
                        title=f"{title} ({rolling_window})",
                        xaxis_title="Date",
                        yaxis_title=title,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                # Drawdown analysis
                fig_abs, fig_rel = analyzer.plot_drawdowns(data, data.columns[1])
                st.plotly_chart(fig_abs, use_container_width=True)
                st.plotly_chart(fig_rel, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.markdown("""
            Please check:
            - Date format (YYYY-MM-DD)
            - Returns in decimal form
            - Required columns present
            """)

if __name__ == "__main__":
    main()
