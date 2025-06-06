import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import plotly.subplots as sp
from typing import List, Tuple, Dict, Optional
from scipy import stats

class PerformanceMetrics:
    """Calculate various performance and risk metrics."""
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, rf_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - rf_rate/12  # Monthly risk-free rate
        return np.sqrt(12) * excess_returns.mean() / returns.std()
    
    @staticmethod
    def sortino_ratio(returns: pd.Series, rf_rate: float = 0.02, mar: float = 0) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - rf_rate/12
        downside_returns = returns[returns < mar]
        downside_deviation = np.sqrt((downside_returns**2).mean()) * np.sqrt(12)
        return np.sqrt(12) * excess_returns.mean() / downside_deviation if downside_deviation > 0 else np.nan
    
    @staticmethod
    def information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate Information ratio."""
        active_returns = returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(12)
        return np.sqrt(12) * active_returns.mean() / tracking_error if tracking_error > 0 else np.nan
    
    @staticmethod
    def calmar_ratio(returns: pd.Series, periods: int = 36) -> float:
        """Calculate Calmar ratio (3-year)."""
        recent_returns = returns.iloc[-periods:] if len(returns) >= periods else returns
        cum_returns = (1 + recent_returns).cumprod()
        ann_return = (cum_returns.iloc[-1] ** (12/len(recent_returns))) - 1
        max_dd = PerformanceMetrics.max_drawdown(recent_returns)
        return ann_return / abs(max_dd) if max_dd != 0 else np.nan
    
    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        return drawdowns.min()
    
    @staticmethod
    def downside_deviation(returns: pd.Series, mar: float = 0) -> float:
        """Calculate downside deviation."""
        downside_returns = returns[returns < mar]
        return np.sqrt((downside_returns**2).mean()) * np.sqrt(12)
    
    @staticmethod
    def skewness(returns: pd.Series) -> float:
        """Calculate skewness."""
        return stats.skew(returns)
    
    @staticmethod
    def kurtosis(returns: pd.Series) -> float:
        """Calculate excess kurtosis."""
        return stats.kurtosis(returns)
    
    @staticmethod
    def omega_ratio(returns: pd.Series, threshold: float = 0) -> float:
        """Calculate Omega ratio."""
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        return gains.sum() / losses.sum() if losses.sum() > 0 else np.nan

class QorbaAnalyzer:
    def __init__(self, rf_rate: float = 0.02):
        self.rf_rate = rf_rate
        self.metrics = PerformanceMetrics()
        
    def calculate_summary_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive summary statistics."""
        stats_dict = {}
        
        for column in data.columns:
            returns = data[column]
            ann_return = (1 + returns).prod() ** (12/len(returns)) - 1
            ann_vol = returns.std() * np.sqrt(12)
            
            stats_dict[column] = {
                'Annualized Return': ann_return,
                'Annualized Volatility': ann_vol,
                'Sharpe Ratio': self.metrics.sharpe_ratio(returns, self.rf_rate),
                'Sortino Ratio': self.metrics.sortino_ratio(returns, self.rf_rate),
                'Max Drawdown': self.metrics.max_drawdown(returns),
                'Calmar Ratio': self.metrics.calmar_ratio(returns),
                'Downside Deviation': self.metrics.downside_deviation(returns),
                'Skewness': self.metrics.skewness(returns),
                'Excess Kurtosis': self.metrics.kurtosis(returns),
                'Best Month': returns.max(),
                'Worst Month': returns.min(),
                'Positive Months %': (returns > 0).sum() / len(returns)
            }
            
            # Add Information Ratio relative to primary benchmark
            if column != data.columns[1]:  # Not the benchmark itself
                stats_dict[column]['Information Ratio'] = self.metrics.information_ratio(
                    returns, data[data.columns[1]]
                )
        
        return pd.DataFrame(stats_dict).T
    
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
            "1M": 1,
            "3M": 3,
            "6M": 6,
            "1Y": 12,
            "3Y": 36,
            "5Y": 60,
            "10Y": 120,
            "ITD": len(data)
        }
        
        trailing_returns = {}
        for period, months in periods.items():
            if len(data) >= months:
                period_returns = data.iloc[-months:]
                ann_factor = 12 / months if months < 12 else 12 / months
                returns = {}
                for col in data.columns:
                    if months == 1:
                        returns[col] = period_returns[col].iloc[-1]
                    else:
                        returns[col] = (1 + period_returns[col]).prod() ** ann_factor - 1
                trailing_returns[period] = returns
                
        return pd.DataFrame(trailing_returns)
    
    def calculate_rolling_metrics(self, data: pd.DataFrame, window: int, 
                                benchmark_col: str) -> Dict[str, pd.DataFrame]:
        """Calculate rolling performance metrics with proper handling."""
        metrics = {}
        
        # Rolling returns (annualized)
        metrics['returns'] = data.rolling(window).apply(
            lambda x: (1 + x).prod() ** (12/window) - 1
        )
        
        # Rolling volatility (annualized)
        metrics['volatility'] = data.rolling(window).std() * np.sqrt(12)
        
        # Rolling Sharpe ratio
        metrics['sharpe'] = data.rolling(window).apply(
            lambda x: self.metrics.sharpe_ratio(x, self.rf_rate)
        )
        
        # Rolling correlation
        metrics['correlation'] = pd.DataFrame(index=data.index, columns=data.columns)
        for col in data.columns:
            metrics['correlation'][col] = data[col].rolling(window).corr(data[benchmark_col])
        
        # Rolling beta
        metrics['beta'] = pd.DataFrame(index=data.index, columns=data.columns)
        for col in data.columns:
            cov = data[col].rolling(window).cov(data[benchmark_col])
            var = data[benchmark_col].rolling(window).var()
            metrics['beta'][col] = cov / var
        
        # Rolling alpha (annualized)
        metrics['alpha'] = metrics['returns'] - (self.rf_rate + 
            metrics['beta'] * (metrics['returns'][benchmark_col] - self.rf_rate))
        
        # Rolling up/down capture
        metrics['up_capture'] = pd.DataFrame(index=data.index, columns=data.columns)
        metrics['down_capture'] = pd.DataFrame(index=data.index, columns=data.columns)
        
        for i in range(window-1, len(data)):
            window_data = data.iloc[i-window+1:i+1]
            benchmark_window = window_data[benchmark_col]
            
            up_months = benchmark_window > 0
            down_months = benchmark_window <= 0
            
            for col in data.columns:
                if up_months.sum() > 0:
                    metrics['up_capture'][col].iloc[i] = (
                        window_data[col][up_months].mean() / 
                        benchmark_window[up_months].mean()
                    )
                if down_months.sum() > 0:
                    metrics['down_capture'][col].iloc[i] = (
                        window_data[col][down_months].mean() / 
                        benchmark_window[down_months].mean()
                    )
        
        # Rolling max drawdown
        metrics['max_drawdown'] = data.rolling(window).apply(
            lambda x: self.metrics.max_drawdown(x)
        )
        
        return metrics
    
    def plot_cumulative_returns(self, data: pd.DataFrame) -> go.Figure:
        """Plot cumulative returns with percentage y-axis."""
        cum_returns = (1 + data).cumprod()
        
        fig = go.Figure()
        colors = px.colors.qualitative.Set1
        
        for i, column in enumerate(cum_returns.columns):
            fig.add_trace(
                go.Scatter(
                    x=cum_returns.index,
                    y=(cum_returns[column] - 1) * 100,  # Convert to percentage
                    name=column,
                    mode='lines',
                    line=dict(color=colors[i % len(colors)], width=2)
                )
            )
            
        fig.update_layout(
            title='Cumulative Returns',
            xaxis_title='Date',
            yaxis_title='Cumulative Return (%)',
            yaxis_tickformat=',.0f',
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig

    def plot_returns_distribution(self, data: pd.DataFrame) -> go.Figure:
        """Plot returns distribution histogram."""
        fig = go.Figure()
        colors = px.colors.qualitative.Set1
        
        for i, column in enumerate(data.columns):
            fig.add_trace(
                go.Histogram(
                    x=data[column] * 100,
                    name=column,
                    opacity=0.7,
                    nbinsx=30,
                    marker_color=colors[i % len(colors)]
                )
            )
            
        fig.update_layout(
            title='Returns Distribution',
            xaxis_title='Monthly Return (%)',
            yaxis_title='Frequency',
            height=400,
            barmode='overlay'
        )
        
        return fig

    def plot_risk_return(self, data: pd.DataFrame) -> go.Figure:
        """Enhanced risk-return scatter plot."""
        stats = self.calculate_summary_statistics(data)
        
        fig = go.Figure()
        colors = px.colors.qualitative.Set1
        
        # Add efficient frontier line (simplified)
        min_vol = stats['Annualized Volatility'].min()
        max_vol = stats['Annualized Volatility'].max()
        vol_range = np.linspace(min_vol * 0.8, max_vol * 1.2, 100)
        
        # Add scatter points
        for i, (index, row) in enumerate(stats.iterrows()):
            fig.add_trace(
                go.Scatter(
                    x=[row['Annualized Volatility'] * 100],
                    y=[row['Annualized Return'] * 100],
                    name=index,
                    mode='markers+text',
                    marker=dict(
                        size=15,
                        color=colors[i % len(colors)]
                    ),
                    text=[index],
                    textposition="top center"
                )
            )
        
        # Add CAL line from risk-free rate
        max_sharpe_idx = stats['Sharpe Ratio'].idxmax()
        max_sharpe_vol = stats.loc[max_sharpe_idx, 'Annualized Volatility']
        max_sharpe_ret = stats.loc[max_sharpe_idx, 'Annualized Return']
        
        cal_x = [0, max_sharpe_vol * 100 * 1.5]
        cal_y = [self.rf_rate * 100, self.rf_rate * 100 + stats.loc[max_sharpe_idx, 'Sharpe Ratio'] * max_sharpe_vol * 100 * 1.5]
        
        fig.add_trace(
            go.Scatter(
                x=cal_x,
                y=cal_y,
                name='Capital Allocation Line',
                mode='lines',
                line=dict(dash='dash', color='gray'),
                showlegend=True
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

    def plot_drawdown_analysis(self, data: pd.DataFrame) -> go.Figure:
        """Enhanced drawdown analysis with statistics."""
        cum_returns = (1 + data).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        
        # Create subplots
        fig = sp.make_subplots(
            rows=2, cols=1,
            subplot_titles=('Cumulative Returns', 'Drawdowns'),
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4]
        )
        
        colors = px.colors.qualitative.Set1
        
        # Plot cumulative returns
        for i, column in enumerate(cum_returns.columns):
            fig.add_trace(
                go.Scatter(
                    x=cum_returns.index,
                    y=(cum_returns[column] - 1) * 100,
                    name=column,
                    mode='lines',
                    line=dict(color=colors[i % len(colors)]),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Plot drawdowns
        for i, column in enumerate(drawdowns.columns):
            fig.add_trace(
                go.Scatter(
                    x=drawdowns.index,
                    y=drawdowns[column] * 100,
                    name=column,
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color=colors[i % len(colors)]),
                    showlegend=False
                ),
                row=2, col=1
            )
            
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        fig.update_layout(height=800, title_text="Drawdown Analysis")
        
        return fig

    def calculate_drawdown_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate detailed drawdown statistics."""
        stats_dict = {}
        
        for column in data.columns:
            returns = data[column]
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = (cum_returns - rolling_max) / rolling_max
            
            # Find all drawdown periods
            dd_start = []
            dd_end = []
            in_dd = False
            
            for i in range(len(drawdowns)):
                if not in_dd and drawdowns.iloc[i] < 0:
                    dd_start.append(i)
                    in_dd = True
                elif in_dd and (drawdowns.iloc[i] == 0 or i == len(drawdowns) - 1):
                    dd_end.append(i)
                    in_dd = False
            
            # Calculate statistics
            max_dd = drawdowns.min()
            max_dd_idx = drawdowns.idxmin()
            
            # Find the start of the max drawdown
            max_dd_start_idx = max_dd_idx
            for i in range(drawdowns.index.get_loc(max_dd_idx), -1, -1):
                if drawdowns.iloc[i] == 0:
                    max_dd_start_idx = drawdowns.index[i+1] if i+1 < len(drawdowns) else drawdowns.index[i]
                    break
            
            # Recovery time
            recovery_idx = None
            max_dd_loc = drawdowns.index.get_loc(max_dd_idx)
            for i in range(max_dd_loc, len(drawdowns)):
                if drawdowns.iloc[i] == 0:
                    recovery_idx = drawdowns.index[i]
                    break
            
            stats_dict[column] = {
                'Max Drawdown': max_dd,
                'Max DD Start': max_dd_start_idx,
                'Max DD End': max_dd_idx,
                'Recovery Date': recovery_idx if recovery_idx else 'Not Recovered',
                'Current Drawdown': drawdowns.iloc[-1],
                'Number of Drawdowns': len(dd_start),
                'Average Drawdown': drawdowns[drawdowns < 0].mean() if len(drawdowns[drawdowns < 0]) > 0 else 0
            }
        
        return pd.DataFrame(stats_dict).T

def main():
    st.set_page_config(page_title="Qorba - Enhanced", layout="wide")
    
    st.title("Qorba - Enhanced Quality of Returns Analysis")
    st.markdown("""
    ### Comprehensive Performance Analytics Tool
    
    This enhanced version of Qorba provides institutional-grade performance analytics including:
    - Additional risk metrics (Sortino, Information Ratio, Calmar, etc.)
    - Improved rolling analysis with proper calculations
    - Enhanced visualizations
    - Detailed drawdown statistics
    
    ### Instructions
    1. Upload a CSV file with monthly returns data
    2. First column: Date (YYYY-MM-DD format)
    3. Subsequent columns: Manager and benchmark returns (in decimal form)
    """)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        rf_rate = st.number_input("Risk-Free Rate (Annual)", value=0.02, format="%.4f")
        mar = st.number_input("Minimum Acceptable Return", value=0.0, format="%.4f")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load and validate data
            data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
            
            # Validate data
            if len(data.columns) < 2:
                st.error("Please provide at least one manager and one benchmark column")
                return
                
            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Analysis Period", f"{data.index[0].strftime('%b %Y')} - {data.index[-1].strftime('%b %Y')}")
            with col2:
                st.metric("Total Months", len(data))
            with col3:
                st.metric("Primary Benchmark", data.columns[1])
            
            # Initialize analyzer
            analyzer = QorbaAnalyzer(rf_rate=rf_rate)
            
            # Create tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Summary Statistics", 
                "Returns Analysis", 
                "Risk Analysis",
                "Rolling Analysis",
                "Drawdown Analysis"
            ])
            
            with tab1:
                st.subheader("Comprehensive Performance Statistics")
                summary_stats = analyzer.calculate_summary_statistics(data)
                
                # Format the statistics nicely
                formatted_stats = summary_stats.copy()
                pct_cols = ['Annualized Return', 'Annualized Volatility', 'Max Drawdown', 
                           'Downside Deviation', 'Best Month', 'Worst Month', 'Positive Months %']
                ratio_cols = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Information Ratio']
                
                for col in pct_cols:
                    if col in formatted_stats.columns:
                        formatted_stats[col] = formatted_stats[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")
                
                for col in ratio_cols:
                    if col in formatted_stats.columns:
                        formatted_stats[col] = formatted_stats[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
                
                for col in ['Skewness', 'Excess Kurtosis']:
                    if col in formatted_stats.columns:
                        formatted_stats[col] = formatted_stats[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")
                
                st.dataframe(formatted_stats)
                
                # Risk-Return scatter
                st.plotly_chart(analyzer.plot_risk_return(data), use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Calendar Year Returns")
                    calendar_returns = analyzer.calculate_calendar_returns(data)
                    st.dataframe(calendar_returns.style.format("{:.2%}"))
                
                with col2:
                    st.subheader("Trailing Returns")
                    trailing_returns = analyzer.calculate_trailing_returns(data)
                    st.dataframe(trailing_returns.T.style.format("{:.2%}"))
                
                # Cumulative returns chart
                st.plotly_chart(analyzer.plot_cumulative_returns(data), use_container_width=True)
                
                # Returns distribution
                st.plotly_chart(analyzer.plot_returns_distribution(data), use_container_width=True)
            
            with tab3:
                # Correlation matrix
                st.subheader("Correlation Matrix")
                corr_matrix = data.corr()
                
                fig = go.Figure(
                    data=go.Heatmap(
                        z=corr_matrix,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        text=np.round(corr_matrix, 2),
                        texttemplate='%{text}',
                        textfont={"size": 10},
                        colorscale='RdBu_r',
                        zmin=-1,
                        zmax=1
                    )
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional risk metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Monthly Return Statistics")
                    monthly_stats = pd.DataFrame({
                        'Mean': data.mean() * 100,
                        'Std Dev': data.std() * 100,
                        'Min': data.min() * 100,
                        'Max': data.max() * 100,
                        'Skew': data.skew(),
                        'Kurtosis': data.kurtosis()
                    })
                    st.dataframe(monthly_stats.style.format("{:.2f}"))
                
                with col2:
                    st.subheader("Value at Risk (95% confidence)")
                    var_95 = data.quantile(0.05) * 100
                    var_df = pd.DataFrame({'VaR 95%': var_95})
                    st.dataframe(var_df.style.format("{:.2f}%"))
            
            with tab4:
                # Rolling period selection
                rolling_window = st.selectbox(
                    "Select rolling period",
                    ["1-Year", "3-Year", "5-Year"],
                    index=1
                )
                window = int(rolling_window[0]) * 12
                
                if len(data) >= window:
                    # Calculate rolling metrics
                    rolling_metrics = analyzer.calculate_rolling_metrics(
                        data, window, data.columns[1]
                    )
                    
                    # Create a 2x2 grid of charts
                    metrics_to_plot = [
                        ("returns", "Rolling Returns (%)", True),
                        ("volatility", "Rolling Volatility (%)", True),
                        ("sharpe", "Rolling Sharpe Ratio", False),
                        ("max_drawdown", "Rolling Max Drawdown (%)", True),
                        ("beta", "Rolling Beta", False),
                        ("alpha", "Rolling Alpha (%)", True)
                    ]
                    
                    for i in range(0, len(metrics_to_plot), 2):
                        col1, col2 = st.columns(2)
                        
                        for j, col in enumerate([col1, col2]):
                            if i + j < len(metrics_to_plot):
                                metric, title, is_pct = metrics_to_plot[i + j]
                                with col:
                                    fig = go.Figure()
                                    for column in data.columns[:3]:  # Limit to first 3 for clarity
                                        y_data = rolling_metrics[metric][column]
                                        if is_pct:
                                            y_data = y_data * 100
                                        fig.add_trace(
                                            go.Scatter(
                                                x=rolling_metrics[metric].index,
                                                y=y_data,
                                                name=column,
                                                mode='lines'
                                            )
                                        )
                                    fig.update_layout(
                                        title=f"{title} ({rolling_window})",
                                        xaxis_title="Date",
                                        yaxis_title=title.split("(")[0].strip(),
                                        height=400
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Insufficient data for {rolling_window} rolling analysis")
            
            with tab5:
                # Drawdown chart
                st.plotly_chart(analyzer.plot_drawdown_analysis(data), use_container_width=True)
                
                # Drawdown statistics table
                st.subheader("Drawdown Statistics")
                dd_stats = analyzer.calculate_drawdown_statistics(data)
                
                # Format the statistics
                for col in ['Max Drawdown', 'Current Drawdown', 'Average Drawdown']:
                    if col in dd_stats.columns:
                        dd_stats[col] = dd_stats[col].apply(lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else x)
                
                st.dataframe(dd_stats)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.markdown("""
            Please ensure your CSV file:
            - Has dates in the first column (YYYY-MM-DD format)
            - Contains returns in decimal form (0.05 for 5%)
            - Has column headers for each return series
            """)

if __name__ == "__main__":
    main()
