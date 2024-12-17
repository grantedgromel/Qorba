import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

class HedgeFundAnalyzer:
    def __init__(self):
        self.rf_rate = 0.02  # Assuming 2% risk-free rate
        
    def calculate_statistics(self, data):
        stats_dict = {}
        
        for column in data.columns:
            returns = data[column]
            
            # Basic statistics
            annualized_return = returns.mean() * 12
            volatility = returns.std() * np.sqrt(12)
            sharpe_ratio = (annualized_return - self.rf_rate) / volatility
            
            # Drawdown analysis
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = (cum_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # Gain/Loss analysis
            gains = returns[returns > 0].mean()
            losses = returns[returns < 0].mean()
            gain_loss_ratio = abs(gains/losses) if losses != 0 else np.inf
            
            stats_dict[column] = {
                'Annualized Return': f"{annualized_return:.2%}",
                'Volatility': f"{volatility:.2%}",
                'Sharpe Ratio': f"{sharpe_ratio:.2f}",
                'Max Drawdown': f"{max_drawdown:.2%}",
                'Gain/Loss Ratio': f"{gain_loss_ratio:.2f}"
            }
            
        return pd.DataFrame(stats_dict)

    def plot_cumulative_returns(self, data):
        cum_returns = (1 + data).cumprod()
        
        fig = go.Figure()
        for column in cum_returns.columns:
            fig.add_trace(
                go.Scatter(
                    x=cum_returns.index,
                    y=cum_returns[column],
                    name=column,
                    mode='lines'
                )
            )
            
        fig.update_layout(
            title='Cumulative Returns',
            xaxis_title='Date',
            yaxis_title='Cumulative Return',
            height=600
        )
        
        return fig

    def plot_returns_distribution(self, data):
        fig = go.Figure()
        for column in data.columns:
            fig.add_trace(
                go.Histogram(
                    x=data[column],
                    name=column,
                    opacity=0.7,
                    nbinsx=30
                )
            )
            
        fig.update_layout(
            title='Monthly Returns Distribution',
            xaxis_title='Monthly Return',
            yaxis_title='Frequency',
            barmode='overlay',
            height=600
        )
        
        return fig

    def plot_correlation_heatmap(self, data):
        corr_matrix = data.corr()
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        
        fig.update_layout(
            title='Correlation Matrix',
            height=600
        )
        
        return fig

def main():
    st.set_page_config(page_title="Hedge Fund Return Analyzer", layout="wide")
    
    st.title("Hedge Fund Return Analyzer")
    
    st.markdown("""
    ### Instructions
    1. Prepare your CSV file with monthly returns data:
        - First column should be named 'Date' in YYYY-MM-DD format
        - Returns should be in decimal form (e.g., 0.05 for 5%)
        - Each column should represent one fund/benchmark
    2. Upload your file below
    3. View the analysis results
    """)
    
    # Add sample data format
    st.markdown("""
    **Sample Data Format:**
    ```
    Date,Fund1,Fund2,Benchmark
    2023-01-31,0.02,-0.01,0.015
    2023-02-28,0.03,0.02,0.01
    2023-03-31,-0.01,0.04,0.02
    ```
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # First verify the file can be read
            try:
                data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
            except pd.errors.ParserError:
                st.error("Failed to parse the CSV file. Please check the file format.")
                st.markdown("""
                **File Format Requirements:**
                - File must be a valid CSV
                - First column must be dates
                - All other columns must contain numeric values
                """)
                return
                
            # Verify we have date index
            if not isinstance(data.index, pd.DatetimeIndex):
                st.error("Date column could not be parsed properly.")
                st.markdown("""
                **Date Format Requirements:**
                - First column must be named 'Date'
                - Dates should be in YYYY-MM-DD format
                - Example: 2023-12-31
                """)
                return
                
            # Verify we have numeric data
            if not data.select_dtypes(include=[np.number]).columns.tolist():
                st.error("No numeric columns found in the data.")
                st.markdown("""
                **Return Data Requirements:**
                - Returns must be in decimal form
                - Example: 0.05 for 5% return
                - No text or special characters allowed
                """)
                return

            analyzer = HedgeFundAnalyzer()
            
            # Display statistics
            st.header("Performance Statistics")
            stats = analyzer.calculate_statistics(data)
            st.dataframe(stats, use_container_width=True)
            
            # Display plots
            st.header("Analysis Charts")
            
            # Cumulative returns
            st.subheader("Cumulative Returns")
            cum_returns_fig = analyzer.plot_cumulative_returns(data)
            st.plotly_chart(cum_returns_fig, use_container_width=True)
            
            # Returns distribution
            st.subheader("Returns Distribution")
            dist_fig = analyzer.plot_returns_distribution(data)
            st.plotly_chart(dist_fig, use_container_width=True)
            
            # Correlation heatmap
            st.subheader("Correlation Matrix")
            corr_fig = analyzer.plot_correlation_heatmap(data)
            st.plotly_chart(corr_fig, use_container_width=True)
            
        except Exception as e:
            st.error("An unexpected error occurred while processing the file.")
            st.markdown(f"""
            **Error Details:**
            ```
            {str(e)}
            ```
            
            **Common Solutions:**
            - Ensure all returns are numeric values
            - Remove any currency symbols or percentage signs
            - Check for missing values
            - Verify date format in first column
            
            If the problem persists, please check the sample file format in the instructions above.
            """)

if __name__ == "__main__":
    main()
