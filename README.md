# Qorba

Hedge fund analytics dashboard built with Streamlit. Upload monthly fund returns (CSV, Excel, or PDF tearsheet) along with optional benchmark and peer-group series, and Qorba produces an interactive performance, risk, regression, drawdown, and peer-comparison report.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app serves at `http://localhost:8501`. A devcontainer (`.devcontainer/devcontainer.json`) is configured to launch this automatically in Codespaces and forward port 8501.

To explore the UI without your own data, tick **Use Sample Data** in the sidebar.

## Input file formats

All numeric series are interpreted as **monthly returns**. Values with magnitude > 1.0 are auto-detected as percentages and rescaled to decimals (so `2.5` and `0.025` are both treated as 2.5%).

- **Fund returns (required)** — CSV, XLSX, XLS, or PDF tearsheet.
  - Tabular files: a date column (named `date`, `month`, `period`, etc., or the first column) plus a return column. The first numeric column is used.
  - PDF: parsed with PyMuPDF; review the extracted preview before relying on it.
- **Benchmark returns (optional)** — CSV/Excel; one or more return columns alongside a date column.
- **Peer group returns (optional)** — CSV/Excel; one return column per peer fund.

Recognized date formats include `YYYY-MM-DD`, `MM/DD/YYYY`, `YYYY-MM`, `DD/MM/YYYY`, and `YYYYMMDD`. Fund/benchmark/peer series are aligned to their common date intersection.

## Sidebar parameters

- **Fund Type** — Long-Only Equity, Long-Short Equity, Quantitative, or Other Hedge Fund.
- **Risk-Free Rate (%)** — annualized, used in Sharpe and excess-return calculations.
- **MAR (%)** — annualized minimum acceptable return for Sortino and downside metrics.
- **Omega Threshold (%)** — monthly threshold for the Omega ratio.

## Pages

| Page | Contents |
| --- | --- |
| Overview | Headline KPIs, cumulative growth, return distribution. |
| Return Measures | CAGR, arithmetic/geometric means, period returns. |
| Risk Measures | Volatility, downside deviation, VaR/CVaR, Sharpe/Sortino/Omega/Calmar/Sterling. |
| Regression Analysis | Alpha/beta vs. benchmarks, capture ratios, scatter and correlation. |
| Rolling Analytics | Rolling return, vol, Sharpe, beta. |
| Peer Group Analysis | Percentile ranks and peer comparison charts. |
| Drawdown Analysis | Underwater curve, top drawdowns table. |
| Calendar Performance | Monthly heatmap and yearly returns. |

## Project layout

```
app.py                # Streamlit entry point: sidebar, data loading, navigation
config/settings.py    # Defaults, dark-mode palette, CSS, logo
data/
  loader.py           # CSV/Excel parsing, date alignment, validation
  pdf_parser.py       # PDF tearsheet extraction
  sample_data.py      # Synthetic series for the demo toggle
analytics/            # Calculation modules (return, risk, regression, rolling, peer)
charts/               # Plotly chart builders + theme
pages/                # One Streamlit page per analysis section
tests/                # (Currently empty)
```

## Development

The dashboard targets Python 3.11 (matches the devcontainer image). Dependencies are pinned in `requirements.txt`:

- streamlit ≥ 1.45
- plotly ≥ 6.0
- scipy ≥ 1.15
- statsmodels ≥ 0.14
- openpyxl ≥ 3.1, xlrd ≥ 2.0 (Excel)
- pymupdf ≥ 1.24 (PDF)

There is no test suite yet; `tests/` contains only an `__init__.py`.
