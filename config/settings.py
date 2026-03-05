"""Default parameters, constants, and neobrutalist design configuration."""

# ── Annualization ──────────────────────────────────────────────────────────
ANNUALIZATION_FACTOR = 12  # monthly data

# ── Default Parameters ─────────────────────────────────────────────────────
DEFAULT_RISK_FREE_RATE = 0.0       # annualized
DEFAULT_MAR = 0.0                   # annualized minimum acceptable return
DEFAULT_OMEGA_THRESHOLD = 0.0       # monthly threshold for Omega ratio
CALMAR_MONTHS = 36                  # 3-year window for Calmar ratio
STERLING_PENALTY = 0.10             # 10% added to max drawdown

# ── Fund Types ─────────────────────────────────────────────────────────────
FUND_TYPES = [
    "Long-Only Equity",
    "Long-Short Equity",
    "Quantitative",
    "Other Hedge Fund",
]

# ── Neobrutalist Color Palette (Stripe-inspired) ──────────────────────────
COLORS = {
    "blue_600": "#635BFF",
    "blue_500": "#7A73FF",
    "blue_400": "#9B95FF",
    "blue_100": "#E8E6FF",
    "navy_900": "#0A2540",
    "navy_700": "#425466",
    "gray_100": "#F6F9FC",
    "gray_200": "#E3E8EE",
    "white": "#FFFFFF",
    "green": "#00D4AA",
    "red": "#FF5C5C",
    "orange": "#FF9F43",
    "yellow": "#FFD93D",
}

# Chart color sequence for multi-series plots
CHART_COLORS = [
    "#635BFF",  # primary blue
    "#FF5C5C",  # red
    "#00D4AA",  # green
    "#FF9F43",  # orange
    "#7A73FF",  # lighter blue
    "#425466",  # navy
    "#FFD93D",  # yellow
    "#0A2540",  # dark navy
]

# ── Neobrutalist CSS ──────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --blue-600: #635BFF;
    --blue-500: #7A73FF;
    --blue-100: #E8E6FF;
    --navy-900: #0A2540;
    --navy-700: #425466;
    --gray-100: #F6F9FC;
    --gray-200: #E3E8EE;
    --green: #00D4AA;
    --red: #FF5C5C;
}

/* Global */
.stApp {
    background-color: var(--gray-100);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Main container */
.main .block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Inter', sans-serif !important;
    color: var(--navy-900) !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
}

h1 { font-size: 2.5rem !important; }
h2 { font-size: 1.75rem !important; border-bottom: 3px solid var(--navy-900); padding-bottom: 0.5rem; }
h3 { font-size: 1.25rem !important; }

/* Metric cards */
div[data-testid="stMetric"] {
    background: var(--gray-100);
    border: 2px solid var(--navy-900);
    border-left: 6px solid var(--blue-600);
    border-radius: 0px;
    padding: 1rem 1.25rem;
    box-shadow: 4px 4px 0px var(--navy-900);
}

div[data-testid="stMetric"] label {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: var(--navy-700) !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    color: var(--navy-900) !important;
    font-size: 1.8rem !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--navy-900) !important;
    border-right: 3px solid var(--navy-900);
}

section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stFileUploader label {
    font-weight: 600 !important;
    text-transform: uppercase;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 3px solid var(--navy-900);
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    border: 2px solid var(--navy-900);
    border-bottom: none;
    border-radius: 0;
    padding: 0.75rem 1.5rem;
    background: var(--white);
    color: var(--navy-700) !important;
    margin-right: -2px;
}

.stTabs [aria-selected="true"] {
    background: var(--blue-600) !important;
    color: white !important;
    box-shadow: 2px -2px 0px var(--navy-900);
}

/* Tables */
.stDataFrame {
    border: 2px solid var(--navy-900) !important;
    border-radius: 0 !important;
}

/* Expanders */
.streamlit-expanderHeader {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    background: var(--white);
    border: 2px solid var(--navy-900);
    border-radius: 0;
}

/* Upload buttons */
.stFileUploader > div > div {
    border: 2px dashed var(--blue-600) !important;
    border-radius: 0 !important;
    background: rgba(99, 91, 255, 0.05);
}

/* Divider */
hr {
    border: none;
    border-top: 3px solid var(--navy-900);
    margin: 2rem 0;
}

/* Remove default Streamlit link decorations */
a { color: var(--blue-600) !important; }

/* Number displays */
.big-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3rem;
    font-weight: 800;
    color: var(--navy-900);
    line-height: 1;
}

.metric-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--navy-700);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Card container helper */
.neo-card {
    background: var(--white);
    border: 2px solid var(--navy-900);
    padding: 1.5rem;
    box-shadow: 4px 4px 0px var(--navy-900);
    margin-bottom: 1.5rem;
}

.neo-card-blue {
    background: var(--white);
    border: 2px solid var(--navy-900);
    border-left: 6px solid var(--blue-600);
    padding: 1.5rem;
    box-shadow: 4px 4px 0px var(--navy-900);
    margin-bottom: 1.5rem;
}

/* Positive / Negative coloring */
.positive { color: var(--green) !important; }
.negative { color: var(--red) !important; }
</style>
"""


def neo_card(content: str, blue_accent: bool = False) -> str:
    """Wrap content in a neobrutalist card div."""
    cls = "neo-card-blue" if blue_accent else "neo-card"
    return f'<div class="{cls}">{content}</div>'


def metric_html(label: str, value: str, positive: bool | None = None) -> str:
    """Render a large metric with neobrutalist styling."""
    color_cls = ""
    if positive is True:
        color_cls = " positive"
    elif positive is False:
        color_cls = " negative"
    return neo_card(
        f'<div class="metric-label">{label}</div>'
        f'<div class="big-number{color_cls}">{value}</div>',
        blue_accent=True,
    )
