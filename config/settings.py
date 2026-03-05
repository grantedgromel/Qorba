"""Default parameters, constants, and dark-mode design configuration."""

# -- Annualization ──────────────────────────────────────────────────────────
ANNUALIZATION_FACTOR = 12  # monthly data

# -- Default Parameters ─────────────────────────────────────────────────────
DEFAULT_RISK_FREE_RATE = 0.0       # annualized
DEFAULT_MAR = 0.0                   # annualized minimum acceptable return
DEFAULT_OMEGA_THRESHOLD = 0.0       # monthly threshold for Omega ratio
CALMAR_MONTHS = 36                  # 3-year window for Calmar ratio
STERLING_PENALTY = 0.10             # 10% added to max drawdown

# -- Fund Types ─────────────────────────────────────────────────────────────
FUND_TYPES = [
    "Long-Only Equity",
    "Long-Short Equity",
    "Quantitative",
    "Other Hedge Fund",
]

# -- Dark Mode Color Palette ────────────────────────────────────────────────
COLORS = {
    "bg_primary":     "#0D0D0D",
    "bg_card":        "#1A1A1A",
    "bg_elevated":    "#242424",
    "border":         "#2E2E2E",
    "border_light":   "#3A3A3A",
    "text_primary":   "#F5F5F5",
    "text_secondary": "#A0A0A0",
    "text_muted":     "#666666",
    "accent":         "#FFFFFF",
    "green":          "#4ADE80",
    "red":            "#F87171",
    "sidebar_bg":     "#141414",
    # Legacy aliases (pages reference these keys)
    "white":          "#1A1A1A",
    "navy_900":       "#2E2E2E",
    "navy_700":       "#A0A0A0",
    "gray_100":       "#0D0D0D",
    "gray_200":       "#2E2E2E",
}

# Chart color sequence for multi-series plots (dark-mode friendly)
CHART_COLORS = [
    "#818CF8",  # indigo (primary)
    "#F87171",  # red
    "#4ADE80",  # green
    "#FBBF24",  # amber
    "#A78BFA",  # purple
    "#6B7280",  # gray
    "#38BDF8",  # sky blue
    "#FB923C",  # orange
]

# -- Logo SVG ───────────────────────────────────────────────────────────────
LOGO_SVG = """
<div style="padding: 0.5rem 0 0.25rem 0;">
  <svg width="160" height="40" viewBox="0 0 160 40" xmlns="http://www.w3.org/2000/svg">
    <rect x="2" y="12" width="6" height="6" rx="1" fill="#818CF8"/>
    <text x="14" y="27" font-family="Inter, -apple-system, BlinkMacSystemFont, sans-serif"
          font-size="24" font-weight="700" fill="#F5F5F5" letter-spacing="0.15em">QORBA</text>
  </svg>
  <div style="font-family: Inter, sans-serif; font-size: 11px; font-weight: 400;
              color: #666666; padding-left: 2px; margin-top: -2px;">
    Hedge Fund Analytics
  </div>
</div>
"""

# -- Dark Mode CSS ──────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg-primary:     #0D0D0D;
    --bg-card:        #1A1A1A;
    --bg-elevated:    #242424;
    --border:         #2E2E2E;
    --border-light:   #3A3A3A;
    --text-primary:   #F5F5F5;
    --text-secondary: #A0A0A0;
    --text-muted:     #666666;
    --accent:         #FFFFFF;
    --green:          #4ADE80;
    --red:            #F87171;
    --indigo:         #818CF8;
    --sidebar-bg:     #141414;
}

/* ── Global ────────────────────────────────────────────────────────────── */
.stApp {
    background-color: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-secondary) !important;
}

html, body {
    background-color: var(--bg-primary) !important;
}

/* Main container */
.main .block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ── Typography ────────────────────────────────────────────────────────── */
h1, h2, h3 {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}

h1 { font-size: 2.25rem !important; }
h2 { font-size: 1.5rem !important; border-bottom: none !important; padding-bottom: 0.25rem; }
h3 { font-size: 1.15rem !important; }

/* Force all markdown text to light color */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {
    color: var(--text-secondary) !important;
}

/* Bold text in markdown */
.stMarkdown strong, .stMarkdown b {
    color: var(--text-primary) !important;
}

/* ── Metric Cards ──────────────────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--indigo) !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
}

div[data-testid="stMetric"] label,
[data-testid="stMetricLabel"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"],
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    font-size: 1.8rem !important;
}

[data-testid="stMetricDelta"] {
    color: var(--text-secondary) !important;
}

[data-testid="stMetricDelta"] svg {
    fill: var(--text-secondary) !important;
}

/* ── Sidebar ───────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span {
    color: var(--text-secondary) !important;
}

section[data-testid="stSidebar"] .stMarkdown strong {
    color: var(--text-primary) !important;
}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stFileUploader label,
section[data-testid="stSidebar"] .stCheckbox label {
    font-weight: 600 !important;
    text-transform: uppercase;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em;
    color: var(--text-secondary) !important;
}

/* Sidebar inputs */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] [data-baseweb="select"],
section[data-testid="stSidebar"] [data-baseweb="input"] {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
}

section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: var(--bg-card) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
}

/* Sidebar dividers */
section[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
}

/* Sidebar info/warning boxes */
section[data-testid="stSidebar"] [data-testid="stAlert"] {
    background-color: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-secondary) !important;
    border-radius: 8px !important;
}

/* ── Tabs ──────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid var(--border);
    background-color: transparent;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    border: 1px solid var(--border);
    border-bottom: none;
    border-radius: 8px 8px 0 0;
    padding: 0.6rem 1.25rem;
    background: var(--bg-card) !important;
    color: var(--text-secondary) !important;
}

.stTabs [aria-selected="true"] {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-light);
    box-shadow: none !important;
}

/* ── Tables / DataFrames ───────────────────────────────────────────────── */
.stDataFrame,
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] > div {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* DataFrame cell text */
[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] th,
.stDataFrame td,
.stDataFrame th {
    color: var(--text-primary) !important;
    background-color: var(--bg-card) !important;
    border-color: var(--border) !important;
}

/* Glide data grid (Streamlit internal table renderer) */
[data-testid="stDataFrame"] [role="grid"],
[data-testid="stDataFrame"] [role="gridcell"],
[data-testid="stDataFrame"] [role="columnheader"] {
    color: var(--text-primary) !important;
}

/* ── Expanders ─────────────────────────────────────────────────────────── */
.streamlit-expanderHeader,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
}

[data-testid="stExpander"] summary span {
    color: var(--text-primary) !important;
}

[data-testid="stExpander"] > div {
    background: var(--bg-card) !important;
    color: var(--text-secondary) !important;
}

/* ── File Uploaders ────────────────────────────────────────────────────── */
.stFileUploader > div > div,
[data-testid="stFileUploader"] > div {
    border: 1px dashed var(--border-light) !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.02) !important;
}

.stFileUploader > div > div:hover,
[data-testid="stFileUploader"] > div:hover {
    background: rgba(255,255,255,0.05) !important;
}

/* ── Divider ───────────────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Links ─────────────────────────────────────────────────────────────── */
a { color: var(--indigo) !important; }

/* ── Buttons ───────────────────────────────────────────────────────────── */
.stButton > button {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    transition: background-color 0.15s ease;
}

.stButton > button:hover {
    background-color: var(--bg-elevated) !important;
    border-color: var(--border-light) !important;
}

/* ── Selectbox / Number Input ──────────────────────────────────────────── */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background-color: var(--bg-card) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

[data-baseweb="select"] span,
[data-baseweb="input"] input {
    color: var(--text-primary) !important;
}

/* Dropdown menu */
[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"] {
    background-color: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

[data-baseweb="menu"] li,
[role="option"] {
    color: var(--text-primary) !important;
}

[data-baseweb="menu"] li:hover,
[role="option"]:hover {
    background-color: var(--bg-card) !important;
}

/* ── Info / Warning / Success / Error Alerts ───────────────────────────── */
[data-testid="stAlert"],
.stAlert {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-secondary) !important;
}

[data-testid="stAlert"] p {
    color: var(--text-secondary) !important;
}

/* ── Checkbox ──────────────────────────────────────────────────────────── */
.stCheckbox label span {
    color: var(--text-primary) !important;
}

/* ── Number displays (custom classes) ──────────────────────────────────── */
.big-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
}

.metric-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Card container helper */
.neo-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    margin-bottom: 1rem;
}

.neo-card-accent {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--indigo);
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    margin-bottom: 1rem;
}

/* Positive / Negative coloring */
.positive { color: var(--green) !important; }
.negative { color: var(--red) !important; }

/* ── Plotly chart containers ───────────────────────────────────────────── */
[data-testid="stPlotlyChart"],
.stPlotlyChart {
    background-color: transparent !important;
    border-radius: 12px;
}

/* ── Navigation ────────────────────────────────────────────────────────── */
[data-testid="stSidebarNav"] a,
[data-testid="stSidebarNavLink"] {
    color: var(--text-secondary) !important;
    border-radius: 8px !important;
}

[data-testid="stSidebarNav"] a:hover,
[data-testid="stSidebarNavLink"]:hover {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
}

[data-testid="stSidebarNav"] [aria-selected="true"],
[data-testid="stSidebarNavLink"][aria-current="page"] {
    background-color: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
}

/* ── Caption text ──────────────────────────────────────────────────────── */
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--text-muted) !important;
}

/* ── Scrollbar (subtle dark) ───────────────────────────────────────────── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: var(--bg-primary);
}
::-webkit-scrollbar-thumb {
    background: var(--border-light);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--text-muted);
}

/* ── Tooltip / Help icon ───────────────────────────────────────────────── */
[data-testid="stTooltipIcon"] {
    color: var(--text-muted) !important;
}

/* ── Form submit button ────────────────────────────────────────────────── */
[data-testid="stFormSubmitButton"] button {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
</style>
"""


def neo_card(content: str, accent: bool = False) -> str:
    """Wrap content in a dark-mode card div."""
    cls = "neo-card-accent" if accent else "neo-card"
    return f'<div class="{cls}">{content}</div>'


def metric_html(label: str, value: str, positive: bool | None = None) -> str:
    """Render a large metric with dark-mode styling."""
    color_cls = ""
    if positive is True:
        color_cls = " positive"
    elif positive is False:
        color_cls = " negative"
    return neo_card(
        f'<div class="metric-label">{label}</div>'
        f'<div class="big-number{color_cls}">{value}</div>',
        accent=True,
    )
