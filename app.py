import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

# ── Page config — MUST be the very first Streamlit call ───────────────────────
st.set_page_config(
    page_title="DermAI Monitor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Static imports ─────────────────────────────────────────────────────────────
from database.db import init_db
from pages import home, scan, progress, history, about

# ── DB init (once per process) ────────────────────────────────────────────────
@st.cache_resource
def _init_db_once():
    init_db()
    return True

_init_db_once()


# ── Global CSS ────────────────────────────────────────────────────────────────
@st.cache_resource
def _get_css() -> str:
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Mono:wght@400;500&display=swap');

/* ─── Reset & Root ─────────────────────────────────────── */
:root {
    --bg:          #0d1117;
    --surface:     #161b22;
    --surface2:    #1c2128;
    --border:      #30363d;
    --border2:     #21262d;
    --text:        #c9d1d9;
    --text-dim:    #8b949e;
    --text-bright: #f0f6fc;
    --accent:      #58a6ff;
    --green:       #3fb950;
    --yellow:      #d29922;
    --red:         #f85149;
    --purple:      #bc8cff;
    --radius:      8px;
    --radius-lg:   12px;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ─── Streamlit scaffolding ────────────────────────────── */
.stApp { background-color: var(--bg) !important; }
.main .block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }
section[data-testid="stSidebar"] { background-color: var(--surface) !important; border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] > div { padding: 1.5rem 1rem; }

/* ─── Sidebar logo ──────────────────────────────────────── */
.sidebar-logo {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-bright);
    padding: 0.25rem 0 1.25rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sidebar-logo span { color: var(--accent); }

/* ─── Sidebar nav buttons ───────────────────────────────── */
.nav-section-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.75rem 0.85rem 0.35rem;
}

/* Nav button base — overrides the global .stButton only inside sidebar */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: var(--text-dim) !important;
    font-weight: 500 !important;
    border: 1px solid transparent !important;
    border-radius: var(--radius) !important;
    padding: 0.55rem 0.85rem !important;
    font-size: 0.92rem !important;
    text-align: left !important;
    width: 100% !important;
    transition: background 0.15s, color 0.15s, border-color 0.15s !important;
    margin-bottom: 0.15rem !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--surface2) !important;
    color: var(--text-bright) !important;
    border-color: var(--border2) !important;
    opacity: 1 !important;
}
/* Active nav button */
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: rgba(88,166,255,0.12) !important;
    color: var(--accent) !important;
    border-color: rgba(88,166,255,0.3) !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    opacity: 1 !important;
    background: rgba(88,166,255,0.18) !important;
}

/* ─── Headings ──────────────────────────────────────────── */
h1, h2, h3, h4 { color: var(--text-bright) !important; letter-spacing: -0.02em; }
.page-title  { font-size: 1.9rem; font-weight: 700; margin-bottom: 0.25rem; color: var(--text-bright) !important; }
.page-sub    { color: var(--text-dim); margin-bottom: 1.5rem; font-size: 0.95rem; }
.section-title { font-size: 1.05rem; font-weight: 600; color: var(--text-bright); margin: 1.2rem 0 0.75rem; border-bottom: 1px solid var(--border); padding-bottom: 0.4rem; }
.section-title.small { font-size: 0.95rem; }
.page-header { margin-bottom: 2rem; }
.page-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
.subtitle { color: var(--text-dim); font-size: 1rem; margin-top: 0.3rem; }

/* ─── Metric boxes ──────────────────────────────────────── */
.metric-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.2rem 1rem;
    text-align: center;
    transition: border-color 0.2s, transform 0.15s;
    margin-bottom: 0.5rem;
    border-left: 3px solid var(--border);
}
.metric-box:hover { border-color: var(--accent); transform: translateY(-1px); }
.metric-icon  { font-size: 1.6rem; margin-bottom: 0.4rem; }
.metric-value { font-size: 1.9rem; font-weight: 700; color: var(--text-bright); line-height: 1.1; }
.metric-value.small { font-size: 1.1rem; }
.metric-label { font-size: 0.78rem; color: var(--text-dim); margin-top: 0.3rem; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-box.accent-blue   { border-left-color: var(--accent); }
.metric-box.accent-green  { border-left-color: var(--green); }
.metric-box.accent-yellow { border-left-color: var(--yellow); }
.metric-box.accent-red    { border-left-color: var(--red); }
.metric-box.accent-purple { border-left-color: var(--purple); }

/* ─── Severity badges ───────────────────────────────────── */
.badge-mild     { background:#0f2a1a; color:#3fb950; border:1px solid #238636; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.badge-moderate { background:#2a1f00; color:#e3b341; border:1px solid #9e6a03; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.badge-severe   { background:#2d0e0c; color:#f85149; border:1px solid #b91c1c; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }

/* ─── Disease tag ───────────────────────────────────────── */
.disease-tag       { background:#0d2e4a; color:#58a6ff; border:1px solid #1f6feb; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.disease-tag.small { font-size:0.75rem; padding:2px 8px; }

/* ─── Step cards (home) ─────────────────────────────────── */
.step-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.2rem 0.8rem;
    text-align: center;
    height: 100%;
    transition: border-color 0.2s, transform 0.15s;
}
.step-card:hover { border-color: var(--accent); transform: translateY(-2px); }
.step-icon  { font-size: 2rem; margin-bottom: 0.5rem; }
.step-title { font-size: 0.9rem; font-weight: 600; color: var(--text-bright); margin-bottom: 0.4rem; }
.step-desc  { font-size: 0.78rem; color: var(--text-dim); line-height: 1.5; }

/* ─── Recent scan cards (home) ──────────────────────────── */
.recent-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.75rem 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.5rem;
    flex-wrap: wrap;
    gap: 0.5rem;
    transition: border-color 0.2s;
}
.recent-card:hover { border-color: var(--accent); }
.recent-left, .recent-middle, .recent-right { display: flex; align-items: center; gap: 0.5rem; }
.patient-name { font-weight: 600; color: var(--text-bright); font-size: 0.9rem; }
.stat-chip { background: var(--surface2); color: var(--text-dim); padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; border: 1px solid var(--border2); }
.scan-date { color: var(--text-dim); font-size: 0.75rem; }

/* ─── Step header (scan page) ───────────────────────────── */
.step-header {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-bright);
    margin: 1.5rem 0 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.step-num {
    background: var(--accent);
    color: #0d1117;
    width: 24px; height: 24px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: 700;
    flex-shrink: 0;
}

/* ─── Info box ──────────────────────────────────────────── */
.info-box {
    background: #0d2e4a;
    border: 1px solid #1f6feb;
    border-radius: var(--radius);
    padding: 0.75rem 1rem;
    color: var(--text);
    font-size: 0.9rem;
    margin: 0.5rem 0;
    line-height: 1.6;
}

/* ─── Image label ───────────────────────────────────────── */
.img-label { font-size: 0.85rem; color: var(--text-dim); font-weight: 500; margin-bottom: 0.4rem; }

/* ─── Disease info card ─────────────────────────────────── */
.disease-info-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.2rem 1.2rem 1.2rem 1.5rem;
}
.disease-card-title { font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem; }
.disease-card-desc  { font-size: 0.9rem; color: var(--text); line-height: 1.6; margin-bottom: 0.6rem; }
.disease-card-advice{ font-size: 0.88rem; color: var(--text-dim); line-height: 1.5; }

/* ─── Confidence bars ───────────────────────────────────── */
.conf-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.35rem;
}
.conf-label  { width: 200px; font-size: 0.82rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.conf-bar-bg { flex: 1; background: var(--border2); height: 8px; border-radius: 4px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 4px; transition: width 0.4s ease; }
.conf-pct    { width: 45px; text-align: right; font-size: 0.82rem; }

/* ─── History patient card ──────────────────────────────── */
.history-patient-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1rem 1.2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.6rem;
    gap: 1rem;
    flex-wrap: wrap;
    transition: border-color 0.2s;
}
.history-patient-card:hover { border-color: var(--accent); }
.hpc-left   { display: flex; align-items: center; gap: 1rem; }
.hpc-right  { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
.hpc-avatar { width: 42px; height: 42px; background: var(--accent); color: #0d1117; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 1.1rem; flex-shrink: 0; }
.hpc-name   { font-weight: 600; color: var(--text-bright); font-size: 1rem; }
.hpc-meta   { font-size: 0.78rem; color: var(--text-dim); margin-top: 0.15rem; }
.count-label{ color: var(--text-dim); font-size: 0.85rem; margin-bottom: 0.75rem; }

/* ─── Detail table (history expand) ────────────────────── */
.detail-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
.detail-table td { padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--border2); color: var(--text); }
.detail-table td:first-child { color: var(--text-dim); width: 130px; font-weight: 500; }
.no-img-placeholder { background: var(--surface2); border: 1px dashed var(--border); border-radius: var(--radius); height: 140px; display: flex; align-items: center; justify-content: center; color: var(--text-dim); font-size: 0.85rem; }

/* ─── Report card ───────────────────────────────────────── */
.report-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
}
.report-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; border-bottom: 1px solid var(--border); padding-bottom: 0.75rem; }
.report-title  { font-size: 1.1rem; font-weight: 700; color: var(--text-bright); }
.report-meta   { font-size: 0.8rem; color: var(--text-dim); }
.report-section { font-size: 0.9rem; color: var(--text); line-height: 1.7; margin-bottom: 0.75rem; padding-bottom: 0.75rem; border-bottom: 1px solid var(--border2); }
.report-disclaimer { background: #2a1400; border: 1px solid #6e3a00; border-radius: var(--radius); padding: 0.75rem 1rem; font-size: 0.82rem; color: #e3b341; margin-top: 0.5rem; }

/* ─── About page ────────────────────────────────────────── */
.about-hero {
    background: linear-gradient(135deg, #161b22 0%, #0d2e4a 100%);
    border: 1px solid #1f6feb;
    border-radius: var(--radius-lg);
    padding: 2.5rem;
    text-align: center;
    margin-bottom: 1rem;
}
.about-hero-icon  { font-size: 3.5rem; margin-bottom: 0.75rem; }
.about-hero-title { font-size: 2rem; font-weight: 700; color: var(--text-bright); margin-bottom: 0.4rem; }
.about-hero-sub   { font-size: 1rem; color: var(--accent); margin-bottom: 1rem; }
.about-hero-desc  { font-size: 0.92rem; color: var(--text-dim); max-width: 640px; margin: 0 auto; line-height: 1.7; }

.feature-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem;
    display: flex;
    gap: 0.9rem;
    align-items: flex-start;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s;
}
.feature-card:hover { border-color: var(--accent); }
.feature-icon  { font-size: 1.5rem; flex-shrink: 0; }
.feature-title { font-weight: 600; color: var(--text-bright); font-size: 0.95rem; margin-bottom: 0.2rem; }
.feature-desc  { font-size: 0.82rem; color: var(--text-dim); line-height: 1.5; }

.tech-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; margin-bottom: 1rem; }
.tech-table th { background: var(--surface2); color: var(--text-bright); padding: 0.6rem 1rem; text-align: left; border: 1px solid var(--border); font-weight: 600; }
.tech-table td { padding: 0.5rem 1rem; border: 1px solid var(--border2); color: var(--text); }
.tech-table code { background: var(--border2); padding: 2px 6px; border-radius: 4px; font-family: 'DM Mono', monospace; font-size: 0.82rem; color: var(--accent); }

.disclaimer-box {
    background: #1a0a0a;
    border: 1px solid #6e1c1c;
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    color: #fca5a5;
    font-size: 0.88rem;
    line-height: 1.7;
}
.disclaimer-title { font-size: 1rem; font-weight: 700; margin-bottom: 0.75rem; color: var(--red); }
.disclaimer-box p { margin: 0 0 0.75rem; }

/* ─── Empty state ───────────────────────────────────────── */
.empty-state {
    background: var(--surface);
    border: 1px dashed var(--border);
    border-radius: var(--radius-lg);
    padding: 3rem;
    text-align: center;
    color: var(--text-dim);
    font-size: 0.95rem;
}

/* ─── Streamlit widget overrides ────────────────────────── */

/* Primary buttons — scoped to main content area only (NOT sidebar) */
.main .stButton > button[kind="primary"],
.main .stButton > button:not([kind="secondary"]) {
    background: var(--accent) !important;
    color: #0d1117 !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 0.55rem 1.2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    transition: opacity 0.15s !important;
}
.main .stButton > button[kind="primary"]:hover,
.main .stButton > button:not([kind="secondary"]):hover {
    opacity: 0.88 !important;
}

/* Secondary buttons */
.main .stButton > button[kind="secondary"] {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    font-weight: 500 !important;
    border-radius: var(--radius) !important;
    padding: 0.55rem 1.2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    transition: border-color 0.15s, background 0.15s !important;
}
.main .stButton > button[kind="secondary"]:hover {
    border-color: var(--accent) !important;
    background: var(--surface) !important;
    opacity: 1 !important;
}

div[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1rem !important;
}
div[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }

.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: var(--radius) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.12) !important;
}

/* Input labels */
.stTextInput label, .stTextArea label, .stNumberInput label,
.stSelectbox label, .stRadio label, .stFileUploader label {
    color: var(--text) !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}

.stSelectbox > div > div,
div[data-testid="stSelectbox"] > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
}

/* Selectbox dropdown option text */
div[data-baseweb="select"] span { color: var(--text) !important; }

.stRadio > div { gap: 0.5rem !important; }
.stRadio label { color: var(--text) !important; }
.stRadio div[data-testid="stMarkdownContainer"] p { color: var(--text-dim) !important; font-size: 0.85rem; }

div[data-testid="stTab"] button {
    color: var(--text-dim) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
}
div[data-testid="stTab"] button[aria-selected="true"] {
    color: var(--text-bright) !important;
    border-bottom-color: var(--accent) !important;
}

div[data-testid="stExpander"] details {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
div[data-testid="stExpander"] summary { color: var(--text) !important; font-weight: 500 !important; }

.stDataFrame { border: 1px solid var(--border) !important; border-radius: var(--radius) !important; overflow: hidden; }
div[data-testid="stDataFrameResizable"] { background: var(--surface) !important; }

.stSuccess { background: #0f2a1a !important; border-color: #238636 !important; color: #3fb950 !important; }
.stWarning { background: #2a1f00 !important; border-color: #9e6a03 !important; color: #e3b341 !important; }
.stError   { background: #2d0e0c !important; border-color: #b91c1c !important; color: #f85149 !important; }
.stInfo    { background: #0d2e4a !important; border-color: #1f6feb !important; color: #58a6ff !important; }

div[data-testid="stForm"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1.2rem !important;
}

/* Spinner */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* Progress bar */
.stProgress > div > div { background: var(--accent) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }
</style>
"""

st.markdown(_get_css(), unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────────
NAV_PAGES = [
    "🏠  Dashboard",
    "🖼️  New Scan",
    "📈  Progress Report",
    "📋  History",
    "ℹ️  About",
]

if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = NAV_PAGES[0]

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        🔬 <span>DermAI</span> Monitor
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-section-label">Navigation</div>', unsafe_allow_html=True)

    # Button-based nav — no st.radio, no duplicate widget bleed
    for nav_item in NAV_PAGES:
        is_active = st.session_state["nav_page"] == nav_item
        if st.button(
            nav_item,
            key=f"nav_btn_{nav_item}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state["nav_page"] = nav_item
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Sidebar status indicators
    st.markdown("""
    <div style="background:var(--surface2);border:1px solid var(--border2);border-radius:var(--radius);padding:0.75rem 0.9rem;margin-bottom:0.75rem;">
        <div style="font-size:0.72rem;color:var(--text-dim);font-weight:600;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.5rem;">System Status</div>
        <div style="display:flex;align-items:center;gap:0.5rem;font-size:0.8rem;color:var(--text);margin-bottom:0.25rem;">
            <span style="width:7px;height:7px;background:#3fb950;border-radius:50%;display:inline-block;"></span> AI Model Ready
        </div>
        <div style="display:flex;align-items:center;gap:0.5rem;font-size:0.8rem;color:var(--text);">
            <span style="width:7px;height:7px;background:#3fb950;border-radius:50%;display:inline-block;"></span> Database Online
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.72rem; color:#8b949e; border-top:1px solid #30363d; padding-top:0.85rem; line-height:1.7;">
        ⚠️ For educational &amp; research use only.<br>
        Not a medical diagnostic tool.<br><br>
        <span style="color:var(--text-dim);">Model:</span> EfficientNetB3<br>
        <span style="color:var(--text-dim);">Dataset:</span> HAM10000 + ISIC 2019<br>
        <span style="color:var(--text-dim);">v2.0</span>
    </div>
    """, unsafe_allow_html=True)

# ── Page routing ────────────────────────────────────────────────────────────────
page = st.session_state["nav_page"]

if "Dashboard" in page:
    home.show()
elif "New Scan" in page:
    scan.show()
elif "Progress" in page:
    progress.show()
elif "History" in page:
    history.show()
elif "About" in page:
    about.show()
