import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import os

load_dotenv()

from src.simulator import render_simulator
from utils.validators import validate_required_columns, validate_basic_format, validate_numeric_columns
from src.data_loader import load_csv, preprocess_data
from src.analytics import (
    compute_metrics, sales_by_hour, sales_by_day,
    top_sellers, top_brands, detect_low_conversion, detect_patterns
)
from src.recommendations import generate_recommendations
from src.chat_ai import get_chat_response
from src.services.ml_service import train_sales_model, predict_sales

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="APEXCO AI · Auditoría Inteligente",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── GLOBAL CSS — TEMA CLARO PROFESIONAL ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,300;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── VARIABLES ── */
:root {
    --bg:          #F7F7F5;
    --bg-card:     #FFFFFF;
    --bg-subtle:   #F0EFEC;
    --bg-hover:    #ECEAE6;
    --border:      #E2E0DB;
    --border-med:  #D4D1CB;
    --text-primary:   #1A1916;
    --text-secondary: #5C5A55;
    --text-muted:     #9B9890;
    --accent:      #D4521A;
    --accent-light: #F5EDE7;
    --accent-soft:  #FAF3EF;
    --accent2:     #E8860A;
    --green:       #2A7D4F;
    --green-light: #EAF5EE;
    --red:         #C1392B;
    --red-light:   #FCECEA;
    --blue:        #2563EB;
    --blue-light:  #EFF4FF;
    --amber:       #B45309;
    --amber-light: #FFFBEB;
    --shadow-xs:   0 1px 3px rgba(26,25,22,0.06);
    --shadow-sm:   0 2px 8px rgba(26,25,22,0.08);
    --shadow-md:   0 4px 16px rgba(26,25,22,0.10);
    --radius-sm:   8px;
    --radius-md:   12px;
    --radius-lg:   16px;
    --radius-xl:   20px;
}

/* ── RESET BASE ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text-primary) !important;
}
.stApp {
    background: var(--bg) !important;
}
*, *::before, *::after { box-sizing: border-box; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-subtle); }
::-webkit-scrollbar-thumb { background: var(--border-med); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ── HERO ── */
.apex-hero {
    background: #FFFFFF;
    border-bottom: 1px solid var(--border);
    padding: 2.5rem 2.5rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    position: relative;
    overflow: hidden;
}
.apex-hero::before {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 320px; height: 100%;
    background: linear-gradient(135deg, var(--accent-soft) 0%, transparent 60%);
    pointer-events: none;
}
.apex-hero-inner {
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1.5rem;
}
.apex-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: var(--accent-light);
    border: 1px solid rgba(212,82,26,0.25);
    color: var(--accent);
    font-size: 0.68rem;
    padding: 0.25rem 0.8rem;
    border-radius: 100px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    width: fit-content;
}
.apex-badge-dot {
    width: 6px; height: 6px;
    background: var(--accent);
    border-radius: 50%;
    animation: pulse-dot 2.5s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.6); }
}
.apex-logo {
    font-family: 'Instrument Serif', serif;
    font-size: 2.6rem;
    font-weight: 400;
    color: var(--text-primary);
    line-height: 1;
    margin: 0;
    letter-spacing: -0.02em;
}
.apex-logo span {
    color: var(--accent);
    font-style: italic;
}
.apex-tagline {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-top: 0.5rem;
    font-weight: 400;
    letter-spacing: 0.04em;
}
.apex-hero-stats {
    display: flex;
    gap: 2.5rem;
    flex-wrap: wrap;
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.2rem 1.8rem;
}
.hero-stat-val {
    font-family: 'Instrument Serif', serif;
    font-size: 1.6rem;
    font-weight: 400;
    color: var(--text-primary);
    line-height: 1;
}
.hero-stat-lbl {
    font-size: 0.65rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.2rem;
}

/* ── SECTION HEADERS ── */
.section-eyebrow {
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin-bottom: 0.3rem;
}
.section-heading {
    font-family: 'Instrument Serif', serif;
    font-size: 1.3rem;
    font-weight: 400;
    color: var(--text-primary);
    margin: 0 0 1.2rem 0;
    letter-spacing: -0.01em;
}

/* ── DIVIDER ── */
.apex-divider {
    height: 1px;
    background: var(--border);
    margin: 2.5rem 0;
}

/* ── METRIC CARDS ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.2s ease, border-color 0.2s ease, transform 0.2s ease;
    box-shadow: var(--shadow-xs);
}
.metric-card:hover {
    border-color: var(--border-med);
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}
.metric-card-accent {
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--accent2));
    border-radius: 0;
}
.metric-icon {
    font-size: 1.4rem;
    margin-bottom: 0.9rem;
    display: block;
}
.metric-value {
    font-family: 'Instrument Serif', serif;
    font-size: 2.4rem;
    font-weight: 400;
    color: var(--text-primary);
    line-height: 1;
    margin-bottom: 0.3rem;
    letter-spacing: -0.02em;
}
.metric-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 500;
}
.metric-delta {
    font-size: 0.75rem;
    margin-top: 0.6rem;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.2rem 0.65rem;
    border-radius: 100px;
    font-weight: 500;
}
.metric-delta.good {
    background: var(--green-light);
    color: var(--green);
    border: 1px solid rgba(42,125,79,0.2);
}
.metric-delta.bad {
    background: var(--red-light);
    color: var(--red);
    border: 1px solid rgba(193,57,43,0.2);
}
.metric-delta.neutral {
    background: var(--amber-light);
    color: var(--amber);
    border: 1px solid rgba(180,83,9,0.2);
}

/* ── CHART CARDS ── */
.chart-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem 1.5rem 0.5rem;
    box-shadow: var(--shadow-xs);
    margin-bottom: 1rem;
}

/* ── AUDIT / ALERT CARDS ── */
.audit-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1rem 1.3rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: flex-start;
    gap: 0.9rem;
    transition: box-shadow 0.2s;
    box-shadow: var(--shadow-xs);
    font-size: 0.875rem;
    color: var(--text-secondary);
    line-height: 1.6;
}
.audit-card:hover { box-shadow: var(--shadow-sm); }
.audit-icon { font-size: 1.1rem; flex-shrink: 0; margin-top: 0.1rem; }
.audit-card.warning {
    border-left: 3px solid var(--accent2);
    background: var(--amber-light);
}
.audit-card.danger {
    border-left: 3px solid var(--red);
    background: var(--red-light);
}
.audit-card.success {
    border-left: 3px solid var(--green);
    background: var(--green-light);
}
.audit-card.info {
    border-left: 3px solid var(--blue);
    background: var(--blue-light);
}

/* ── PATTERN PILLS ── */
.pills-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 1.2rem; }
.pattern-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--bg-subtle);
    border: 1px solid var(--border-med);
    border-radius: 100px;
    padding: 0.38rem 1rem;
    font-size: 0.8rem;
    color: var(--text-secondary);
    font-weight: 500;
    white-space: nowrap;
}
.pattern-pill.good  { border-color: rgba(42,125,79,0.35);  color: var(--green);  background: var(--green-light); }
.pattern-pill.bad   { border-color: rgba(193,57,43,0.35);  color: var(--red);    background: var(--red-light); }
.pattern-pill.info  { border-color: rgba(180,83,9,0.35);   color: var(--amber);  background: var(--amber-light); }

/* ── STATUS BADGE ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--green-light);
    border: 1px solid rgba(42,125,79,0.25);
    color: var(--green);
    font-size: 0.75rem;
    padding: 0.4rem 1rem;
    border-radius: 100px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.status-dot {
    width: 7px; height: 7px;
    background: var(--green);
    border-radius: 50%;
    box-shadow: 0 0 0 2px rgba(42,125,79,0.25);
}

/* ── CHAT ── */
.chat-wrapper {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    overflow: hidden;
    box-shadow: var(--shadow-sm);
    margin-bottom: 1rem;
}
.chat-header {
    background: linear-gradient(90deg, var(--accent-soft), #FFFFFF);
    border-bottom: 1px solid var(--border);
    padding: 1.2rem 1.6rem;
    display: flex;
    align-items: center;
    gap: 0.9rem;
}
.chat-header-dot {
    width: 10px; height: 10px;
    background: var(--green);
    border-radius: 50%;
    box-shadow: 0 0 0 3px var(--green-light);
    flex-shrink: 0;
}
.chat-header-title {
    font-family: 'Instrument Serif', serif;
    font-size: 1rem;
    font-weight: 400;
    color: var(--text-primary);
}
.chat-header-sub {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 0.1rem;
}
.chat-body {
    padding: 1.2rem 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.8rem;
}
.msg-user {
    background: var(--accent-light);
    border: 1px solid rgba(212,82,26,0.15);
    border-radius: 14px 14px 4px 14px;
    padding: 0.8rem 1.1rem;
    margin-left: 3rem;
    font-size: 0.875rem;
    color: var(--text-primary);
    line-height: 1.55;
}
.msg-assistant {
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    border-radius: 14px 14px 14px 4px;
    padding: 0.8rem 1.1rem;
    margin-right: 3rem;
    font-size: 0.875rem;
    color: var(--text-secondary);
    line-height: 1.65;
}
.msg-label {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.25rem;
    font-weight: 600;
}
.msg-label.user { color: var(--accent); }
.msg-label.apex { color: var(--text-muted); }

/* ── LANDING ── */
.landing-empty {
    text-align: center;
    padding: 5rem 2rem 4rem;
    max-width: 600px;
    margin: 0 auto;
}
.landing-icon {
    font-size: 3.5rem;
    margin-bottom: 1.5rem;
    display: block;
}
.landing-title {
    font-family: 'Instrument Serif', serif;
    font-size: 2rem;
    font-weight: 400;
    color: var(--text-primary);
    margin-bottom: 0.8rem;
    letter-spacing: -0.02em;
}
.landing-sub {
    color: var(--text-muted);
    font-size: 0.92rem;
    line-height: 1.8;
    margin: 0 auto 2rem;
}
.landing-features {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1.5rem;
}

/* ── ML SECTION ── */
.ml-section {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    padding: 1.8rem 2rem;
    position: relative;
    overflow: hidden;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-xs);
}
.ml-section::before {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 200px; height: 100%;
    background: linear-gradient(135deg, var(--blue-light), transparent);
    pointer-events: none;
    opacity: 0.5;
}
.ml-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.ml-icon-wrap {
    width: 46px; height: 46px;
    background: var(--blue-light);
    border: 1px solid rgba(37,99,235,0.2);
    border-radius: var(--radius-md);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    flex-shrink: 0;
}
.ml-title {
    font-family: 'Instrument Serif', serif;
    font-size: 1.05rem;
    font-weight: 400;
    color: var(--text-primary);
    line-height: 1;
}
.ml-subtitle {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 0.25rem;
    letter-spacing: 0.04em;
}
.ml-badge {
    margin-left: auto;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--green-light);
    border: 1px solid rgba(42,125,79,0.25);
    color: var(--green);
    font-size: 0.68rem;
    padding: 0.25rem 0.8rem;
    border-radius: 100px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.ml-stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.ml-stat-card {
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.1rem 1.2rem;
    text-align: center;
    transition: border-color 0.2s;
}
.ml-stat-card:hover { border-color: var(--border-med); }
.ml-stat-val {
    font-family: 'Instrument Serif', serif;
    font-size: 1.8rem;
    font-weight: 400;
    color: var(--text-primary);
    line-height: 1;
    margin-bottom: 0.3rem;
    letter-spacing: -0.02em;
}
.ml-stat-val.accent { color: var(--blue); }
.ml-stat-val.good   { color: var(--green); }
.ml-stat-val.warn   { color: var(--amber); }
.ml-stat-lbl {
    font-size: 0.68rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 500;
}
.ml-accuracy-bar-wrap { margin-top: 0.8rem; }
.ml-accuracy-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-bottom: 0.35rem;
}
.ml-accuracy-bar {
    height: 5px;
    background: var(--border);
    border-radius: 100px;
    overflow: hidden;
}
.ml-accuracy-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--blue), #60A5FA);
    border-radius: 100px;
    transition: width 1s ease;
}
.ml-predict-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
}
.ml-predict-card {
    background: var(--blue-light);
    border: 1px solid rgba(37,99,235,0.15);
    border-radius: var(--radius-sm);
    padding: 0.9rem 1rem;
    text-align: center;
}
.ml-predict-hour {
    font-size: 0.65rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
}
.ml-predict-val {
    font-family: 'Instrument Serif', serif;
    font-size: 1.4rem;
    font-weight: 400;
    color: var(--blue);
    line-height: 1;
}
.ml-predict-label {
    font-size: 0.65rem;
    color: var(--text-muted);
    margin-top: 0.2rem;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border-med) !important;
    border-radius: var(--radius-lg) !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

/* ── PLOTLY CHARTS — Fondo blanco ── */
.js-plotly-plot { border-radius: var(--radius-md); overflow: hidden; }

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] textarea {
    background: var(--bg-subtle) !important;
    border-color: var(--border-med) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-md) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(212,82,26,0.1) !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] { border-radius: var(--radius-md) !important; overflow: hidden !important; }

/* ── SPINNER ── */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* ── SLIDER ── */
[data-testid="stSlider"] > div > div > div > div {
    background-color: var(--accent) !important;
}

/* ── HIDE BRANDING ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── ALERT/INFO TEXT STRONG ── */
.audit-text strong { color: var(--text-primary); font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─── SESSION STATE ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False


# ─── PLOTLY THEME — FONDO BLANCO LIMPIO ────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(color="#5C5A55", family="DM Sans", size=12),
    xaxis=dict(
        gridcolor="#F0EFEC",
        linecolor="#E2E0DB",
        tickfont=dict(color="#9B9890", size=11),
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor="#F0EFEC",
        linecolor="#E2E0DB",
        tickfont=dict(color="#9B9890", size=11),
        zeroline=False,
    ),
    margin=dict(l=12, r=12, t=36, b=12),
    hoverlabel=dict(
        bgcolor="#FFFFFF",
        bordercolor="#E2E0DB",
        font=dict(color="#1A1916", family="DM Sans"),
    ),
)

# Paleta principal
C_PRIMARY  = "#D4521A"
C_SECONDARY = "#E8860A"
C_BLUE     = "#2563EB"
COLORS     = [C_PRIMARY, C_SECONDARY, "#2A7D4F", "#2563EB", "#7C3AED", "#0891B2", "#B45309"]


# ─── HERO HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="apex-hero">
    <div class="apex-hero-inner">
        <div>
            <div class="apex-badge">
                <span class="apex-badge-dot"></span>
                Dashboard en vivo
            </div>
            <div class="apex-logo">APEX<span>CO</span> AI</div>
            <div class="apex-tagline">Auditoría inteligente para concesionarios · Powered by Claude AI</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── FILE UPLOAD ────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📂  Carga el archivo de datos del concesionario",
    type=["csv"],
    help="CSV con columnas: fecha, hora, concesionario, vendedor, marca, modelo, precio_lista, descuento, precio_final, visita, venta, canal, estado_lead"
)


# ─── BLOQUE PRINCIPAL ──────────────────────────────────────────────────────────
if uploaded_file is not None:
    try:
        df_raw = load_csv(uploaded_file)

        ok_cols, missing_cols = validate_required_columns(df_raw)
        ok_fmt,  format_errors = validate_basic_format(df_raw)
        ok_num,  num_errors   = validate_numeric_columns(df_raw)

        if not ok_cols:
            st.markdown(f"""
            <div class="audit-card danger">
                <span class="audit-icon">❌</span>
                <span class="audit-text">Faltan columnas requeridas: <strong>{', '.join(missing_cols)}</strong></span>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        if not ok_fmt:
            for err in format_errors:
                st.markdown(f'<div class="audit-card danger"><span class="audit-icon">⚠️</span><span class="audit-text">{err}</span></div>', unsafe_allow_html=True)
            st.stop()

        if not ok_num:
            for err in num_errors:
                st.markdown(f'<div class="audit-card warning"><span class="audit-icon">⚠️</span><span class="audit-text">{err}</span></div>', unsafe_allow_html=True)
            st.stop()

        df = preprocess_data(df_raw)

        # ── ML: entrenamiento temprano ──────────────────────────────────────
        ml_result      = None
        ml_mae         = None
        ml_predictions = []

        has_ml_cols = (
            "hora_int" in df.columns
            and "ventas"  in df.columns
            and "visitas" in df.columns
        )

        if has_ml_cols:
            try:
                ml_result      = train_sales_model(df)
                ml_mae         = ml_result.get("mae")
                ml_predictions = ml_result.get("predictions", [])
            except Exception as ml_err:
                ml_result = {"error": str(ml_err)}

        # ── Calcular todo ───────────────────────────────────────────────────
        metrics         = compute_metrics(df)
        hourly_df       = sales_by_hour(df)
        daily_df        = sales_by_day(df)
        sellers_df      = top_sellers(df)
        brands_df       = top_brands(df)
        low_conv_df     = detect_low_conversion(df)
        patterns        = detect_patterns(df)
        recommendations = generate_recommendations(metrics, low_conv_df, sellers_df, brands_df)

        # generate_recommendations puede retornar (list, ...) o solo list — normalizar
        if isinstance(recommendations, tuple):
            recommendations = recommendations[0]

        st.session_state.data_loaded = True

        # ── KPI CARDS ───────────────────────────────────────────────────────
        st.markdown('<div class="section-eyebrow">Resumen operativo</div>', unsafe_allow_html=True)

        conv_pct   = metrics['conversion']
        conv_cls   = "good"    if conv_pct >= 0.20 else "bad"
        conv_label = "✓ Buena conversión" if conv_pct >= 0.20 else "↓ Conversión baja"

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-accent"></div>
                <span class="metric-icon">🏷️</span>
                <div class="metric-value">{metrics['total_ventas']:.0f}</div>
                <div class="metric-label">Ventas totales</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-accent"></div>
                <span class="metric-icon">👥</span>
                <div class="metric-value">{metrics['total_visitas']:.0f}</div>
                <div class="metric-label">Visitas totales</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-accent"></div>
                <span class="metric-icon">📈</span>
                <div class="metric-value">{conv_pct:.1%}</div>
                <div class="metric-label">Tasa de conversión</div>
                <span class="metric-delta {conv_cls}">{conv_label}</span>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-accent"></div>
                <span class="metric-icon">💰</span>
                <div class="metric-value">${metrics['ticket_promedio']:,.0f}</div>
                <div class="metric-label">Ticket promedio</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="apex-divider"></div>', unsafe_allow_html=True)

        # ── GRÁFICAS ROW 1 ──────────────────────────────────────────────────
        st.markdown('<div class="section-eyebrow">Análisis temporal</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Comportamiento de ventas por hora y por día</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.bar(
                hourly_df, x="hora_int", y="ventas",
                color_discrete_sequence=[C_PRIMARY],
                labels={"hora_int": "Hora", "ventas": "Ventas"}
            )
            fig1.update_layout(**PLOTLY_LAYOUT, title="Ventas por hora del día", bargap=0.28)
            fig1.update_traces(
                marker_line_width=0,
                hovertemplate="<b>%{x}:00 h</b><br>Ventas: <b>%{y}</b><extra></extra>"
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.area(
                hourly_df, x="hora_int", y="conversion",
                color_discrete_sequence=[C_SECONDARY],
                labels={"hora_int": "Hora", "conversion": "Conversión"}
            )
            fig2.update_layout(**PLOTLY_LAYOUT, title="Conversión por hora del día")
            fig2.update_traces(
                line_width=2.5,
                fillcolor="rgba(232,134,10,0.10)",
                hovertemplate="<b>%{x}:00 h</b><br>Conversión: <b>%{y:.1%}</b><extra></extra>"
            )
            st.plotly_chart(fig2, use_container_width=True)

        # ── GRÁFICAS ROW 2 ──────────────────────────────────────────────────
        col3, col4 = st.columns(2)

        with col3:
            fig3 = px.line(
                daily_df, x="fecha_str", y="ventas",
                markers=True,
                color_discrete_sequence=[C_PRIMARY],
                labels={"fecha_str": "Fecha", "ventas": "Ventas"}
            )
            fig3.update_layout(**PLOTLY_LAYOUT, title="Tendencia de ventas diarias")
            fig3.update_traces(
                line_width=2.5,
                marker_size=7,
                marker_color=C_SECONDARY,
                marker_line_color=C_PRIMARY,
                marker_line_width=1.5,
                hovertemplate="<b>%{x}</b><br>Ventas: <b>%{y}</b><extra></extra>"
            )
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            fig4 = px.bar(
                brands_df.head(8), x="ventas", y="marca",
                orientation="h",
                color_discrete_sequence=[C_PRIMARY],
                labels={"ventas": "Ventas", "marca": "Marca"}
            )
            fig4.update_layout(**PLOTLY_LAYOUT, title="Top marcas por ventas", bargap=0.3)
            fig4.update_traces(
                marker_line_width=0,
                hovertemplate="<b>%{y}</b><br>Ventas: <b>%{x}</b><extra></extra>"
            )
            st.plotly_chart(fig4, use_container_width=True)

        # ── GRÁFICAS ROW 3 ──────────────────────────────────────────────────
        col5, col6 = st.columns(2)

        with col5:
            fig5 = px.bar(
                sellers_df.head(8), x="ventas", y="vendedor",
                orientation="h",
                color="ventas",
                color_continuous_scale=[[0, "#FCDCC8"], [1, C_PRIMARY]],
                labels={"ventas": "Ventas", "vendedor": "Vendedor"}
            )
            fig5.update_layout(
                **PLOTLY_LAYOUT,
                title="Ranking de vendedores",
                bargap=0.3,
                coloraxis_showscale=False
            )
            fig5.update_traces(
                marker_line_width=0,
                hovertemplate="<b>%{y}</b><br>Ventas: <b>%{x}</b><extra></extra>"
            )
            st.plotly_chart(fig5, use_container_width=True)

        with col6:
            fig6 = px.pie(
                brands_df.head(6),
                values="ventas",
                names="marca",
                hole=0.62,
                color_discrete_sequence=COLORS
            )
            fig6.update_layout(
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(color="#5C5A55", family="DM Sans"),
                margin=dict(l=10, r=10, t=30, b=10),
                title=dict(text="Distribución por marca (%)", font=dict(color="#1A1916", size=13)),
                showlegend=True,
                legend=dict(
                    font=dict(color="#5C5A55"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                hoverlabel=dict(
                    bgcolor="#FFFFFF",
                    bordercolor="#E2E0DB",
                    font=dict(color="#1A1916", family="DM Sans"),
                ),
            )
            fig6.update_traces(
                textfont_color="white",
                hovertemplate="<b>%{label}</b><br>Ventas: %{value}<br>%{percent}<extra></extra>"
            )
            st.plotly_chart(fig6, use_container_width=True)

        st.markdown('<div class="apex-divider"></div>', unsafe_allow_html=True)

        # ── PATRONES ────────────────────────────────────────────────────────
        st.markdown('<div class="section-eyebrow">Diagnóstico automático</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Patrones detectados en la operación</div>', unsafe_allow_html=True)

        pills_html = '<div class="pills-row">'
        if patterns.get('best_hour') is not None:
            pills_html += f'<span class="pattern-pill good">✓ Mejor hora: {patterns["best_hour"]:02d}:00</span>'
        if patterns.get('worst_hour') is not None:
            pills_html += f'<span class="pattern-pill bad">↓ Peor hora: {patterns["worst_hour"]:02d}:00</span>'
        for h in patterns.get('high_traffic_hours', []):
            pills_html += f'<span class="pattern-pill info">⚡ Alto tráfico: {h:02d}:00</span>'
        pills_html += '</div>'
        st.markdown(pills_html, unsafe_allow_html=True)

        # ── BAJA CONVERSIÓN ─────────────────────────────────────────────────
        if len(low_conv_df) > 0:
            st.markdown(f"""
            <div class="audit-card danger">
                <span class="audit-icon">🔴</span>
                <span class="audit-text">Se detectaron <strong>{len(low_conv_df)} hora(s)</strong> con conversión
                inferior al 15 %. Revisar asignación de personal en esos turnos.</span>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("📋 Ver detalle de horas con baja conversión"):
                st.dataframe(low_conv_df, use_container_width=True)
        else:
            st.markdown("""
            <div class="audit-card success">
                <span class="audit-icon">✅</span>
                <span class="audit-text">No se detectaron horas con conversión baja. La operación se mantiene estable.</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="apex-divider"></div>', unsafe_allow_html=True)

        # ── RECOMENDACIONES ─────────────────────────────────────────────────
        st.markdown('<div class="section-eyebrow">Motor de auditoría</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Recomendaciones accionables</div>', unsafe_allow_html=True)

        icons  = ["🎯", "📊", "👥", "⚡", "💡", "🔧", "📈"]
        levels = ["warning", "danger", "warning", "success", "warning", "danger", "success"]
        for i, rec in enumerate(recommendations):
            icon  = icons[i % len(icons)]
            level = levels[i % len(levels)]
            st.markdown(f"""
            <div class="audit-card {level}">
                <span class="audit-icon">{icon}</span>
                <span class="audit-text">{rec}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="apex-divider"></div>', unsafe_allow_html=True)

        # ── SIMULADOR ───────────────────────────────────────────────────────
        st.markdown('<div class="section-eyebrow">Inteligencia predictiva</div>', unsafe_allow_html=True)
        render_simulator(metrics=metrics, patterns=patterns, sellers_df=sellers_df)

        st.markdown('<div class="apex-divider"></div>', unsafe_allow_html=True)

        # ── ML SECTION ──────────────────────────────────────────────────────
        st.markdown('<div class="section-eyebrow">Machine Learning · Predicción de ventas</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">🧠 Motor predictivo APEX</div>', unsafe_allow_html=True)

        if not has_ml_cols:
            st.markdown("""
            <div class="audit-card warning">
                <span class="audit-icon">⚠️</span>
                <span class="audit-text">El dataset no contiene las columnas <strong>hora</strong>,
                <strong>ventas</strong> y <strong>visitas</strong> necesarias para entrenar el modelo predictivo.</span>
            </div>
            """, unsafe_allow_html=True)

        elif ml_result and "error" in ml_result:
            st.markdown(f"""
            <div class="audit-card danger">
                <span class="audit-icon">❌</span>
                <span class="audit-text">Error al entrenar el modelo: <strong>{ml_result['error']}</strong></span>
            </div>
            """, unsafe_allow_html=True)

        else:
            ml_mae         = ml_result.get("mae", 0)
            ml_predictions = ml_result.get("predictions", [])

            accuracy_pct = max(0.0, min(100.0, 100.0 - (ml_mae / max(metrics['total_ventas'], 1)) * 100)) if ml_mae is not None else 0.0
            acc_cls  = "good"   if accuracy_pct >= 80  else ("warn" if accuracy_pct >= 60 else "accent")
            mae_cls  = "good"   if ml_mae <= 2          else ("warn" if ml_mae <= 5 else "accent")
            mae_bar  = max(10,  100 - ml_mae * 10)
            acc_bar  = accuracy_pct

            preds_to_show = ml_predictions[:8]
            pred_cards_html = ""
            for p in preds_to_show:
                hora  = p.get("hora", 0)
                valor = p.get("ventas_pred", p.get("pred", 0))
                pred_cards_html += f"""
                    <div class="ml-predict-card">
                        <div class="ml-predict-hour">{hora:02d}:00 h</div>
                        <div class="ml-predict-val">{valor:.1f}</div>
                        <div class="ml-predict-label">ventas est.</div>
                    </div>"""

            st.markdown(f"""
            <div class="ml-section">
                <div class="ml-header">
                    <div class="ml-icon-wrap">🧠</div>
                    <div>
                        <div class="ml-title">Modelo predictivo entrenado</div>
                        <div class="ml-subtitle">Regresión supervisada sobre datos históricos del concesionario</div>
                    </div>
                    <div class="ml-badge">✓ Activo</div>
                </div>

                <div class="ml-stats-row">
                    <div class="ml-stat-card">
                        <div class="ml-stat-val {mae_cls}">{ml_mae:.2f}</div>
                        <div class="ml-stat-lbl">Error promedio (MAE)</div>
                        <div class="ml-accuracy-bar-wrap">
                            <div class="ml-accuracy-label"><span>Precisión del error</span><span>{mae_bar:.0f}%</span></div>
                            <div class="ml-accuracy-bar">
                                <div class="ml-accuracy-fill" style="width:{mae_bar:.0f}%"></div>
                            </div>
                        </div>
                    </div>
                    <div class="ml-stat-card">
                        <div class="ml-stat-val {acc_cls}">{accuracy_pct:.1f}%</div>
                        <div class="ml-stat-lbl">Precisión estimada</div>
                        <div class="ml-accuracy-bar-wrap">
                            <div class="ml-accuracy-label"><span>Precisión</span><span>{acc_bar:.0f}%</span></div>
                            <div class="ml-accuracy-bar">
                                <div class="ml-accuracy-fill" style="width:{acc_bar:.0f}%"></div>
                            </div>
                        </div>
                    </div>
                    <div class="ml-stat-card">
                        <div class="ml-stat-val accent">{len(ml_predictions)}</div>
                        <div class="ml-stat-lbl">Predicciones generadas</div>
                        <div class="ml-accuracy-bar-wrap">
                            <div class="ml-accuracy-label"><span>Cobertura</span><span>100%</span></div>
                            <div class="ml-accuracy-bar">
                                <div class="ml-accuracy-fill" style="width:100%"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="ml-predict-row">
                    {pred_cards_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Gráfico Real vs Predicho
            if ml_predictions:
                import pandas as _pd
                pred_df  = _pd.DataFrame(ml_predictions)
                real_col = "ventas"      if "ventas"      in pred_df.columns else None
                pred_col = "ventas_pred" if "ventas_pred" in pred_df.columns else ("pred" if "pred" in pred_df.columns else None)
                hora_col = "hora"        if "hora"        in pred_df.columns else None

                if real_col and pred_col and hora_col:
                    fig_ml = go.Figure()
                    fig_ml.add_trace(go.Scatter(
                        x=pred_df[hora_col], y=pred_df[real_col],
                        name="Real", mode="lines+markers",
                        line=dict(color=C_PRIMARY, width=2.5),
                        marker=dict(size=7, color=C_SECONDARY),
                        hovertemplate="<b>%{x}:00 h</b><br>Real: <b>%{y:.1f}</b><extra></extra>"
                    ))
                    fig_ml.add_trace(go.Scatter(
                        x=pred_df[hora_col], y=pred_df[pred_col],
                        name="Predicho", mode="lines+markers",
                        line=dict(color=C_BLUE, width=2.5, dash="dot"),
                        marker=dict(size=7, color=C_BLUE),
                        hovertemplate="<b>%{x}:00 h</b><br>Predicho: <b>%{y:.1f}</b><extra></extra>"
                    ))
                    fig_ml.update_layout(
                        **PLOTLY_LAYOUT,
                        title="Real vs. Predicho por hora",
                        legend=dict(
                            font=dict(color="#5C5A55"),
                            bgcolor="rgba(0,0,0,0)",
                            orientation="h",
                            yanchor="bottom", y=1.02,
                            xanchor="right", x=1,
                        )
                    )
                    st.plotly_chart(fig_ml, use_container_width=True)

        st.markdown('<div class="apex-divider"></div>', unsafe_allow_html=True)

        # ── DATOS RAW ───────────────────────────────────────────────────────
        with st.expander("📋 Vista previa de datos cargados"):
            st.dataframe(df, use_container_width=True)

        st.markdown('<div class="apex-divider"></div>', unsafe_allow_html=True)

        # ── INSIGHTS IA ─────────────────────────────────────────────────────
        st.markdown('<div class="section-eyebrow">Inteligencia aumentada</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">🧠 Insights inteligentes</div>', unsafe_allow_html=True)

        insights = []

        if metrics["conversion"] < 0.15:
            insights.append(("danger", "⚠️",
                "La tasa de conversión general es baja. Se recomienda revisar el seguimiento comercial y la calidad de los leads."))

        if low_conv_df is not None and len(low_conv_df) > 0:
            horas = low_conv_df["hora_int"].tolist()[:5]
            insights.append(("warning", "📉",
                f"Se detectaron horarios críticos con baja conversión: <strong>{horas}</strong>"))

        if sellers_df is not None and len(sellers_df) > 0:
            top_seller = sellers_df.iloc[0]
            insights.append(("success", "🏆",
                f"<strong>{top_seller['vendedor']}</strong> lidera el rendimiento comercial con <strong>{int(top_seller['ventas'])}</strong> ventas."))

        if brands_df is not None and len(brands_df) > 0:
            top_brand = brands_df.iloc[0]
            insights.append(("info", "🚗",
                f"La marca con mejor desempeño actual es <strong>{top_brand['marca']}</strong>."))

        if not insights:
            st.markdown("""
            <div class="audit-card success">
                <span class="audit-icon">✅</span>
                <span class="audit-text">No se detectaron alertas críticas. La operación muestra indicadores saludables.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            for level, icon, text in insights:
                st.markdown(f"""
                <div class="audit-card {level}">
                    <span class="audit-icon">{icon}</span>
                    <span class="audit-text">{text}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="apex-divider"></div>', unsafe_allow_html=True)

        # ── CHAT ────────────────────────────────────────────────────────────
        st.markdown("""
        <div class="status-badge">
            <span class="status-dot"></span>
            APEX AI LOCAL ENGINE — Ollama · Privado · Sin consumo de créditos
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="chat-wrapper">
            <div class="chat-header">
                <div class="chat-header-dot"></div>
                <div>
                    <div class="chat-header-title">APEX — Asistente de Auditoría</div>
                    <div class="chat-header-sub">Pregúntame sobre ventas, conversión, vendedores, horarios o estrategia</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Historial de mensajes
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="msg-label user">Tú</div>
                <div class="msg-user">{msg["content"]}</div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="msg-label apex">⬡ APEX</div>
                <div class="msg-assistant">{msg["content"]}</div>
                """, unsafe_allow_html=True)

        # Input del chat
        prompt = st.chat_input("Pregúntale a APEX sobre tu operación...")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.spinner("APEX está analizando..."):
                response = get_chat_response(
                    user_text=prompt,
                    metrics=metrics,
                    recommendations=recommendations,
                    patterns=patterns,
                    sellers_df=sellers_df,
                    brands_df=brands_df,
                    low_conversion_df=low_conv_df,
                    conversation_history=st.session_state.messages[:-1]
                )

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    except Exception as e:
        st.markdown(f"""
        <div class="audit-card danger">
            <span class="audit-icon">❌</span>
            <span class="audit-text">Error al procesar el archivo: <strong>{e}</strong></span>
        </div>
        """, unsafe_allow_html=True)
        import traceback
        with st.expander("🔍 Ver detalle del error"):
            st.code(traceback.format_exc(), language="python")

else:
    # ── LANDING VACÍO ───────────────────────────────────────────────────────
    st.markdown("""
    <div class="landing-empty">
        <span class="landing-icon">🚗</span>
        <div class="landing-title">Carga los datos de tu concesionario</div>
        <div class="landing-sub">
            Sube un CSV con tus registros de ventas y visitas.
            APEX analizará tu operación en segundos y generará
            recomendaciones accionables con inteligencia artificial.
        </div>
        <div class="landing-features">
            <span class="pattern-pill info">📊 Métricas en tiempo real</span>
            <span class="pattern-pill good">🤖 IA conversacional</span>
            <span class="pattern-pill bad">🔍 Detección de pérdidas</span>
            <span class="pattern-pill info">🎯 Recomendaciones accionables</span>
            <span class="pattern-pill good">⚡ Diagnóstico automático</span>
            <span class="pattern-pill good">🧠 Motor ML predictivo</span>
        </div>
    </div>
    """, unsafe_allow_html=True)