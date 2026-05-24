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
# ─── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="APEXCO AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

/* ── RESET & BASE ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #07070f;
    color: #ddddf0;
}
.stApp { background: #07070f; }
*, *::before, *::after { box-sizing: border-box; }

/* ── HERO ── */
.apex-hero {
    background: linear-gradient(135deg, #0c0c1e 0%, #07070f 55%, #0e0d16 100%);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 3rem 2.5rem 2.5rem;
    margin: -1rem -1rem 2.5rem -1rem;
    position: relative;
    overflow: hidden;
}
.apex-hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 420px; height: 420px;
    background: radial-gradient(circle, rgba(255,75,20,0.13) 0%, transparent 68%);
    border-radius: 50%;
    pointer-events: none;
}
.apex-hero::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 18%;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(255,160,0,0.08) 0%, transparent 68%);
    border-radius: 50%;
    pointer-events: none;
}
.apex-hero-inner {
    position: relative;
    z-index: 1;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
}
.apex-logo-block {}
.apex-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255,75,20,0.12);
    border: 1px solid rgba(255,75,20,0.35);
    color: #FF7043;
    font-size: 0.68rem;
    padding: 0.22rem 0.75rem;
    border-radius: 100px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
}
.apex-badge-dot {
    width: 6px; height: 6px;
    background: #FF5722;
    border-radius: 50%;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.7); }
}
.apex-logo {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #ffffff;
    line-height: 1;
    margin: 0;
}
.apex-logo span { color: #FF5722; }
.apex-tagline {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.38);
    margin-top: 0.5rem;
    font-weight: 400;
    letter-spacing: 0.09em;
    text-transform: uppercase;
}
.apex-hero-stats {
    display: flex;
    gap: 2rem;
    flex-wrap: wrap;
}
.hero-stat {
    text-align: right;
}
.hero-stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #fff;
    line-height: 1;
}
.hero-stat-lbl {
    font-size: 0.65rem;
    color: rgba(255,255,255,0.3);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.2rem;
}

/* ── METRIC CARDS ── */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(145deg, #111120, #0e0e1a);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 1.5rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s ease, transform 0.2s ease;
}
.metric-card:hover {
    border-color: rgba(255,87,34,0.35);
    transform: translateY(-2px);
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #FF5722, #FF9800);
    opacity: 0;
    transition: opacity 0.25s;
    border-radius: 18px 18px 0 0;
}
.metric-card:hover::after { opacity: 1; }
.metric-card-bg {
    position: absolute;
    top: -20px; right: -20px;
    font-size: 4rem;
    opacity: 0.05;
    pointer-events: none;
    line-height: 1;
}
.metric-icon {
    font-size: 1.3rem;
    margin-bottom: 0.8rem;
    display: block;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1;
    margin-bottom: 0.35rem;
}
.metric-label {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.38);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 500;
}
.metric-delta {
    font-size: 0.74rem;
    margin-top: 0.5rem;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.15rem 0.55rem;
    border-radius: 100px;
    font-weight: 500;
}
.metric-delta.good {
    background: rgba(76,175,80,0.12);
    color: #66BB6A;
    border: 1px solid rgba(76,175,80,0.25);
}
.metric-delta.bad {
    background: rgba(255,87,34,0.12);
    color: #FF7043;
    border: 1px solid rgba(255,87,34,0.25);
}

/* ── SECTION LABELS ── */
.section-label {
    font-size: 0.65rem;
    color: rgba(255,87,34,0.7);
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin-bottom: 0.3rem;
    font-weight: 600;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 1rem 0;
    letter-spacing: -0.01em;
}

/* ── CHART WRAPPERS ── */
.chart-card {
    background: linear-gradient(145deg, #111120, #0e0e1a);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 1.4rem 1.4rem 0.8rem;
    margin-bottom: 1rem;
}

/* ── AUDIT / INSIGHT CARDS ── */
.audit-card {
    background: #10101e;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 13px;
    padding: 1.05rem 1.3rem;
    margin-bottom: 0.65rem;
    display: flex;
    align-items: flex-start;
    gap: 0.85rem;
    transition: border-color 0.2s;
}
.audit-card:hover { border-color: rgba(255,255,255,0.14); }
.audit-icon { font-size: 1.1rem; flex-shrink: 0; margin-top: 0.08rem; }
.audit-text {
    font-size: 0.875rem;
    color: rgba(255,255,255,0.72);
    line-height: 1.6;
}
.audit-card.warning {
    border-left: 3px solid #FF9800;
    background: rgba(255,152,0,0.04);
}
.audit-card.danger {
    border-left: 3px solid #FF5722;
    background: rgba(255,87,34,0.04);
}
.audit-card.success {
    border-left: 3px solid #4CAF50;
    background: rgba(76,175,80,0.04);
}
.audit-card.info {
    border-left: 3px solid #29B6F6;
    background: rgba(41,182,246,0.04);
}

/* ── INSIGHTS GRID ── */
.insights-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem;
    margin-bottom: 1rem;
}

/* ── PATTERN PILLS ── */
.pills-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 1.2rem; }
.pattern-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 100px;
    padding: 0.38rem 1rem;
    font-size: 0.8rem;
    color: rgba(255,255,255,0.65);
    white-space: nowrap;
}
.pattern-pill.good  { border-color: rgba(76,175,80,0.4);  color: #81C784; background: rgba(76,175,80,0.06); }
.pattern-pill.bad   { border-color: rgba(255,87,34,0.4);  color: #FF7043; background: rgba(255,87,34,0.06); }
.pattern-pill.info  { border-color: rgba(255,152,0,0.4);  color: #FFB74D; background: rgba(255,152,0,0.06); }

/* ── CHAT ── */
.chat-container {
    background: linear-gradient(145deg, #111120, #0e0e1a);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    overflow: hidden;
    margin-bottom: 1rem;
}
.chat-header {
    background: linear-gradient(90deg, rgba(255,87,34,0.14), rgba(255,152,0,0.07));
    border-bottom: 1px solid rgba(255,255,255,0.07);
    padding: 1.1rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.chat-dot {
    width: 9px; height: 9px;
    background: #4CAF50;
    border-radius: 50%;
    box-shadow: 0 0 10px rgba(76,175,80,0.7);
    flex-shrink: 0;
}
.chat-title { font-family: 'Syne', sans-serif; font-size: 0.95rem; font-weight: 700; color: #fff; }
.chat-subtitle { font-size: 0.72rem; color: rgba(255,255,255,0.38); margin-top: 0.1rem; }
.chat-body { padding: 1.2rem 1.4rem; display: flex; flex-direction: column; gap: 0.6rem; }
.msg-user {
    background: rgba(255,87,34,0.1);
    border: 1px solid rgba(255,87,34,0.2);
    border-radius: 14px 14px 4px 14px;
    padding: 0.75rem 1.1rem;
    margin-left: 2.5rem;
    font-size: 0.875rem;
    color: rgba(255,255,255,0.88);
    line-height: 1.55;
}
.msg-assistant {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px 14px 14px 4px;
    padding: 0.75rem 1.1rem;
    margin-right: 2.5rem;
    font-size: 0.875rem;
    color: rgba(255,255,255,0.82);
    line-height: 1.65;
}
.msg-label { font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.11em; margin-bottom: 0.22rem; font-weight: 600; }
.msg-label.user { color: #FF7043; }
.msg-label.apex { color: #FFB74D; }

/* ── STATUS BADGE ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(76,175,80,0.1);
    border: 1px solid rgba(76,175,80,0.25);
    color: #66BB6A;
    font-size: 0.75rem;
    padding: 0.4rem 1rem;
    border-radius: 100px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.status-dot {
    width: 6px; height: 6px;
    background: #4CAF50;
    border-radius: 50%;
    box-shadow: 0 0 6px #4CAF50;
}

/* ── DIVIDER ── */
.apex-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.07), transparent);
    margin: 2.5rem 0;
}

/* ── LANDING EMPTY ── */
.landing-empty {
    text-align: center;
    padding: 5rem 2rem;
}
.landing-icon {
    font-size: 3.5rem;
    margin-bottom: 1.2rem;
    display: block;
}
.landing-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 0.6rem;
}
.landing-sub {
    color: rgba(255,255,255,0.38);
    font-size: 0.9rem;
    max-width: 440px;
    margin: 0 auto 2rem;
    line-height: 1.75;
}
.landing-features {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1.5rem;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: rgba(255,87,34,0.04) !important;
    border: 2px dashed rgba(255,87,34,0.3) !important;
    border-radius: 14px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(255,87,34,0.55) !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] { border-radius: 14px !important; overflow: hidden !important; }

/* ── HIDE STREAMLIT BRANDING ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #07070f; }
::-webkit-scrollbar-thumb { background: #2a2a3a; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #3a3a4a; }

/* ── PLOTLY CHARTS ── */
.js-plotly-plot { border-radius: 14px; overflow: hidden; }

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] textarea {
    background: #1a1a2e !important;
    border-color: rgba(255,87,34,0.3) !important;
    color: #ddddf0 !important;
    border-radius: 14px !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(255,87,34,0.6) !important;
    box-shadow: 0 0 0 3px rgba(255,87,34,0.1) !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: #10101e !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
}

/* ── SPINNER ── */
[data-testid="stSpinner"] { color: #FF7043 !important; }

/* ── ML PREDICTIVE SECTION ── */
.ml-section {
    background: linear-gradient(145deg, #0f0f1e, #0c0c18);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 1.8rem 2rem;
    position: relative;
    overflow: hidden;
    margin-bottom: 1rem;
}
.ml-section::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(99,102,241,0.1) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.ml-header {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    margin-bottom: 1.4rem;
}
.ml-icon-wrap {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.15));
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    flex-shrink: 0;
}
.ml-title-block {}
.ml-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #fff;
    line-height: 1;
}
.ml-subtitle {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.35);
    margin-top: 0.2rem;
    letter-spacing: 0.06em;
}
.ml-badge {
    margin-left: auto;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.3);
    color: #818CF8;
    font-size: 0.68rem;
    padding: 0.22rem 0.75rem;
    border-radius: 100px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.ml-stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.9rem;
    margin-bottom: 1.4rem;
}
.ml-stat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    text-align: center;
    transition: border-color 0.2s;
}
.ml-stat-card:hover { border-color: rgba(99,102,241,0.3); }
.ml-stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: #fff;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.ml-stat-val.accent { color: #818CF8; }
.ml-stat-val.good   { color: #66BB6A; }
.ml-stat-val.warn   { color: #FFB74D; }
.ml-stat-lbl {
    font-size: 0.68rem;
    color: rgba(255,255,255,0.35);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 500;
}
.ml-accuracy-bar-wrap {
    margin-top: 0.8rem;
}
.ml-accuracy-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: rgba(255,255,255,0.4);
    margin-bottom: 0.4rem;
}
.ml-accuracy-bar {
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    overflow: hidden;
}
.ml-accuracy-fill {
    height: 100%;
    background: linear-gradient(90deg, #6366F1, #818CF8);
    border-radius: 100px;
    transition: width 1s ease;
}
.ml-predict-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
}
.ml-predict-card {
    background: rgba(99,102,241,0.06);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 13px;
    padding: 1rem 1.1rem;
    text-align: center;
}
.ml-predict-hour {
    font-size: 0.65rem;
    color: rgba(255,255,255,0.35);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}
.ml-predict-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #818CF8;
    line-height: 1;
}
.ml-predict-label {
    font-size: 0.65rem;
    color: rgba(255,255,255,0.3);
    margin-top: 0.2rem;
}
.ml-error-card {
    background: rgba(255,87,34,0.06);
    border: 1px solid rgba(255,87,34,0.2);
    border-radius: 13px;
    padding: 1rem 1.3rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    font-size: 0.85rem;
    color: rgba(255,255,255,0.65);
}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# ─── PLOTLY THEME ──────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="rgba(255,255,255,0.5)", family="DM Sans", size=12),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        linecolor="rgba(255,255,255,0.08)",
        tickfont=dict(color="rgba(255,255,255,0.4)"),
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        linecolor="rgba(255,255,255,0.08)",
        tickfont=dict(color="rgba(255,255,255,0.4)"),
        zeroline=False,
    ),
    margin=dict(l=10, r=10, t=30, b=10),
    hoverlabel=dict(
        bgcolor="#1a1a2e",
        bordercolor="rgba(255,87,34,0.4)",
        font=dict(color="#fff", family="DM Sans"),
    ),
)
ORANGE  = "#FF5722"
ORANGE2 = "#FF9800"
COLORS  = [ORANGE, ORANGE2, "#E91E63", "#AB47BC", "#5C6BC0", "#26C6DA", "#66BB6A"]

# ─── HERO HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="apex-hero">
    <div class="apex-hero-inner">
        <div class="apex-logo-block">
            <div class="apex-badge">
                <span class="apex-badge-dot"></span>
                Live Dashboard
            </div>
            <div class="apex-logo">APEX<span>CO</span> AI</div>
            <div class="apex-tagline">Auditoría inteligente para concesionarios · Powered by Claude AI</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── UPLOAD ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Carga tu archivo de datos del concesionario",
    type=["csv"],
    help="Formato CSV con columnas: fecha, hora, concesionario, vendedor, marca, modelo, precio_lista, descuento, precio_final, visita, venta, canal, estado_lead"
)

# ───────────────────────────────────────────────────────────────────────────────
# MAIN BLOCK — todo el análisis vive aquí dentro (indentación correcta)
# ───────────────────────────────────────────────────────────────────────────────
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

        # ── IA PREDICTIVA — entrenamiento temprano ──────────────────────────
        ml_result  = None
        ml_mae     = None
        ml_predictions = []

        has_ml_cols = (
    "hora_int" in df.columns
    and "ventas" in df.columns
    and "visitas" in df.columns
)
        # ── IA PREDICTIVA — entrenamiento
        if has_ml_cols:
            try:
                ml_result      = train_sales_model(df)
                ml_mae         = ml_result.get("mae")
                ml_predictions = ml_result.get("predictions", [])
            except Exception as ml_err:
                ml_result = {"error": str(ml_err)}

        # ── Compute everything ──────────────────────────────────────────────
        metrics         = compute_metrics(df)
        hourly_df       = sales_by_hour(df)
        daily_df        = sales_by_day(df)
        sellers_df      = top_sellers(df)
        brands_df       = top_brands(df)
        low_conv_df     = detect_low_conversion(df)
        patterns        = detect_patterns(df)
        recommendations = generate_recommendations(metrics, low_conv_df, sellers_df, brands_df)

        st.session_state.data_loaded = True

        # ── KPI CARDS ───────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Resumen operativo</div>', unsafe_allow_html=True)

        conv_pct    = metrics['conversion']
        conv_cls    = "good" if conv_pct >= 0.20 else "bad"
        conv_label  = "✓ Buena conversión" if conv_pct >= 0.20 else "↓ Conversión baja"

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-card-bg">🏷️</span>
                <span class="metric-icon">🏷️</span>
                <div class="metric-value">{metrics['total_ventas']:.0f}</div>
                <div class="metric-label">Ventas totales</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-card-bg">👥</span>
                <span class="metric-icon">👥</span>
                <div class="metric-value">{metrics['total_visitas']:.0f}</div>
                <div class="metric-label">Visitas totales</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-card-bg">📈</span>
                <span class="metric-icon">📈</span>
                <div class="metric-value">{conv_pct:.1%}</div>
                <div class="metric-label">Conversión</div>
                <span class="metric-delta {conv_cls}">{conv_label}</span>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-card-bg">💰</span>
                <span class="metric-icon">💰</span>
                <div class="metric-value">${metrics['ticket_promedio']:,.0f}</div>
                <div class="metric-label">Ticket promedio</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="apex-divider"></div>', unsafe_allow_html=True)

        # ── CHARTS ROW 1 ────────────────────────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-label">Análisis temporal</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Ventas por hora</div>', unsafe_allow_html=True)
            fig1 = px.bar(hourly_df, x="hora_int", y="ventas", color_discrete_sequence=[ORANGE])
            fig1.update_layout(**PLOTLY_LAYOUT, title=None, bargap=0.25)
            fig1.update_traces(marker_line_width=0, hovertemplate="<b>%{x}:00h</b><br>Ventas: %{y}<extra></extra>")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown('<div class="section-label">Eficiencia</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Conversión por hora</div>', unsafe_allow_html=True)
            fig2 = px.area(hourly_df, x="hora_int", y="conversion", color_discrete_sequence=[ORANGE2])
            fig2.update_layout(**PLOTLY_LAYOUT, title=None)
            fig2.update_traces(
                line_width=2.5,
                fillcolor="rgba(255,152,0,0.08)",
                hovertemplate="<b>%{x}:00h</b><br>Conversión: %{y:.1%}<extra></extra>"
            )
            st.plotly_chart(fig2, use_container_width=True)

        # ── CHARTS ROW 2 ────────────────────────────────────────────────────
        col3, col4 = st.columns(2)

        with col3:
            st.markdown('<div class="section-label">Tendencia</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Ventas por día</div>', unsafe_allow_html=True)
            fig3 = px.line(daily_df, x="fecha_str", y="ventas", markers=True, color_discrete_sequence=[ORANGE])
            fig3.update_layout(**PLOTLY_LAYOUT, title=None)
            fig3.update_traces(
                line_width=2.5,
                marker_size=7,
                marker_color=ORANGE2,
                marker_line_color=ORANGE,
                marker_line_width=1.5,
                hovertemplate="<b>%{x}</b><br>Ventas: %{y}<extra></extra>"
            )
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            st.markdown('<div class="section-label">Portafolio</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Top marcas</div>', unsafe_allow_html=True)
            fig4 = px.bar(
                brands_df.head(8), x="ventas", y="marca",
                orientation='h', color_discrete_sequence=[ORANGE]
            )
            fig4.update_layout(**PLOTLY_LAYOUT, title=None, bargap=0.3)
            fig4.update_traces(
                marker_line_width=0,
                hovertemplate="<b>%{y}</b><br>Ventas: %{x}<extra></extra>"
            )
            st.plotly_chart(fig4, use_container_width=True)

        # ── CHARTS ROW 3 — Sellers + Donut ──────────────────────────────────
        col5, col6 = st.columns(2)

        with col5:
            st.markdown('<div class="section-label">Equipo comercial</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Ranking de vendedores</div>', unsafe_allow_html=True)
            fig5 = px.bar(
                sellers_df.head(8), x="ventas", y="vendedor",
                orientation='h', color="ventas",
                color_continuous_scale=[[0, "#FF5722"], [1, "#FF9800"]]
            )
            fig5.update_layout(**PLOTLY_LAYOUT, title=None, bargap=0.3, coloraxis_showscale=False)
            fig5.update_traces(
                marker_line_width=0,
                hovertemplate="<b>%{y}</b><br>Ventas: %{x}<extra></extra>"
            )
            st.plotly_chart(fig5, use_container_width=True)

        with col6:
            st.markdown('<div class="section-label">Distribución</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Ventas por marca (%)</div>', unsafe_allow_html=True)
            fig6 = px.pie(
                brands_df.head(6), values="ventas", names="marca",
                hole=0.62, color_discrete_sequence=COLORS
            )
            fig6.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="rgba(255,255,255,0.55)", family="DM Sans"),
                margin=dict(l=10, r=10, t=20, b=10),
                showlegend=True,
                legend=dict(
                    font=dict(color="rgba(255,255,255,0.55)"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                hoverlabel=dict(
                    bgcolor="#1a1a2e",
                    bordercolor="rgba(255,87,34,0.4)",
                    font=dict(color="#fff", family="DM Sans"),
                ),
            )
            fig6.update_traces(
                textfont_color="white",
                hovertemplate="<b>%{label}</b><br>Ventas: %{value}<br>%{percent}<extra></extra>"
            )
            st.plotly_chart(fig6, use_container_width=True)

        st.markdown('<div class="apex-divider"></div>', unsafe_allow_html=True)

        # ── PATRONES ────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Diagnóstico automático</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Patrones detectados</div>', unsafe_allow_html=True)

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
                inferior al 15%. Revisar asignación de personal en esos turnos.</span>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("📋 Ver detalle de horas con baja conversión"):
                st.dataframe(low_conv_df, use_container_width=True)
        else:
            st.markdown("""
            <div class="audit-card success">
                <span class="audit-icon">✅</span>
                <span class="audit-text">No se detectaron horas con conversión baja. Operación estable.</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="apex-divider"></div>', unsafe_allow_html=True)

        # ── RECOMENDACIONES ─────────────────────────────────────────────────
        st.markdown('<div class="section-label">Motor de auditoría</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Recomendaciones accionables</div>', unsafe_allow_html=True)

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
        st.markdown('<div class="section-label">Inteligencia predictiva</div>', unsafe_allow_html=True)
        render_simulator(metrics=metrics, patterns=patterns, sellers_df=sellers_df)

        st.markdown('<div class="apex-divider"></div>', unsafe_allow_html=True)

        # ── MODELO IA PREDICTIVO ─────────────────────────────────────────────
        st.markdown('<div class="section-label">Machine Learning · Predicción de ventas</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧠 Motor predictivo APEX</div>', unsafe_allow_html=True)

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
            <div class="ml-error-card">
                <span style="font-size:1.3rem">❌</span>
                <span>Error al entrenar el modelo: <strong>{ml_result['error']}</strong></span>
            </div>
            """, unsafe_allow_html=True)

        else:
            # ── Calcular métricas de precisión ──────────────────────────────
            accuracy_pct = max(0.0, min(100.0, 100.0 - (ml_mae / max(metrics['total_ventas'], 1)) * 100)) if ml_mae is not None else 0.0
            acc_cls  = "good" if accuracy_pct >= 80 else ("warn" if accuracy_pct >= 60 else "accent")
            mae_cls  = "good" if ml_mae <= 2 else ("warn" if ml_mae <= 5 else "accent")

            st.markdown(f"""
            <div class="ml-section">
                <div class="ml-header">
                    <div class="ml-icon-wrap">🧠</div>
                    <div class="ml-title-block">
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
                            <div class="ml-accuracy-bar">
                                <div class="ml-accuracy-fill" style="width:{max(10, 100 - ml_mae*10):.0f}%"></div>
                            </div>
                        </div>
                    </div>
                    <div class="ml-stat-card">
                        <div class="ml-stat-val {acc_cls}">{accuracy_pct:.1f}%</div>
                        <div class="ml-stat-lbl">Precisión estimada</div>
                        <div class="ml-accuracy-bar-wrap">
                            <div class="ml-accuracy-bar">
                                <div class="ml-accuracy-fill" style="width:{accuracy_pct:.0f}%"></div>
                            </div>
                        </div>
                    </div>
                    <div class="ml-stat-card">
                        <div class="ml-stat-val accent">{len(ml_predictions) if ml_predictions else '—'}</div>
                        <div class="ml-stat-lbl">Predicciones generadas</div>
                        <div class="ml-accuracy-bar-wrap">
                            <div class="ml-accuracy-bar">
                                <div class="ml-accuracy-fill" style="width:100%"></div>
                            </div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # ── Predicciones por hora (si existen) ──────────────────────────
            if ml_predictions and len(ml_predictions) > 0:
                preds_to_show = ml_predictions[:8]
                pred_cards = ""
                for p in preds_to_show:
                    hora  = p.get("hora", "?")
                    valor = p.get("ventas_pred", p.get("pred", 0))
                    pred_cards += f"""
                    <div class="ml-predict-card">
                        <div class="ml-predict-hour">{hora:02d}:00h</div>
                        <div class="ml-predict-val">{valor:.1f}</div>
                        <div class="ml-predict-label">ventas est.</div>
                    </div>
                    """
                st.markdown(f'<div class="ml-predict-row">{pred_cards}</div>', unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)  # cierra .ml-section

            # ── Chart: Real vs Predicho ──────────────────────────────────────
            if ml_predictions and len(ml_predictions) > 0:
                pred_df = pd.DataFrame(ml_predictions)
                real_col = "ventas" if "ventas" in pred_df.columns else None
                pred_col = "ventas_pred" if "ventas_pred" in pred_df.columns else ("pred" if "pred" in pred_df.columns else None)
                hora_col = "hora" if "hora" in pred_df.columns else None

                if real_col and pred_col and hora_col:
                    fig_ml = go.Figure()
                    fig_ml.add_trace(go.Scatter(
                        x=pred_df[hora_col], y=pred_df[real_col],
                        name="Real",
                        mode="lines+markers",
                        line=dict(color=ORANGE, width=2.5),
                        marker=dict(size=7, color=ORANGE2),
                        hovertemplate="<b>%{x}:00h</b><br>Real: %{y:.1f}<extra></extra>"
                    ))
                    fig_ml.add_trace(go.Scatter(
                        x=pred_df[hora_col], y=pred_df[pred_col],
                        name="Predicho",
                        mode="lines+markers",
                        line=dict(color="#818CF8", width=2.5, dash="dot"),
                        marker=dict(size=7, color="#6366F1"),
                        hovertemplate="<b>%{x}:00h</b><br>Predicho: %{y:.1f}<extra></extra>"
                    ))
                    fig_ml.update_layout(
                        **PLOTLY_LAYOUT,
                        title=None,
                        legend=dict(
                            font=dict(color="rgba(255,255,255,0.55)"),
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
        st.markdown('<div class="section-label">Inteligencia aumentada</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧠 Insights Inteligentes</div>', unsafe_allow_html=True)

        insights = []

        if metrics["conversion"] < 0.15:
            insights.append(("danger", "⚠️", "La tasa de conversión general es baja. Se recomienda revisar seguimiento comercial y calidad de leads."))

        if low_conv_df is not None and len(low_conv_df) > 0:
            horas = low_conv_df["hora_int"].tolist()[:5]
            insights.append(("warning", "📉", f"Se detectaron horarios críticos con baja conversión: <strong>{horas}</strong>"))

        if sellers_df is not None and len(sellers_df) > 0:
            top_seller = sellers_df.iloc[0]
            insights.append(("success", "🏆", f"<strong>{top_seller['vendedor']}</strong> lidera el rendimiento comercial con <strong>{top_seller['ventas']}</strong> ventas."))

        if brands_df is not None and len(brands_df) > 0:
            top_brand = brands_df.iloc[0]
            insights.append(("info", "🚗", f"La marca con mejor desempeño actual es <strong>{top_brand['marca']}</strong>."))

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
            APEX AI LOCAL ENGINE — Ollama + Gemma 3 · Privado · Sin consumo de créditos
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="chat-container">
            <div class="chat-header">
                <div class="chat-dot"></div>
                <div>
                    <div class="chat-title">APEX — Asistente de Auditoría</div>
                    <div class="chat-subtitle">Pregúntame sobre ventas, conversión, vendedores, horarios o estrategia</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Render historial
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

        # Input
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

else:
    # ── LANDING VACÍO ───────────────────────────────────────────────────────
    st.markdown("""
    <div class="landing-empty">
        <span class="landing-icon">🚗</span>
        <div class="landing-title">Carga los datos de tu concesionario</div>
        <div class="landing-sub">
            Sube un CSV con tus registros de ventas y visitas.<br>
            APEX analizará tu operación en segundos y generará<br>recomendaciones accionables con IA.
        </div>
        <div class="landing-features">
            <span class="pattern-pill info">📊 Métricas en tiempo real</span>
            <span class="pattern-pill good">🤖 IA conversacional</span>
            <span class="pattern-pill bad">🔍 Detección de pérdidas</span>
            <span class="pattern-pill info">🎯 Recomendaciones accionables</span>
            <span class="pattern-pill good">⚡ Diagnóstico automático</span>
        </div>
    </div>
    """, unsafe_allow_html=True) 