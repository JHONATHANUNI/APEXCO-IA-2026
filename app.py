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


# ─── GLOBAL CSS — TEMA OSCURO PROFESIONAL ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&family=Geist:wght@300;400;500;600&display=swap');

/* ── DESIGN TOKENS ── */
:root {
    --bg:           #09090B;
    --bg-1:         #101014;
    --bg-2:         #18181C;
    --bg-3:         #1E1E24;
    --bg-hover:     #242429;
    --border:       rgba(255,255,255,0.06);
    --border-med:   rgba(255,255,255,0.10);
    --border-hi:    rgba(255,255,255,0.16);

    --text-1:       #F4F4F5;
    --text-2:       #A1A1AA;
    --text-3:       #52525B;

    --copper:       #C8622A;
    --copper-lo:    rgba(200,98,42,0.08);
    --copper-mid:   rgba(200,98,42,0.16);
    --copper-hi:    rgba(200,98,42,0.28);
    --copper-glow:  rgba(200,98,42,0.40);

    --green:        #22C55E;
    --green-lo:     rgba(34,197,94,0.08);
    --green-mid:    rgba(34,197,94,0.16);
    --red:          #EF4444;
    --red-lo:       rgba(239,68,68,0.08);
    --red-mid:      rgba(239,68,68,0.16);
    --amber:        #F59E0B;
    --amber-lo:     rgba(245,158,11,0.08);
    --amber-mid:    rgba(245,158,11,0.16);
    --blue:         #3B82F6;
    --blue-lo:      rgba(59,130,246,0.08);
    --blue-mid:     rgba(59,130,246,0.16);

    --radius-xs:    4px;
    --radius-sm:    6px;
    --radius-md:    10px;
    --radius-lg:    14px;
    --radius-xl:    18px;

    --shadow-card:  0 1px 0 rgba(255,255,255,0.04), 0 4px 24px rgba(0,0,0,0.4);
    --shadow-glow:  0 0 40px rgba(200,98,42,0.12);
}

/* ── BASE ── */
html, body, [class*="css"] {
    font-family: 'Geist', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text-1) !important;
    -webkit-font-smoothing: antialiased;
}
.stApp { background: var(--bg) !important; }
*, *::before, *::after { box-sizing: border-box; }
.block-container { padding-top: 0 !important; max-width: 100% !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--bg-3); border-radius: 2px; }

/* ── HERO ── */
.apex-hero {
    background: var(--bg-1);
    border-bottom: 1px solid var(--border);
    padding: 2.8rem 3rem 2.4rem;
    margin: 0 -1rem 2.5rem -1rem;
    position: relative;
    overflow: hidden;
}
.apex-hero::after {
    content: '';
    position: absolute;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.025'/%3E%3C/svg%3E");
    opacity: 0.4;
    pointer-events: none;
}
.apex-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 380px; height: 380px;
    background: radial-gradient(circle, rgba(200,98,42,0.14) 0%, transparent 70%);
    pointer-events: none;
}
.apex-hero-inner {
    position: relative;
    z-index: 2;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 2rem;
}
.apex-eyebrow {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.2rem;
}
.apex-eyebrow-line {
    width: 24px;
    height: 1px;
    background: var(--copper);
    opacity: 0.7;
}
.apex-eyebrow-text {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--copper);
    font-weight: 400;
}
.apex-logo {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: var(--text-1);
    line-height: 0.95;
    letter-spacing: -0.04em;
    margin: 0;
}
.apex-logo-accent { color: var(--copper); }
.apex-tagline {
    font-size: 0.78rem;
    color: var(--text-3);
    margin-top: 0.75rem;
    font-weight: 400;
    letter-spacing: 0.02em;
    font-family: 'IBM Plex Mono', monospace;
}
.apex-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--green-lo);
    border: 1px solid rgba(34,197,94,0.2);
    color: var(--green);
    font-size: 0.65rem;
    padding: 0.3rem 0.9rem;
    border-radius: 100px;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.4rem;
}
.apex-status-dot {
    width: 5px; height: 5px;
    background: var(--green);
    border-radius: 50%;
    box-shadow: 0 0 6px var(--green);
    animation: blink 3s ease-in-out infinite;
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── SECTION HEADERS ── */
.section-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    font-weight: 400;
    color: var(--text-3);
    text-transform: uppercase;
    letter-spacing: 0.22em;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.section-eyebrow::before {
    content: '';
    display: inline-block;
    width: 16px;
    height: 1px;
    background: var(--copper);
    opacity: 0.6;
    flex-shrink: 0;
}
.section-heading {
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-1);
    margin: 0 0 1.5rem 0;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

/* ── DIVIDER ── */
.apex-divider {
    height: 1px;
    background: var(--border);
    margin: 3rem 0;
    position: relative;
}
.apex-divider::after {
    content: '';
    position: absolute;
    left: 0; top: 0;
    width: 60px; height: 1px;
    background: var(--copper);
    opacity: 0.4;
}

/* ── METRIC CARDS ── */
.metric-card {
    backdrop-filter: blur(10px);
    background: var(--bg-1);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: var(--radius-lg);
    padding: 1.6rem 1.8rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s ease, box-shadow 0.3s ease;
    box-shadow: var(--shadow-card);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, var(--copper) 0%, transparent 60%);
    opacity: 0.5;
}
.metric-card:hover {
    border-color: var(--border-med);
    box-shadow: var(--shadow-card), var(--shadow-glow);
}
.metric-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-3);
    margin-bottom: 1.2rem;
    display: block;
}
.metric-value {
    font-family: 'Geist', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-1);
    line-height: 1.05;
    margin-bottom: 0.25rem;
    letter-spacing: -0.03em;

    font-variant-numeric: tabular-nums;
    font-feature-settings: "tnum";
}
 /* ── KPI PERCENT FIX ── */
.metric-percent {
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.04em;
    font-variant-numeric: tabular-nums;
}          
.metric-label {
    font-size: 0.72rem;
    color: var(--text-2);
    letter-spacing: 0.02em;
    font-weight: 400;
}
.metric-delta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    margin-top: 0.8rem;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.2rem 0.7rem;
    border-radius: var(--radius-xs);
    font-weight: 400;
    letter-spacing: 0.05em;
}
.metric-delta.good  { background: var(--green-lo);  color: var(--green);  border: 1px solid rgba(34,197,94,0.15); }
.metric-delta.bad   { background: var(--red-lo);    color: var(--red);    border: 1px solid rgba(239,68,68,0.15); }
.metric-delta.warn  { background: var(--amber-lo);  color: var(--amber);  border: 1px solid rgba(245,158,11,0.15); }

/* ── CHART CARDS ── */
.chart-card {
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    box-shadow: var(--shadow-card);
    margin-bottom: 1rem;
}

/* ── AUDIT / ALERT CARDS ── */
.audit-card {
            
    backdrop-filter: blur(8px);       
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1rem 1.4rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    transition: border-color 0.2s;
    font-size: 0.83rem;
    color: var(--text-2);
    line-height: 1.65;
}
.audit-card:hover { border-color: var(--border-med); }
.audit-icon {
    font-size: 0.95rem;
    flex-shrink: 0;
    margin-top: 0.15rem;
    font-family: 'IBM Plex Mono', monospace;
    color: var(--text-3);
    font-style: normal;
}
.audit-card.warning {
    border-left: 2px solid var(--amber);
    background: var(--amber-lo);
}
.audit-card.danger {
    border-left: 2px solid var(--red);
    background: var(--red-lo);
}
.audit-card.success {
    border-left: 2px solid var(--green);
    background: var(--green-lo);
}
.audit-card.info {
    border-left: 2px solid var(--blue);
    background: var(--blue-lo);
}
.audit-text strong { color: var(--text-1); font-weight: 600; }

/* ── PATTERN PILLS ── */
.pills-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 1.4rem; }
.pattern-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--bg-2);
    border: 1px solid var(--border-med);
    border-radius: var(--radius-xs);
    padding: 0.35rem 0.9rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-2);
    font-weight: 400;
    letter-spacing: 0.05em;
    white-space: nowrap;
}
.pattern-pill.good  { border-color: rgba(34,197,94,0.25);   color: var(--green);  background: var(--green-lo); }
.pattern-pill.bad   { border-color: rgba(239,68,68,0.25);   color: var(--red);    background: var(--red-lo); }
.pattern-pill.info  { border-color: rgba(245,158,11,0.25);  color: var(--amber);  background: var(--amber-lo); }

/* ── STATUS BADGE ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--green-lo);
    border: 1px solid rgba(34,197,94,0.2);
    color: var(--green);
    font-size: 0.62rem;
    padding: 0.35rem 0.9rem;
    border-radius: var(--radius-xs);
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.status-dot {
    width: 5px; height: 5px;
    background: var(--green);
    border-radius: 50%;
    box-shadow: 0 0 6px var(--green);
    animation: blink 3s ease-in-out infinite;
}

/* ── CHAT ── */
.chat-wrapper {
    background: #FFFFFF;
    border: 1px solid var(--border-med);
    border-radius: var(--radius-lg);
    overflow: hidden;
    box-shadow: var(--shadow-card);
    margin-bottom: 1rem;
}
.chat-header {
    background: #F7F7F7;
    border-bottom: 1px solid var(--border-med);
    padding: 0.9rem 1.4rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.chat-header-indicator {
    width: 7px; height: 7px;
    background: var(--green);
    border-radius: 50%;
    box-shadow: 0 0 6px var(--green);
    flex-shrink: 0;
    animation: blink 3s ease-in-out infinite;
}
.chat-header-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: #1A1916;
    letter-spacing: -0.01em;
}
.chat-header-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    color: #9B9890;
    margin-top: 0.15rem;
    letter-spacing: 0.02em;
}
.msg-user {
    background: #F0F0EE;
    border: 1px solid #E2E0DB;
    border-radius: var(--radius-sm);
    padding: 0.7rem 1rem;
    margin-left: 3rem;
    font-size: 0.82rem;
    color: #1A1916;
    line-height: 1.6;
}
.msg-assistant {
    background: #FFFFFF;
    border: 1px solid #E2E0DB;
    border-left: 2px solid var(--copper);
    border-radius: var(--radius-sm);
    padding: 0.7rem 1rem;
    margin-right: 3rem;
    font-size: 0.82rem;
    color: #3A3A3A;
    line-height: 1.7;
}
.msg-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.56rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    margin-bottom: 0.3rem;
    font-weight: 400;
}
[data-testid="stChatInput"] textarea {
    background: #FFFFFF !important;
    border: 1px solid #D4D1CB !important;
    color: #1A1916 !important;
    border-radius: var(--radius-md) !important;
    font-family: 'Geist', sans-serif !important;
    font-size: 0.85rem !important;
}
.msg-label.user { color: #9B9890; }
.msg-label.apex { color: var(--copper); }    


/* ── LANDING ── */
.landing-empty {
    text-align: center;
    padding: 6rem 2rem 5rem;
    max-width: 580px;
    margin: 0 auto;
}
.landing-mark {
    width: 52px; height: 52px;
    background: var(--copper-lo);
    border: 1px solid var(--copper-mid);
    border-radius: var(--radius-md);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 2rem;
}
.landing-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--text-1);
    margin-bottom: 1rem;
    letter-spacing: -0.04em;
    line-height: 1.1;
}
.landing-sub {
    color: var(--text-2);
    font-size: 0.85rem;
    line-height: 1.85;
    margin: 0 auto 2rem;
    font-weight: 400;
}
.landing-features {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 2rem;
}
.landing-divider {
    width: 40px;
    height: 1px;
    background: var(--copper);
    opacity: 0.5;
    margin: 1.5rem auto;
}

/* ── ML SECTION ── */
.ml-section {
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 2rem 2.2rem;
    position: relative;
    overflow: hidden;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-card);
}
.ml-section::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, var(--blue) 0%, transparent 50%);
    opacity: 0.4;
}
.ml-section::after {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(59,130,246,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.ml-header {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    margin-bottom: 2rem;
    position: relative;
    z-index: 1;
}
.ml-icon-wrap {
    width: 42px; height: 42px;
    background: var(--blue-lo);
    border: 1px solid var(--blue-mid);
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.ml-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-1);
    letter-spacing: -0.01em;
}
.ml-subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: var(--text-3);
    margin-top: 0.3rem;
    letter-spacing: 0.05em;
}
.ml-badge {
    margin-left: auto;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--green-lo);
    border: 1px solid rgba(34,197,94,0.2);
    color: var(--green);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    padding: 0.25rem 0.8rem;
    border-radius: var(--radius-xs);
    font-weight: 400;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.ml-stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.8rem;
    position: relative;
    z-index: 1;
}
.ml-stat-card {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.2rem 1.4rem;
    transition: border-color 0.2s;
}
.ml-stat-card:hover { border-color: var(--border-hi); }
.ml-stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--text-1);
    line-height: 1;
    margin-bottom: 0.3rem;
    letter-spacing: -0.03em;
}
.ml-stat-val.accent { color: var(--blue); }
.ml-stat-val.good   { color: var(--green); }
.ml-stat-val.warn   { color: var(--amber); }
.ml-stat-lbl {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: var(--text-3);
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
.ml-accuracy-bar-wrap { margin-top: 1rem; }
.ml-accuracy-label {
    display: flex;
    justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: var(--text-3);
    margin-bottom: 0.4rem;
    letter-spacing: 0.04em;
}
.ml-accuracy-bar {
    height: 3px;
    background: var(--bg-3);
    border-radius: 100px;
    overflow: hidden;
}
.ml-accuracy-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--blue), #60A5FA);
    border-radius: 100px;
    transition: width 1.2s cubic-bezier(0.4,0,0.2,1);
}
.ml-predict-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.6rem;
    position: relative;
    z-index: 1;
}
.ml-predict-card {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1rem;
    text-align: center;
    transition: border-color 0.2s, background 0.2s;
}
.ml-predict-card:hover {
    border-color: rgba(59,130,246,0.3);
    background: var(--bg-3);
}
.ml-predict-hour {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: var(--text-3);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.ml-predict-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--blue);
    line-height: 1;
    letter-spacing: -0.03em;
}
.ml-predict-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    color: var(--text-3);
    margin-top: 0.3rem;
    letter-spacing: 0.06em;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: var(--bg-1) !important;
    border: 1px dashed var(--border-med) !important;
    border-radius: var(--radius-lg) !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--copper) !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span {
    color: var(--text-2) !important;
}

/* ── PLOTLY ── */
.js-plotly-plot { border-radius: var(--radius-md); overflow: hidden; }
            

/* ── CHATGPT STYLE INPUT ───────────────────────── */

[data-testid="stBottom"] {
    background: transparent !important;
    height: auto !important;
}

[data-testid="stBottom"] > div {
    background: transparent !important;
    padding: 0.8rem 0 1rem 0 !important;
}

/* CONTENEDOR GENERAL */
[data-testid="stChatInput"] {
    max-width: 900px !important;
    margin: 0 auto !important;
    padding: 0 !important;
    background: transparent !important;
}
            

            
/* CAJA PRINCIPAL */
[data-testid="stChatInput"] > div {
    background: rgba(255,255,255,0.98) !important;
    border: 1px solid #E4E4E7 !important;
    border-radius: 18px !important;
    min-height: 56px !important;

    box-shadow:
        0 1px 2px rgba(0,0,0,0.04),
        0 8px 30px rgba(0,0,0,0.06) !important;

    transition: all 0.2s ease !important;
}

/* EFECTO FOCUS */
[data-testid="stChatInput"] > div:focus-within {
    border: 1px solid rgba(200,98,42,0.45) !important;

    box-shadow:
        0 0 0 4px rgba(200,98,42,0.08),
        0 10px 35px rgba(0,0,0,0.08) !important;
}

/* TEXTAREA */
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;

    color: #18181B !important;

    font-family: 'Geist', sans-serif !important;
    font-size: 0.92rem !important;
    line-height: 1.5 !important;

    min-height: 24px !important;
    max-height: 140px !important;

    padding: 1rem 0.2rem 1rem 0.2rem !important;

    resize: none !important;
}

/* PLACEHOLDER */
[data-testid="stChatInput"] textarea::placeholder {
    color: #A1A1AA !important;
}

/* BOTÓN SEND */
[data-testid="stChatInput"] button {
    background: #C8622A !important;
    border: none !important;

    width: 38px !important;
    height: 38px !important;

    border-radius: 12px !important;

    transition: all 0.2s ease !important;

    margin-right: 0.5rem !important;
    margin-bottom: 0.45rem !important;
}

/* HOVER BOTÓN */
[data-testid="stChatInput"] button:hover {
    background: #B45322 !important;
    transform: scale(1.04);
}

/* ICONO */
[data-testid="stChatInput"] button svg {
    color: white !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: var(--bg-1) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-2) !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] { border-radius: var(--radius-md) !important; overflow: hidden !important; }

/* ── SPINNER ── */
[data-testid="stSpinner"] { color: var(--copper) !important; }

/* ── SLIDER ── */
[data-testid="stSlider"] > div > div > div > div {
    background-color: var(--copper) !important;
}

/* ── SELECT / INPUT ── */
[data-testid="stSelectbox"] > div,
[data-testid="stNumberInput"] input {
    background: var(--bg-2) !important;
    border-color: var(--border-med) !important;
    color: var(--text-1) !important;
}

/* ── HIDE BRANDING ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── COLUMNS gap ── */
[data-testid="column"] { padding: 0 0.4rem !important; }
</style>

            
            
""", unsafe_allow_html=True)


# ─── SESSION STATE ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False


# ─── PLOTLY THEME — OSCURO PROFESIONAL ─────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#111113",
    plot_bgcolor="#111113",
    font=dict(color="#52525B", family="IBM Plex Mono", size=11),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        linecolor="rgba(255,255,255,0.06)",
        tickfont=dict(color="#52525B", size=10, family="IBM Plex Mono"),
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        linecolor="rgba(255,255,255,0.06)",
        tickfont=dict(color="#52525B", size=10, family="IBM Plex Mono"),
        zeroline=False,
    ),
    margin=dict(l=12, r=12, t=40, b=12),
    hoverlabel=dict(
        bgcolor="#18181C",
        bordercolor="rgba(255,255,255,0.1)",
        font=dict(color="#F4F4F5", family="IBM Plex Mono", size=11),
    ),
    title_font=dict(
    color="#E4E4E7",
    size=15,
    family="Geist",

    ), 

)

# ── Paleta principal ────────────────────────────────────────────────────────────
C_PRIMARY   = "#C8622A"
C_SECONDARY = "#E8860A"
C_BLUE      = "#3B82F6"
COLORS      = ["#C8622A", "#E8860A", "#22C55E", "#3B82F6", "#A78BFA", "#06B6D4", "#F59E0B"]


# ─── HERO HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="apex-hero">
    <div class="apex-hero-inner">
        <div>
            <div class="apex-status-pill">
                <span class="apex-status-dot"></span>
                Sistema activo
            </div>
            <div class="apex-eyebrow">
                <span class="apex-eyebrow-line"></span>
                <span class="apex-eyebrow-text">Automotive Intelligence Platform</span>
            </div>
            <h1 class="apex-logo">APEX<span class="apex-logo-accent">COIA</span></h1>
            <div class="apex-tagline">auditoría_inteligente.v2 · claude_ai_engine · concesionarios</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── FILE UPLOAD ────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Carga el archivo de datos del concesionario",
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

        # ── FORMATOS COLOMBIA ────────────────────────────────────────────────
        def format_cop(value):
            try:
                value = float(value)

                if value >= 1_000_000_000:
                    return f"${value/1_000_000_000:.1f}B COP"

                elif value >= 1_000_000:
                    return f"${value/1_000_000:.1f}M COP"

                elif value >= 1_000:
                    return f"${value/1_000:.0f}K COP"

                return f"${value:,.0f} COP"

            except:
                return "$0 COP"
            
                # ── KPI CARDS ───────────────────────────────────────────────────────
        st.markdown('<div class="section-eyebrow">Resumen operativo</div>', unsafe_allow_html=True)

        conv_pct   = metrics['conversion']
        conv_cls   = "good" if conv_pct >= 0.20 else "bad"
        conv_label = "✓ Buena conversión" if conv_pct >= 0.20 else "↓ Conversión baja"

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-accent"></div>
                <span class="metric-icon">◉</span>
                <div class="metric-value">{int(metrics['total_ventas']):,}</div>
                <div class="metric-label">Ventas totales</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-accent"></div>
                <span class="metric-icon">◉</span>
                <div class="metric-value">{int(metrics['total_visitas']):,}</div>
                <div class="metric-label">Visitas totales</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-accent"></div>
                <span class="metric-icon">◉</span>
                <div class="metric-value metric-percent">{conv_pct:.1%}</div>
                <div class="metric-label">Tasa de conversión</div>
                <span class="metric-delta {conv_cls}">{conv_label}</span>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-accent"></div>
                <span class="metric-icon">◉</span>
                <div class="metric-value">{format_cop(metrics['ticket_promedio'])}</div>
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
        st.markdown(
            '<div class="section-eyebrow">Machine Learning · Predicción de ventas</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="section-heading">🧠 Motor predictivo APEX</div>',
            unsafe_allow_html=True
        )

        if not has_ml_cols:
            st.markdown("""
            <div class="audit-card warning">
                <span class="audit-icon">⚠️</span>
                <span class="audit-text">
                    El dataset no contiene las columnas
                    <strong>hora</strong>,
                    <strong>ventas</strong> y
                    <strong>visitas</strong>
                    necesarias para entrenar el modelo predictivo.
                </span>
            </div>
            """, unsafe_allow_html=True)

        elif ml_result and "error" in ml_result:
            st.markdown(f"""
            <div class="audit-card danger">
                <span class="audit-icon">❌</span>
                <span class="audit-text">
                    Error al entrenar el modelo:
                    <strong>{ml_result['error']}</strong>
                </span>
            </div>
            """, unsafe_allow_html=True)

        else:
            ml_mae = ml_result.get("mae", 0)
            ml_predictions = ml_result.get("predictions", [])

            try:
                ml_mae = float(ml_mae)
            except Exception:
                ml_mae = 0.0

            base_sales = max(float(metrics.get("total_ventas", 1)), 1.0)

            if ml_mae is not None:
                accuracy_pct = max(0.0, min(100.0, 100.0 - ((ml_mae / base_sales) * 100)))
            else:
                accuracy_pct = 0.0

            acc_cls = "good" if accuracy_pct >= 80 else ("warn" if accuracy_pct >= 60 else "accent")
            mae_cls = "good" if ml_mae <= 2 else ("warn" if ml_mae <= 5 else "accent")
            mae_bar = int(max(10, min(100, 100 - (ml_mae * 10))))
            acc_bar = int(max(0, min(100, accuracy_pct)))

            preds_to_show = ml_predictions[:8]

            pred_cards_html = ""
            for p in preds_to_show:
                hora = int(p.get("hora", 0))
                try:
                    valor = float(p.get("ventas_pred", p.get("pred", 0)))
                except Exception:
                    valor = 0.0
                pred_cards_html += (
                    f'<div class="ml-predict-card">'
                    f'<div class="ml-predict-hour">{hora:02d}:00 h</div>'
                    f'<div class="ml-predict-val">{valor:.1f}</div>'
                    f'<div class="ml-predict-label">ventas est.</div>'
                    f'</div>'
                )

            ml_html = f"""
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
                <div class="ml-accuracy-label">
                    <span>Precisión del error</span><span>{mae_bar}%</span>
                </div>
                <div class="ml-accuracy-bar">
                    <div class="ml-accuracy-fill" style="width:{mae_bar}%;"></div>
                </div>
            </div>
        </div>
        <div class="ml-stat-card">
            <div class="ml-stat-val {acc_cls}">{accuracy_pct:.1f}%</div>
            <div class="ml-stat-lbl">Precisión estimada</div>
            <div class="ml-accuracy-bar-wrap">
                <div class="ml-accuracy-label">
                    <span>Precisión</span><span>{acc_bar}%</span>
                </div>
                <div class="ml-accuracy-bar">
                    <div class="ml-accuracy-fill" style="width:{acc_bar}%;"></div>
                </div>
            </div>
        </div>
        <div class="ml-stat-card">
            <div class="ml-stat-val accent">{len(ml_predictions)}</div>
            <div class="ml-stat-lbl">Predicciones generadas</div>
            <div class="ml-accuracy-bar-wrap">
                <div class="ml-accuracy-label">
                    <span>Cobertura</span><span>100%</span>
                </div>
                <div class="ml-accuracy-bar">
                    <div class="ml-accuracy-fill" style="width:100%;"></div>
                </div>
            </div>
        </div>
    </div>
    <div class="ml-predict-row">
        {pred_cards_html}
    </div>
</div>
"""
            st.markdown(ml_html, unsafe_allow_html=True)

            # Gráfico Real vs Predicho
            if ml_predictions:
                import pandas as _pd
                pred_df = _pd.DataFrame(ml_predictions)
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
            insights.append(("info", "◉",
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
        <div style="display:inline-flex;align-items:center;gap:0.5rem;
            background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.2);
            color:#22C55E;font-size:0.6rem;padding:0.3rem 0.8rem;
            border-radius:4px;font-family:'IBM Plex Mono',monospace;
            letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.6rem;">
            <span style="width:5px;height:5px;background:#22C55E;border-radius:50%;
                box-shadow:0 0 6px #22C55E;display:inline-block;"></span>
            APEX AI · Privado · Sin consumo de créditos
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#FAFAFA;border:1px solid #E2E0DB;
            border-radius:8px 8px 0 0;padding:0.7rem 1.2rem;
            border-bottom:1px solid #E2E0DB;">
            <div style="font-family:'Syne',sans-serif;font-size:0.82rem;
                font-weight:700;color:#1A1916;">APEX Intelligence Engine</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                color:#9B9890;margin-top:0.1rem;">
                Pregúntame sobre ventas, conversión, vendedores, horarios o estrategia
            </div>
        </div>
        """, unsafe_allow_html=True)

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="margin:0.4rem 0;">
                    <div class="msg-label user">Tú</div>
                    <div class="msg-user">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="margin:0.4rem 0;">
                    <div class="msg-label apex">APEX</div>
                    <div class="msg-assistant">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)

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