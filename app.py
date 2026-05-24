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
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* BASE */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

.stApp {
    background: #0a0a0f;
}

/* HEADER HERO */
.apex-hero {
    background: linear-gradient(135deg, #0d0d1a 0%, #0a0a0f 50%, #0d1117 100%);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 2.5rem 2rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    position: relative;
    overflow: hidden;
}

.apex-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(255, 87, 34, 0.12) 0%, transparent 70%);
    border-radius: 50%;
}

.apex-hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 20%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(255, 152, 0, 0.07) 0%, transparent 70%);
    border-radius: 50%;
}

.apex-logo {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #ffffff;
    margin: 0;
    line-height: 1;
}

.apex-logo span {
    color: #FF5722;
}

.apex-tagline {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.4);
    margin-top: 0.4rem;
    font-weight: 300;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.apex-badge {
    display: inline-block;
    background: rgba(255, 87, 34, 0.15);
    border: 1px solid rgba(255, 87, 34, 0.3);
    color: #FF7043;
    font-size: 0.7rem;
    padding: 0.2rem 0.7rem;
    border-radius: 100px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* METRIC CARDS */
.metric-card {
    background: #111118;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s ease;
}

.metric-card:hover {
    border-color: rgba(255, 87, 34, 0.3);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #FF5722, #FF9800);
    opacity: 0;
    transition: opacity 0.2s;
}

.metric-card:hover::before {
    opacity: 1;
}

.metric-icon {
    font-size: 1.4rem;
    margin-bottom: 0.6rem;
    display: block;
}

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1;
    margin-bottom: 0.3rem;
}

.metric-label {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.4);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 500;
}

.metric-delta-good {
    font-size: 0.75rem;
    color: #4CAF50;
    margin-top: 0.4rem;
}

.metric-delta-bad {
    font-size: 0.75rem;
    color: #FF5722;
    margin-top: 0.4rem;
}

/* SECTION TITLES */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 1rem 0;
    letter-spacing: -0.01em;
}

.section-label {
    font-size: 0.7rem;
    color: rgba(255,255,255,0.3);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.3rem;
}

/* AUDIT CARDS */
.audit-card {
    background: #111118;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
}

.audit-icon {
    font-size: 1.1rem;
    flex-shrink: 0;
    margin-top: 0.1rem;
}

.audit-text {
    font-size: 0.88rem;
    color: rgba(255,255,255,0.75);
    line-height: 1.5;
}

.audit-card.warning {
    border-left: 3px solid #FF9800;
    background: rgba(255, 152, 0, 0.05);
}

.audit-card.danger {
    border-left: 3px solid #FF5722;
    background: rgba(255, 87, 34, 0.05);
}

.audit-card.success {
    border-left: 3px solid #4CAF50;
    background: rgba(76, 175, 80, 0.05);
}

/* CHAT */
.chat-container {
    background: #111118;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    overflow: hidden;
}

.chat-header {
    background: linear-gradient(90deg, rgba(255,87,34,0.15), rgba(255,152,0,0.08));
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 1rem 1.4rem;
    display: flex;
    align-items: center;
    gap: 0.7rem;
}

.chat-dot {
    width: 8px; height: 8px;
    background: #4CAF50;
    border-radius: 50%;
    box-shadow: 0 0 8px #4CAF50;
}

.chat-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #fff;
}

.chat-subtitle {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.4);
}

.msg-user {
    background: rgba(255, 87, 34, 0.12);
    border: 1px solid rgba(255, 87, 34, 0.2);
    border-radius: 12px 12px 4px 12px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0 0.4rem 2rem;
    font-size: 0.88rem;
    color: rgba(255,255,255,0.9);
}

.msg-assistant {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px 12px 12px 4px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 2rem 0.4rem 0;
    font-size: 0.88rem;
    color: rgba(255,255,255,0.85);
    line-height: 1.6;
}

.msg-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.2rem;
    font-weight: 600;
}

.msg-label.user { color: #FF7043; }
.msg-label.apex { color: #FF9800; }

/* PATTERN PILLS */
.pattern-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 100px;
    padding: 0.35rem 0.9rem;
    font-size: 0.8rem;
    color: rgba(255,255,255,0.7);
    margin: 0.2rem;
}

.pattern-pill.good { border-color: rgba(76,175,80,0.4); color: #81C784; }
.pattern-pill.bad { border-color: rgba(255,87,34,0.4); color: #FF7043; }
.pattern-pill.info { border-color: rgba(255,152,0,0.4); color: #FFB74D; }

/* DIVIDER */
.apex-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    margin: 2rem 0;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: #111118 !important;
    border: 2px dashed rgba(255,87,34,0.3) !important;
    border-radius: 12px !important;
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* HIDE STREAMLIT BRANDING */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* SCROLLBAR */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }

/* PLOTLY CHARTS BG */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }

/* CHAT INPUT */
[data-testid="stChatInput"] textarea {
    background: #1a1a25 !important;
    border-color: rgba(255,87,34,0.3) !important;
    color: #e8e8f0 !important;
    border-radius: 12px !important;
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
    font=dict(color="rgba(255,255,255,0.6)", family="DM Sans"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    margin=dict(l=10, r=10, t=40, b=10),
)
ORANGE = "#FF5722"
ORANGE2 = "#FF9800"
COLORS = [ORANGE, ORANGE2, "#E91E63", "#9C27B0", "#3F51B5", "#00BCD4", "#4CAF50"]

# ─── HERO HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="apex-hero">
    <div class="apex-badge">● Live Dashboard</div>
    <div class="apex-logo">APEX<span>CO</span> AI</div>
    <div class="apex-tagline">Auditoría inteligente para concesionarios · Powered by Claude AI</div>
</div>
""", unsafe_allow_html=True)

# ─── UPLOAD ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Carga tu archivo de datos del concesionario",
    type=["csv"],
    help="Formato CSV con columnas: fecha, hora, concesionario, vendedor, marca, modelo, precio_lista, descuento, precio_final, visita, venta, canal, estado_lead"
)

if uploaded_file is not None:
    try:
        df_raw = load_csv(uploaded_file)

        ok_cols, missing_cols = validate_required_columns(df_raw)
        ok_fmt, format_errors = validate_basic_format(df_raw)
        ok_num, num_errors = validate_numeric_columns(df_raw)

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

        # ── Compute everything ──────────────────────────────────────────────────
        metrics        = compute_metrics(df)
        hourly_df      = sales_by_hour(df)
        daily_df       = sales_by_day(df)
        sellers_df     = top_sellers(df)
        brands_df      = top_brands(df)
        low_conv_df    = detect_low_conversion(df)
        patterns       = detect_patterns(df)
        recommendations = generate_recommendations(metrics, low_conv_df, sellers_df, brands_df)

        st.session_state.data_loaded = True

        # ── KPI CARDS ───────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Resumen operativo</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)

        conv_pct = metrics['conversion']
        conv_color = "metric-delta-good" if conv_pct >= 0.20 else "metric-delta-bad"
        conv_label = "✓ Buena conversión" if conv_pct >= 0.20 else "↓ Conversión baja"

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-icon">🏷️</span>
                <div class="metric-value">{metrics['total_ventas']:.0f}</div>
                <div class="metric-label">Ventas totales</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-icon">👥</span>
                <div class="metric-value">{metrics['total_visitas']:.0f}</div>
                <div class="metric-label">Visitas totales</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-icon">📈</span>
                <div class="metric-value">{conv_pct:.1%}</div>
                <div class="metric-label">Conversión</div>
                <div class="{conv_color}">{conv_label}</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-icon">💰</span>
                <div class="metric-value">${metrics['ticket_promedio']:,.0f}</div>
                <div class="metric-label">Ticket promedio</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="apex-divider"></div>', unsafe_allow_html=True)

        # ── CHARTS ROW 1 ────────────────────────────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-label">Análisis temporal</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Ventas por hora</div>', unsafe_allow_html=True)
            fig1 = px.bar(
                hourly_df, x="hora_int", y="ventas",
                color_discrete_sequence=[ORANGE]
            )
            fig1.update_layout(**PLOTLY_LAYOUT, title=None)
            fig1.update_traces(marker_line_width=0)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown('<div class="section-label">Eficiencia</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Conversión por hora</div>', unsafe_allow_html=True)
            fig2 = px.area(
                hourly_df, x="hora_int", y="conversion",
                color_discrete_sequence=[ORANGE2]
            )
            fig2.update_layout(**PLOTLY_LAYOUT, title=None)
            fig2.update_traces(line_width=2, fillcolor="rgba(255,152,0,0.1)")
            st.plotly_chart(fig2, use_container_width=True)

        # ── CHARTS ROW 2 ────────────────────────────────────────────────────────
        col3, col4 = st.columns(2)

        with col3:
            st.markdown('<div class="section-label">Tendencia</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Ventas por día</div>', unsafe_allow_html=True)
            fig3 = px.line(
                daily_df, x="fecha_str", y="ventas",
                markers=True, color_discrete_sequence=[ORANGE]
            )
            fig3.update_layout(**PLOTLY_LAYOUT, title=None)
            fig3.update_traces(line_width=2, marker_size=6)
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            st.markdown('<div class="section-label">Portafolio</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Top marcas</div>', unsafe_allow_html=True)
            fig4 = px.bar(
                brands_df.head(8), x="ventas", y="marca",
                orientation='h', color_discrete_sequence=[ORANGE]
            )
            fig4.update_layout(**PLOTLY_LAYOUT, title=None)
            fig4.update_traces(marker_line_width=0)
            st.plotly_chart(fig4, use_container_width=True)

        # ── CHARTS ROW 3 — Sellers + Donut ─────────────────────────────────────
        col5, col6 = st.columns(2)

        with col5:
            st.markdown('<div class="section-label">Equipo comercial</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Ranking de vendedores</div>', unsafe_allow_html=True)
            fig5 = px.bar(
                sellers_df.head(8), x="ventas", y="vendedor",
                orientation='h', color_discrete_sequence=COLORS
            )
            fig5.update_layout(**PLOTLY_LAYOUT, title=None)
            fig5.update_traces(marker_line_width=0)
            st.plotly_chart(fig5, use_container_width=True)

        with col6:
            st.markdown('<div class="section-label">Distribución</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Ventas por marca (%)</div>', unsafe_allow_html=True)
            fig6 = px.pie(
                brands_df.head(6), values="ventas", names="marca",
                hole=0.6, color_discrete_sequence=COLORS
            )
            fig6.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="rgba(255,255,255,0.6)", family="DM Sans"),
                margin=dict(l=10, r=10, t=20, b=10),
                showlegend=True,
                legend=dict(font=dict(color="rgba(255,255,255,0.6)"))
            )
            fig6.update_traces(textfont_color="white")
            st.plotly_chart(fig6, use_container_width=True)

        st.markdown('<div class="apex-divider"></div>', unsafe_allow_html=True)

        # ── PATRONES ────────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Diagnóstico automático</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Patrones detectados</div>', unsafe_allow_html=True)

        pills_html = ""
        if patterns['best_hour'] is not None:
            pills_html += f'<span class="pattern-pill good">✓ Mejor hora: {patterns["best_hour"]:02d}:00</span>'
        if patterns['worst_hour'] is not None:
            pills_html += f'<span class="pattern-pill bad">↓ Peor hora: {patterns["worst_hour"]:02d}:00</span>'
        for h in patterns['high_traffic_hours']:
            pills_html += f'<span class="pattern-pill info">⚡ Alto tráfico: {h:02d}:00</span>'

        st.markdown(f'<div style="margin-bottom:1rem">{pills_html}</div>', unsafe_allow_html=True)

        # ── BAJA CONVERSIÓN ─────────────────────────────────────────────────────
        if len(low_conv_df) > 0:
            st.markdown(f"""
            <div class="audit-card danger">
                <span class="audit-icon">🔴</span>
                <span class="audit-text">Se detectaron <strong>{len(low_conv_df)} hora(s)</strong> con conversión inferior al 15%. 
                Revisar asignación de personal en esos turnos.</span>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("Ver detalle de horas con baja conversión"):
                st.dataframe(low_conv_df, use_container_width=True)
        else:
            st.markdown("""
            <div class="audit-card success">
                <span class="audit-icon">✅</span>
                <span class="audit-text">No se detectaron horas con conversión baja. Operación estable.</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="apex-divider"></div>', unsafe_allow_html=True)

    # ── RECOMENDACIONES ─────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Motor de auditoría</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Recomendaciones accionables</div>', unsafe_allow_html=True)

        icons = ["🎯", "📊", "👥", "⚡", "💡", "🔧", "📈"]
        levels = ["warning", "danger", "warning", "success", "warning", "danger", "success"]
        for i, rec in enumerate(recommendations):
            icon = icons[i % len(icons)]
            level = levels[i % len(levels)]
            st.markdown(f"""
            <div class="audit-card {level}">
                <span class="audit-icon">{icon}</span>
                <span class="audit-text">{rec}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="apex-divider"></div>', unsafe_allow_html=True)

        # ── SIMULADOR ────────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Inteligencia predictiva</div>', unsafe_allow_html=True)
        render_simulator(metrics=metrics, patterns=patterns, sellers_df=sellers_df)

        st.markdown('<div class="apex-divider"></div>', unsafe_allow_html=True)

        # ── DATOS RAW ───────────────────────────────────────────────────────────
        with st.expander("📋 Vista previa de datos cargados"):
            st.dataframe(df, use_container_width=True)

        st.markdown('<div class="apex-divider"></div>', unsafe_allow_html=True)
        # ── CHAT ────────────────────────────────────────────────────────────────
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
    # ── LANDING VACÍO ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem;">
        <div style="font-size:3rem; margin-bottom:1rem;">🚗</div>
        <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700; color:#fff; margin-bottom:0.5rem;">
            Carga los datos de tu concesionario
        </div>
        <div style="color:rgba(255,255,255,0.4); font-size:0.9rem; max-width:420px; margin:0 auto; line-height:1.7;">
            Sube un CSV con tus registros de ventas y visitas.<br>
            APEX analizará tu operación en segundos.
        </div>
        <div style="margin-top:2rem; display:flex; justify-content:center; gap:1rem; flex-wrap:wrap;">
            <span class="pattern-pill info">📊 Métricas en tiempo real</span>
            <span class="pattern-pill good">🤖 IA conversacional</span>
            <span class="pattern-pill bad">🔍 Detección de pérdidas</span>
        </div>
    </div>
    """, unsafe_allow_html=True)