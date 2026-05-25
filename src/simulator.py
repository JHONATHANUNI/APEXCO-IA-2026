import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def _fmt_cop(value: float) -> str:
    """
    Formatea un valor en COP a escala legible.
    < 1 MM  → "$ X,XXX M"
    >= 1 MM → "$ X.XX MM"  (miles de millones)
    """
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f} MM"
    return f"${value / 1_000_000:,.0f} M"


def render_simulator(metrics: dict, patterns: dict, sellers_df=None):
    """
    Simulador de escenarios comerciales.

    FIXES aplicados:
    1. Ingreso mostrado en Miles de Millones (MM) — no en millones crudos
       que generaban cifras absurdas como $1,083,612.8 M.
    2. Gráfico comparativo separado en dos subplots para no mezclar
       escalas incomparables (ventas vs ingresos).
    3. Delta de ingresos también en MM para consistencia.
    """

    # ── CSS ────────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    .sim-header {
        background: linear-gradient(135deg, #FFF5F0, #FFFBF7);
        border: 1px solid rgba(212,82,26,0.2);
        border-radius: 16px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .sim-header::before {
        content: '⚡';
        position: absolute;
        right: 1.5rem; top: 1rem;
        font-size: 3rem;
        opacity: 0.10;
    }
    .sim-title {
        font-family: 'Instrument Serif', serif;
        font-size: 1.3rem;
        font-weight: 400;
        color: #1A1916;
        margin-bottom: 0.3rem;
    }
    .sim-subtitle {
        font-size: 0.82rem;
        color: #9B9890;
        line-height: 1.6;
    }
    .sim-badge {
        display: inline-block;
        background: #FEE9DF;
        border: 1px solid rgba(212,82,26,0.3);
        color: #D4521A;
        font-size: 0.65rem;
        padding: 0.18rem 0.65rem;
        border-radius: 100px;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }
    .result-card {
        background: #FFFFFF;
        border: 1px solid #E2E0DB;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 1px 4px rgba(26,25,22,0.06);
        transition: box-shadow 0.2s;
    }
    .result-card:hover { box-shadow: 0 4px 12px rgba(26,25,22,0.10); }
    .result-card.highlight {
        border-color: rgba(212,82,26,0.3);
        background: linear-gradient(135deg, #FFF5F0, #FFFFFF);
    }
    .result-card::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #D4521A, #E8860A);
        border-radius: 14px 14px 0 0;
    }
    .result-card.neutral::after {
        background: #E2E0DB;
    }
    .result-value {
        font-family: 'Instrument Serif', serif;
        font-size: 1.75rem;
        font-weight: 400;
        color: #1A1916;
        line-height: 1;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    .result-value.positive { color: #2A7D4F; }
    .result-value.negative { color: #C1392B; }
    .result-label {
        font-size: 0.7rem;
        color: #9B9890;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 500;
    }
    .result-delta {
        font-size: 0.76rem;
        margin-top: 0.45rem;
        font-weight: 600;
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 100px;
    }
    .result-delta.up   { color: #2A7D4F; background: #EAF5EE; }
    .result-delta.down { color: #C1392B; background: #FCECEA; }
    .result-delta.flat { color: #9B9890; background: #F0EFEC; }
    .scenario-divider {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .scenario-divider-line {
        flex: 1;
        height: 1px;
        background: #E2E0DB;
    }
    .scenario-divider-text {
        font-size: 0.68rem;
        color: #9B9890;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        white-space: nowrap;
        font-weight: 600;
    }
    .insight-box {
        background: #FFFBF0;
        border: 1px solid rgba(180,83,9,0.25);
        border-left: 4px solid #E8860A;
        border-radius: 12px;
        padding: 1rem 1.3rem;
        margin-top: 1rem;
    }
    .insight-box.danger {
        background: #FFF8F8;
        border-color: rgba(193,57,43,0.25);
        border-left-color: #C1392B;
    }
    .insight-box.neutral {
        background: #F7F7F5;
        border-color: #E2E0DB;
        border-left-color: #9B9890;
    }
    .insight-text {
        font-size: 0.875rem;
        color: #5C5A55;
        line-height: 1.65;
    }
    .insight-text strong { color: #1A1916; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

    # ── HEADER ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sim-header">
        <div class="sim-badge">🔬 Simulador de escenarios</div>
        <div class="sim-title">¿Qué pasa si cambias esto?</div>
        <div class="sim-subtitle">
            Ajusta los parámetros y ve el impacto proyectado en ventas e ingresos
            en tiempo real, basado en los datos reales de tu operación.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── DATOS BASE ──────────────────────────────────────────────────────────────
    base_ventas     = float(metrics.get("total_ventas",    50))
    base_visitas    = float(metrics.get("total_visitas",  200))
    base_conversion = float(metrics.get("conversion",    0.25))
    base_ticket     = float(metrics.get("ticket_promedio", 80_000_000))

    # Guardia: si el ticket llega en escala incorrecta, usar fallback
    if base_ticket < 1_000:
        base_ticket = 80_000_000

    # Ingreso base en COP
    base_ingreso = base_ventas * base_ticket

    # ── SLIDERS ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="scenario-divider">
        <div class="scenario-divider-line"></div>
        <div class="scenario-divider-text">Ajusta los parámetros</div>
        <div class="scenario-divider-line"></div>
    </div>
    """, unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        delta_vendedores = st.slider(
            "👥 Vendedores adicionales en hora pico",
            min_value=-2, max_value=5, value=0, step=1,
            help="Cada asesor adicional en hora pico mejora ~4% la conversión"
        )
        delta_tiempo_respuesta = st.slider(
            "⚡ Mejora en tiempo de respuesta (%)",
            min_value=0, max_value=80, value=0, step=5,
            help="Reducir el tiempo de respuesta al cliente mejora la conversión hasta +18%"
        )

    with col_s2:
        delta_marketing = st.slider(
            "📣 Incremento en inversión de marketing (%)",
            min_value=0, max_value=100, value=0, step=10,
            help="Mayor inversión en canales digitales genera más tráfico de visitas"
        )
        delta_descuento = st.slider(
            "🏷️ Ajuste en descuento promedio (%)",
            min_value=-5, max_value=10, value=0, step=1,
            help="Más descuento sube conversión pero reduce el ticket promedio"
        )

    # ── MODELO DE SIMULACIÓN ────────────────────────────────────────────────────
    impacto_visitas         = 1 + (delta_marketing / 100) * 0.6
    impacto_conv_vendedores = delta_vendedores * 0.04
    impacto_conv_respuesta  = (delta_tiempo_respuesta / 100) * 0.18
    impacto_conv_descuento  = (delta_descuento / 100) * 0.50

    nueva_conversion = float(np.clip(
        base_conversion + impacto_conv_vendedores + impacto_conv_respuesta + impacto_conv_descuento,
        0.01, 0.95
    ))

    nuevo_ticket   = base_ticket * (1 - (delta_descuento / 100) * 0.80)
    nuevas_visitas = base_visitas * impacto_visitas
    nuevas_ventas  = nuevas_visitas * nueva_conversion
    nuevo_ingreso  = nuevas_ventas * nuevo_ticket

    delta_ventas_abs  = nuevas_ventas  - base_ventas
    delta_ingreso_abs = nuevo_ingreso  - base_ingreso
    delta_conv_abs    = nueva_conversion - base_conversion

    # ── RESULTADOS ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="scenario-divider">
        <div class="scenario-divider-line"></div>
        <div class="scenario-divider-text">Proyección de resultados</div>
        <div class="scenario-divider-line"></div>
    </div>
    """, unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)

    def _delta_cls(val):
        if val > 0:   return "up"
        if val < 0:   return "down"
        return "flat"

    def _arrow(val):
        if val > 0: return "▲"
        if val < 0: return "▼"
        return "—"

    with r1:
        cv = "positive" if delta_ventas_abs >= 0 else "negative"
        dc = _delta_cls(delta_ventas_abs)
        st.markdown(f"""
        <div class="result-card highlight">
            <div class="result-value {cv}">{nuevas_ventas:.0f}</div>
            <div class="result-label">Ventas proyectadas</div>
            <div class="result-delta {dc}">{_arrow(delta_ventas_abs)} {abs(delta_ventas_abs):.0f} vs. actual</div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        cv = "positive" if delta_conv_abs >= 0 else "negative"
        dc = _delta_cls(delta_conv_abs)
        st.markdown(f"""
        <div class="result-card">
            <div class="result-value {cv}">{nueva_conversion:.1%}</div>
            <div class="result-label">Conversión proyectada</div>
            <div class="result-delta {dc}">{_arrow(delta_conv_abs)} {abs(delta_conv_abs):.1%} vs. actual</div>
        </div>
        """, unsafe_allow_html=True)

    with r3:
        # FIX: mostrar en Miles de Millones (MM) para valores COP reales
        cv = "positive" if delta_ingreso_abs >= 0 else "negative"
        dc = _delta_cls(delta_ingreso_abs)
        st.markdown(f"""
        <div class="result-card highlight">
            <div class="result-value {cv}">{_fmt_cop(nuevo_ingreso)}</div>
            <div class="result-label">Ingreso proyectado</div>
            <div class="result-delta {dc}">{_arrow(delta_ingreso_abs)} {_fmt_cop(abs(delta_ingreso_abs))} vs. actual</div>
        </div>
        """, unsafe_allow_html=True)

    with r4:
        dv = nuevas_visitas - base_visitas
        dc = _delta_cls(dv)
        st.markdown(f"""
        <div class="result-card neutral">
            <div class="result-value">{nuevas_visitas:.0f}</div>
            <div class="result-label">Visitas proyectadas</div>
            <div class="result-delta {dc}">{_arrow(dv)} {abs(dv):.0f} vs. actual</div>
        </div>
        """, unsafe_allow_html=True)

    # ── GRÁFICO COMPARATIVO — DOS SUBPLOTS ──────────────────────────────────────
    # FIX: ventas e ingresos tienen escalas completamente distintas.
    # Mezclarlos en el mismo eje distorsiona la visualización.
    # Solución: subplot izquierdo = ventas/visitas, derecho = ingreso en MM.
    st.markdown("<br>", unsafe_allow_html=True)

    NARANJA = "#D4521A"
    GRIS    = "#D4D1CB"
    NARANJA2 = "#E8860A"

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Ventas y Visitas", "Ingresos (Miles de Millones COP)"),
        horizontal_spacing=0.12,
    )

    # Panel izquierdo: ventas y visitas
    cats_left  = ["Ventas", "Visitas"]
    vals_base_left = [base_ventas,  base_visitas]
    vals_sim_left  = [nuevas_ventas, nuevas_visitas]

    fig.add_trace(go.Bar(
        name="Actual", x=cats_left, y=vals_base_left,
        marker_color=GRIS, marker_line_width=0,
        legendgroup="actual", showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        name="Simulado", x=cats_left, y=vals_sim_left,
        marker_color=NARANJA, marker_line_width=0,
        legendgroup="simulado", showlegend=True,
    ), row=1, col=1)

    # Panel derecho: ingreso en Miles de Millones
    fig.add_trace(go.Bar(
        name="Actual", x=["Ingreso"], y=[base_ingreso / 1e9],
        marker_color=GRIS, marker_line_width=0,
        legendgroup="actual", showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        name="Simulado", x=["Ingreso"], y=[nuevo_ingreso / 1e9],
        marker_color=NARANJA2, marker_line_width=0,
        legendgroup="simulado", showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#5C5A55", family="DM Sans", size=12),
        margin=dict(l=10, r=10, t=45, b=10),
        barmode="group",
        bargap=0.28,
        legend=dict(
            font=dict(color="#5C5A55"),
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom", y=1.05,
            xanchor="right", x=1,
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#E2E0DB",
            font=dict(color="#1A1916", family="DM Sans"),
        ),
    )
    fig.update_xaxes(gridcolor="#F0EFEC", linecolor="#E2E0DB", tickfont=dict(color="#9B9890"))
    fig.update_yaxes(gridcolor="#F0EFEC", linecolor="#E2E0DB", tickfont=dict(color="#9B9890"), zeroline=False)

    # Anotación de delta en el panel de ingresos
    delta_mm = delta_ingreso_abs / 1e9
    if delta_mm != 0:
        color_ann = "#2A7D4F" if delta_mm > 0 else "#C1392B"
        sign      = "+" if delta_mm > 0 else ""
        fig.add_annotation(
            x="Ingreso", y=max(base_ingreso, nuevo_ingreso) / 1e9,
            text=f"<b>{sign}{delta_mm:.2f} MM</b>",
            showarrow=False,
            font=dict(color=color_ann, size=12, family="DM Sans"),
            yshift=16,
            row=1, col=2,
        )

    st.plotly_chart(fig, use_container_width=True)

    # ── INSIGHT NARRATIVO ───────────────────────────────────────────────────────
    if delta_ingreso_abs > 0:
        # Identificar la palanca con mayor impacto
        impactos = {
            "vendedores":  abs(impacto_conv_vendedores),
            "respuesta":   abs(impacto_conv_respuesta),
            "descuento":   abs(impacto_conv_descuento),
            "marketing":   (delta_marketing / 100) * 0.6,
        }
        palanca_key = max(impactos, key=impactos.get)
        palanca_textos = {
            "vendedores": f"agregar <strong>{delta_vendedores} vendedor(es)</strong> en hora pico",
            "respuesta":  f"mejorar el tiempo de respuesta <strong>{delta_tiempo_respuesta}%</strong>",
            "descuento":  f"ajustar el descuento a <strong>+{delta_descuento}%</strong>",
            "marketing":  f"incrementar marketing <strong>{delta_marketing}%</strong>",
        }
        palanca = palanca_textos.get(palanca_key, "los ajustes aplicados")

        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-text">
                💡 La palanca principal es {palanca}. Con este escenario se proyectan
                <strong>{abs(delta_ventas_abs):.0f} ventas adicionales</strong> y un ingreso incremental de
                <strong>{_fmt_cop(delta_ingreso_abs)}</strong>.
                La conversión pasaría de <strong>{base_conversion:.1%}</strong>
                a <strong>{nueva_conversion:.1%}</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif delta_ingreso_abs < 0:
        st.markdown(f"""
        <div class="insight-box danger">
            <div class="insight-text">
                ⚠️ Este escenario <strong>reduciría los ingresos</strong> en
                <strong>{_fmt_cop(abs(delta_ingreso_abs))}</strong>.
                Ajusta los parámetros para encontrar la combinación óptima.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="insight-box neutral">
            <div class="insight-text">
                🎯 Mueve los sliders para simular diferentes escenarios y ver su
                impacto proyectado en ventas e ingresos en tiempo real.
            </div>
        </div>
        """, unsafe_allow_html=True)