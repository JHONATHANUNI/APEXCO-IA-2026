import streamlit as st
import plotly.graph_objects as go
import numpy as np

def render_simulator(metrics: dict, patterns: dict, sellers_df=None):
    """
    Renderiza el módulo de simulación de escenarios dentro de la app Streamlit.
    Recibe las métricas reales del negocio y proyecta el impacto de cambios.
    """

    # CSS del simulador
    st.markdown("""
    <style>
    .sim-header {
        background: linear-gradient(135deg, rgba(255,87,34,0.08), rgba(255,152,0,0.05));
        border: 1px solid rgba(255,87,34,0.2);
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
        opacity: 0.12;
    }
    .sim-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.3rem;
        font-weight: 800;
        color: #fff;
        margin-bottom: 0.3rem;
    }
    .sim-subtitle {
        font-size: 0.82rem;
        color: rgba(255,255,255,0.4);
    }
    .sim-badge {
        display: inline-block;
        background: rgba(255,87,34,0.2);
        border: 1px solid rgba(255,87,34,0.4);
        color: #FF7043;
        font-size: 0.65rem;
        padding: 0.15rem 0.6rem;
        border-radius: 100px;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .result-card {
        background: #111118;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .result-card.highlight {
        border-color: rgba(255,87,34,0.35);
        background: linear-gradient(135deg, rgba(255,87,34,0.08), #111118);
    }
    .result-card::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #FF5722, #FF9800);
    }
    .result-card.neutral::after {
        background: rgba(255,255,255,0.1);
    }
    .result-value {
        font-family: 'Syne', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        color: #fff;
        line-height: 1;
        margin-bottom: 0.3rem;
    }
    .result-value.positive { color: #4CAF50; }
    .result-value.negative { color: #FF5722; }
    .result-label {
        font-size: 0.72rem;
        color: rgba(255,255,255,0.4);
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .result-delta {
        font-size: 0.78rem;
        margin-top: 0.4rem;
        font-weight: 600;
    }
    .result-delta.up { color: #4CAF50; }
    .result-delta.down { color: #FF5722; }
    .scenario-divider {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .scenario-divider-line {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    }
    .scenario-divider-text {
        font-size: 0.7rem;
        color: rgba(255,255,255,0.3);
        text-transform: uppercase;
        letter-spacing: 0.15em;
        white-space: nowrap;
    }
    .insight-box {
        background: rgba(255,152,0,0.07);
        border: 1px solid rgba(255,152,0,0.2);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
    }
    .insight-text {
        font-size: 0.88rem;
        color: rgba(255,255,255,0.8);
        line-height: 1.6;
    }
    .insight-text strong { color: #FFB74D; }
    </style>
    """, unsafe_allow_html=True)

    # ── HEADER ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sim-header">
        <div class="sim-badge">🔬 Simulador de escenarios</div>
        <div class="sim-title">¿Qué pasa si cambias esto?</div>
        <div class="sim-subtitle">
            Ajusta los parámetros y ve el impacto proyectado en ventas e ingresos en tiempo real.
            Basado en los datos reales de tu operación.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── DATOS BASE ──────────────────────────────────────────────────────────────
    base_ventas      = metrics.get("total_ventas", 50)
    base_visitas     = metrics.get("total_visitas", 200)
    base_conversion  = metrics.get("conversion", 0.25)
    base_ticket      = metrics.get("ticket_promedio", 80_000_000)

    if base_ticket < 1_000:
        base_ticket = 80_000_000  # fallback si no tiene precio real

    base_ingreso = base_ventas * base_ticket

    # ── SLIDERS ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="scenario-divider"><div class="scenario-divider-line"></div><div class="scenario-divider-text">Ajusta los parámetros</div><div class="scenario-divider-line"></div></div>', unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        delta_vendedores = st.slider(
            "👥 Vendedores adicionales en hora pico",
            min_value=-2, max_value=5, value=0, step=1,
            help="Agrega o quita asesores en las horas de mayor tráfico"
        )
        delta_tiempo_respuesta = st.slider(
            "⚡ Mejora en tiempo de respuesta (%)",
            min_value=0, max_value=80, value=0, step=5,
            help="Reducir el tiempo de respuesta al cliente mejora la conversión"
        )

    with col_s2:
        delta_marketing = st.slider(
            "📣 Incremento en inversión de marketing (%)",
            min_value=0, max_value=100, value=0, step=10,
            help="Mayor inversión en canales digitales atrae más visitas"
        )
        delta_descuento = st.slider(
            "🏷️ Ajuste en descuento promedio (%)",
            min_value=-5, max_value=10, value=0, step=1,
            help="Ofrecer más descuento puede subir conversión pero baja ticket"
        )

    # ── MODELO DE SIMULACIÓN ────────────────────────────────────────────────────
    # Impacto en visitas: marketing genera tráfico
    impacto_visitas = 1 + (delta_marketing / 100) * 0.6

    # Impacto en conversión
    impacto_conv_vendedores    = delta_vendedores * 0.04        # +4% por vendedor extra
    impacto_conv_respuesta     = (delta_tiempo_respuesta / 100) * 0.18  # hasta +18%
    impacto_conv_descuento     = (delta_descuento / 100) * 0.5  # descuento sube conv

    nueva_conversion = min(
        base_conversion + impacto_conv_vendedores + impacto_conv_respuesta + impacto_conv_descuento,
        0.95
    )
    nueva_conversion = max(nueva_conversion, 0.01)

    # Impacto en ticket: descuento lo baja
    nuevo_ticket = base_ticket * (1 - (delta_descuento / 100) * 0.8)

    # Nuevas ventas e ingresos
    nuevas_visitas = base_visitas * impacto_visitas
    nuevas_ventas  = nuevas_visitas * nueva_conversion
    nuevo_ingreso  = nuevas_ventas * nuevo_ticket

    # Deltas
    delta_ventas_abs   = nuevas_ventas - base_ventas
    delta_ingreso_abs  = nuevo_ingreso - base_ingreso
    delta_conv_abs     = nueva_conversion - base_conversion

    # ── RESULTADOS ──────────────────────────────────────────────────────────────
    st.markdown('<div class="scenario-divider"><div class="scenario-divider-line"></div><div class="scenario-divider-text">Proyección de resultados</div><div class="scenario-divider-line"></div></div>', unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        color_v = "positive" if delta_ventas_abs >= 0 else "negative"
        arrow_v = "▲" if delta_ventas_abs >= 0 else "▼"
        cls_v   = "up" if delta_ventas_abs >= 0 else "down"
        st.markdown(f"""
        <div class="result-card highlight">
            <div class="result-value {color_v}">{nuevas_ventas:.0f}</div>
            <div class="result-label">Ventas proyectadas</div>
            <div class="result-delta {cls_v}">{arrow_v} {abs(delta_ventas_abs):.1f} vs. actual</div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        color_c = "positive" if delta_conv_abs >= 0 else "negative"
        arrow_c = "▲" if delta_conv_abs >= 0 else "▼"
        cls_c   = "up" if delta_conv_abs >= 0 else "down"
        st.markdown(f"""
        <div class="result-card">
            <div class="result-value {color_c}">{nueva_conversion:.1%}</div>
            <div class="result-label">Conversión proyectada</div>
            <div class="result-delta {cls_c}">{arrow_c} {abs(delta_conv_abs):.1%} vs. actual</div>
        </div>
        """, unsafe_allow_html=True)

    with r3:
        color_i = "positive" if delta_ingreso_abs >= 0 else "negative"
        arrow_i = "▲" if delta_ingreso_abs >= 0 else "▼"
        cls_i   = "up" if delta_ingreso_abs >= 0 else "down"
        ingreso_fmt = f"${nuevo_ingreso/1_000_000:.1f}M"
        st.markdown(f"""
        <div class="result-card highlight">
            <div class="result-value {color_i}">{ingreso_fmt}</div>
            <div class="result-label">Ingreso proyectado</div>
            <div class="result-delta {cls_i}">{arrow_i} ${abs(delta_ingreso_abs)/1_000_000:.1f}M vs. actual</div>
        </div>
        """, unsafe_allow_html=True)

    with r4:
        st.markdown(f"""
        <div class="result-card neutral">
            <div class="result-value">{nuevas_visitas:.0f}</div>
            <div class="result-label">Visitas proyectadas</div>
            <div class="result-delta up">▲ {abs(nuevas_visitas - base_visitas):.0f} vs. actual</div>
        </div>
        """, unsafe_allow_html=True)

    # ── GRÁFICO COMPARATIVO ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    categorias = ["Ventas", "Visitas", "Conversión (×100)", "Ingreso (÷1M)"]
    valores_base = [
        base_ventas,
        base_visitas,
        base_conversion * 100,
        base_ingreso / 1_000_000
    ]
    valores_sim = [
        nuevas_ventas,
        nuevas_visitas,
        nueva_conversion * 100,
        nuevo_ingreso / 1_000_000
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Actual",
        x=categorias,
        y=valores_base,
        marker_color="rgba(255,255,255,0.15)",
        marker_line_width=0
    ))
    fig.add_trace(go.Bar(
        name="Simulado",
        x=categorias,
        y=valores_sim,
        marker_color="#FF5722",
        marker_line_width=0
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.6)", family="DM Sans"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        margin=dict(l=10, r=10, t=30, b=10),
        barmode="group",
        legend=dict(
            font=dict(color="rgba(255,255,255,0.5)"),
            bgcolor="rgba(0,0,0,0)"
        ),
        title=dict(
            text="Actual vs. Escenario simulado",
            font=dict(color="rgba(255,255,255,0.7)", size=13)
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── INSIGHT GENERADO ────────────────────────────────────────────────────────
    if delta_ingreso_abs > 0:
        mejor_palanca = ""
        if delta_vendedores > 0 and impacto_conv_vendedores == max(impacto_conv_vendedores, impacto_conv_respuesta, impacto_conv_descuento):
            mejor_palanca = f"agregar <strong>{delta_vendedores} vendedor(es)</strong> en hora pico"
        elif delta_tiempo_respuesta > 0:
            mejor_palanca = f"mejorar el tiempo de respuesta un <strong>{delta_tiempo_respuesta}%</strong>"
        elif delta_marketing > 0:
            mejor_palanca = f"incrementar el marketing un <strong>{delta_marketing}%</strong>"
        else:
            mejor_palanca = "los ajustes aplicados"

        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-text">
                💡 Con este escenario, {mejor_palanca} generaría
                <strong>${delta_ingreso_abs/1_000_000:.1f}M de pesos adicionales</strong>
                y <strong>{delta_ventas_abs:.0f} ventas más</strong> sobre tu operación actual.
                La conversión pasaría de <strong>{base_conversion:.1%}</strong> a <strong>{nueva_conversion:.1%}</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif delta_ingreso_abs < 0:
        st.markdown(f"""
        <div class="insight-box" style="border-color:rgba(255,87,34,0.3); background:rgba(255,87,34,0.05);">
            <div class="insight-text">
                ⚠️ Este escenario reduciría los ingresos en
                <strong>${abs(delta_ingreso_abs)/1_000_000:.1f}M</strong>.
                Ajusta los parámetros para encontrar la combinación óptima.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="insight-box">
            <div class="insight-text">
                🎯 Mueve los sliders para simular diferentes escenarios y ver su impacto proyectado en tiempo real.
            </div>
        </div>
        """, unsafe_allow_html=True)