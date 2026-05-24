from ollama import chat


def build_context(
    metrics=None,
    recommendations=None,
    patterns=None,
    sellers_df=None,
    brands_df=None,
    low_conversion_df=None
):
    """
    Construye el contexto inteligente del concesionario
    para APEXCO IA usando datos reales del negocio.
    """

    context_parts = []

    # =========================
    # MÉTRICAS PRINCIPALES
    # =========================
    if metrics:
        context_parts.append(f"""
MÉTRICAS ACTUALES DEL CONCESIONARIO:
- Ventas totales: {metrics.get('total_ventas', 0):.0f}
- Visitas totales: {metrics.get('total_visitas', 0):.0f}
- Tasa de conversión: {metrics.get('conversion', 0):.2%}
- Ticket promedio: ${metrics.get('ticket_promedio', 0):,.0f}
- Ventas promedio por registro: {metrics.get('ventas_promedio', 0):.2f}
- Visitas promedio por registro: {metrics.get('visitas_promedio', 0):.2f}
""")

    # =========================
    # PATRONES
    # =========================
    if patterns:
        context_parts.append(f"""
PATRONES DETECTADOS:
- Mejor hora de conversión: {patterns.get('best_hour', 'N/A')}:00
- Peor hora de conversión: {patterns.get('worst_hour', 'N/A')}:00
- Horas de mayor tráfico: {patterns.get('high_traffic_hours', [])}
""")

    # =========================
    # RECOMENDACIONES
    # =========================
    if recommendations:
        recommendations_text = "\n".join(
            f"- {r}" for r in recommendations
        )

        context_parts.append(f"""
RECOMENDACIONES DEL SISTEMA:
{recommendations_text}
""")

    # =========================
    # TOP VENDEDORES
    # =========================
    if sellers_df is not None and len(sellers_df) > 0:

        top3 = sellers_df.head(3)

        sellers_info = "\n".join(
            f"• {row['vendedor']}: "
            f"{row['ventas']:.0f} ventas | "
            f"Conversión {(row['ventas'] / row['visitas']):.2%}"
            for _, row in top3.iterrows()
            if row['visitas'] > 0
        )

        context_parts.append(f"""
TOP VENDEDORES:
{sellers_info}
""")

    # =========================
    # TOP MARCAS
    # =========================
    if brands_df is not None and len(brands_df) > 0:

        top3b = brands_df.head(3)

        brands_info = "\n".join(
            f"• {row['marca']}: {row['ventas']:.0f} ventas"
            for _, row in top3b.iterrows()
        )

        context_parts.append(f"""
TOP MARCAS:
{brands_info}
""")

    # =========================
    # HORAS MALAS
    # =========================
    if low_conversion_df is not None and len(low_conversion_df) > 0:

        hours = low_conversion_df['hora_int'].tolist()

        context_parts.append(f"""
HORAS CON BAJA CONVERSIÓN (menos del 15%):
{hours}
""")

    return "\n".join(context_parts) if context_parts else "No hay datos disponibles aún."


# ==========================================================
# CHAT PRINCIPAL APEXCO IA
# ==========================================================

def get_chat_response(
    user_text,
    metrics=None,
    recommendations=None,
    patterns=None,
    sellers_df=None,
    brands_df=None,
    low_conversion_df=None,
    conversation_history=None
):
    """
    Chat inteligente local usando Ollama + Qwen 2.5.
    NO usa créditos.
    NO usa internet.
    TODO corre localmente.
    """

    # =========================
    # CONTEXTO DEL NEGOCIO
    # =========================
    context = build_context(
        metrics,
        recommendations,
        patterns,
        sellers_df,
        brands_df,
        low_conversion_df
    )

    # =========================
    # SYSTEM PROMPT
    # =========================
    system_prompt = f"""
Eres APEX, la inteligencia artificial de APEXCO IA.

APEXCO IA es una plataforma SaaS de auditoría inteligente para concesionarios de autos en Colombia.

Tu trabajo es analizar métricas comerciales y operativas para detectar:

- pérdidas de ventas
- problemas de conversión
- horarios muertos
- vendedores con bajo rendimiento
- oportunidades comerciales
- patrones de comportamiento

Debes actuar como:

- auditor comercial senior
- consultor estratégico automotriz
- analista operativo
- experto en ventas

DATOS ACTUALES:
{context}

REGLAS:

- Responde SIEMPRE en español
- Sé profesional y ejecutivo
- Usa los datos reales
- Nunca inventes información
- Da recomendaciones accionables
- Usa bullets cuando sea útil
- Sé directo y estratégico
- Máximo 180 palabras
- Si detectas un problema importante, dilo claramente
- Prioriza insights de negocio y rentabilidad
"""
    # =========================
    # HISTORIAL DE CHAT
    # =========================
    messages = []

    if conversation_history:

        for msg in conversation_history[-6:]:

            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    # MENSAJE ACTUAL
    messages.append({
        "role": "user",
        "content": user_text
    })

    # =========================
    # RESPUESTA IA LOCAL
    # =========================
    try:

        response = chat(
            model="gemma3:4b",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                *messages
            ],
            
            options={
        "temperature": 0.4,
        "num_predict": 250
    }
)
        return response["message"]["content"]

    except Exception as e:

        return f"❌ Error al conectar con Ollama: {str(e)}"