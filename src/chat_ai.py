import anthropic
import os
import json


def build_context(metrics=None, recommendations=None, patterns=None, sellers_df=None, brands_df=None, low_conversion_df=None):
    """Construye el contexto del negocio para pasarle a Claude."""
    context_parts = []

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

    if patterns:
        context_parts.append(f"""
PATRONES DETECTADOS:
- Mejor hora de conversión: {patterns.get('best_hour', 'N/A')}:00
- Peor hora de conversión: {patterns.get('worst_hour', 'N/A')}:00
- Horas de mayor tráfico: {patterns.get('high_traffic_hours', [])}
""")

    if recommendations:
        context_parts.append(f"""
RECOMENDACIONES DEL SISTEMA:
{chr(10).join(f'- {r}' for r in recommendations)}
""")

    if sellers_df is not None and len(sellers_df) > 0:
        top3 = sellers_df.head(3)
        sellers_info = "\n".join(
            f"  • {row['vendedor']}: {row['ventas']:.0f} ventas, conversión {row['ventas']/row['visitas']:.2%}"
            for _, row in top3.iterrows() if row['visitas'] > 0
        )
        context_parts.append(f"""
TOP VENDEDORES:
{sellers_info}
""")

    if brands_df is not None and len(brands_df) > 0:
        top3b = brands_df.head(3)
        brands_info = "\n".join(
            f"  • {row['marca']}: {row['ventas']:.0f} ventas"
            for _, row in top3b.iterrows()
        )
        context_parts.append(f"""
TOP MARCAS:
{brands_info}
""")

    if low_conversion_df is not None and len(low_conversion_df) > 0:
        hours = low_conversion_df['hora_int'].tolist()
        context_parts.append(f"""
HORAS CON BAJA CONVERSIÓN (menor al 15%): {hours}
""")

    return "\n".join(context_parts) if context_parts else "No hay datos disponibles aún."


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
    Llama a la API de Claude con el contexto del negocio y el historial de conversación.
    Retorna la respuesta como string.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "⚠️ No se encontró la API key de Anthropic. Configura ANTHROPIC_API_KEY en tu archivo .env"

    context = build_context(metrics, recommendations, patterns, sellers_df, brands_df, low_conversion_df)

    system_prompt = f"""Eres APEX, el asistente de inteligencia artificial de APEXCO AI, una plataforma de análisis operativo para concesionarios de autos en Colombia.

Tu rol es actuar como un auditor comercial experto que analiza los datos del concesionario y da recomendaciones claras, directas y accionables.

DATOS ACTUALES DEL CONCESIONARIO:
{context}

INSTRUCCIONES:
- Responde siempre en español, de forma clara y directa
- Usa los datos reales del contexto para fundamentar tus respuestas
- Si detectas problemas, sé específico: menciona números, horas, vendedores o marcas exactas
- Da recomendaciones concretas y accionables, no genéricas
- Si te preguntan algo que no está en los datos, dilo honestamente
- Usa emojis con moderación para hacer la respuesta más visual
- Máximo 150 palabras por respuesta, sé conciso y poderoso
- Nunca inventes datos que no están en el contexto
"""

    messages = []
    if conversation_history:
        for msg in conversation_history[-6:]:  # últimos 6 mensajes para no saturar tokens
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    messages.append({"role": "user", "content": user_text})

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            system=system_prompt,
            messages=messages
        )
        return response.content[0].text

    except anthropic.AuthenticationError:
        return "❌ API key inválida. Verifica tu ANTHROPIC_API_KEY en el archivo .env"
    except anthropic.RateLimitError:
        return "⏳ Límite de uso alcanzado. Intenta en unos segundos."
    except Exception as e:
        return f"❌ Error al conectar con la IA: {str(e)}"