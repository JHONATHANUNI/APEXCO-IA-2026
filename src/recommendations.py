def generate_recommendations(metrics, low_conversion_df, top_sellers_df=None, top_brands_df=None):
    recs = []

    if metrics["conversion"] < 0.15:
        recs.append("La conversión general es baja: revisar atención comercial, seguimiento y cierre.")
    if metrics["visitas_promedio"] < 5:
        recs.append("El tráfico es bajo: reforzar campañas, WhatsApp y marketing local.")
    if metrics["ticket_promedio"] < 90000:
        recs.append("El ticket promedio es bajo: promover versiones superiores o paquetes adicionales.")
    if len(low_conversion_df) > 0:
        recs.append("Hay horas con baja conversión: redistribuir personal en esas franjas.")
    if top_sellers_df is not None and len(top_sellers_df) > 0:
        best_seller = top_sellers_df.iloc[0]["vendedor"]
        recs.append(f"El vendedor con mejor desempeño es {best_seller}: documentar su proceso de venta.")
    if top_brands_df is not None and len(top_brands_df) > 0:
        best_brand = top_brands_df.iloc[0]["marca"]
        recs.append(f"La marca con mejor resultado es {best_brand}: priorizar inventario y campañas.")
    if not recs:
        recs.append("El desempeño es estable: mantener operación y probar mejoras incrementales.")

    return recs