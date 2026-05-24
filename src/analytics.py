import pandas as pd
import numpy as np

def compute_metrics(df: pd.DataFrame):
    total_ventas = float(df["venta"].sum())
    total_visitas = float(df["visita"].sum())
    conversion = (total_ventas / total_visitas) if total_visitas > 0 else 0
    ventas_promedio = float(df["venta"].mean()) if len(df) else 0
    visitas_promedio = float(df["visita"].mean()) if len(df) else 0
    ticket_promedio = float(df["precio_final"].mean()) if "precio_final" in df.columns and len(df) else 0

    return {
        "total_ventas": total_ventas,
        "total_visitas": total_visitas,
        "conversion": conversion,
        "ventas_promedio": ventas_promedio,
        "visitas_promedio": visitas_promedio,
        "ticket_promedio": ticket_promedio,
    }


def sales_by_hour(df: pd.DataFrame):
    hourly = df.groupby("hora_int", as_index=False).agg(
        ventas=("venta", "sum"),
        visitas=("visita", "sum"),
        ticket_promedio=("precio_final", "mean")
    )
    hourly["conversion"] = np.where(hourly["visitas"] > 0, hourly["ventas"] / hourly["visitas"], 0)
    return hourly.sort_values("hora_int")


def sales_by_day(df: pd.DataFrame):
    daily = df.groupby("fecha_str", as_index=False).agg(
        ventas=("venta", "sum"),
        visitas=("visita", "sum"),
        ticket_promedio=("precio_final", "mean")
    )
    daily["conversion"] = np.where(daily["visitas"] > 0, daily["ventas"] / daily["visitas"], 0)
    return daily


def top_sellers(df: pd.DataFrame):
    return df.groupby("vendedor", as_index=False).agg(
        ventas=("venta", "sum"),
        visitas=("visita", "sum"),
        ticket_promedio=("precio_final", "mean")
    ).sort_values("ventas", ascending=False)


def top_brands(df: pd.DataFrame):
    return df.groupby("marca", as_index=False).agg(
        ventas=("venta", "sum"),
        visitas=("visita", "sum"),
        ticket_promedio=("precio_final", "mean")
    ).sort_values("ventas", ascending=False)


def detect_low_conversion(df: pd.DataFrame, threshold=0.15):
    hourly = sales_by_hour(df)
    return hourly[hourly["conversion"] < threshold].copy()


def detect_patterns(df: pd.DataFrame):
    hourly = sales_by_hour(df)
    if len(hourly) == 0:
        return {"best_hour": None, "worst_hour": None, "high_traffic_hours": []}

    best_hour = int(hourly.sort_values("conversion", ascending=False).iloc[0]["hora_int"])
    worst_hour = int(hourly.sort_values("conversion", ascending=True).iloc[0]["hora_int"])
    high_traffic_hours = hourly.sort_values("visitas", ascending=False).head(3)["hora_int"].tolist()

    return {
        "best_hour": best_hour,
        "worst_hour": worst_hour,
        "high_traffic_hours": high_traffic_hours
    }