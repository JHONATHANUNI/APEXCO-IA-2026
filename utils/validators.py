import pandas as pd

REQUIRED_COLUMNS = [
    "fecha", "hora", "concesionario", "vendedor", "marca", "modelo",
    "pais_origen", "tipo_auto", "color", "precio_lista", "descuento",
    "precio_final", "estado_lead", "visita", "venta", "canal"
]

NUMERIC_COLUMNS = ["precio_lista", "descuento", "precio_final", "visita", "venta"]


def validate_required_columns(df: pd.DataFrame):
    df_cols = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df_cols]
    return len(missing) == 0, missing


def validate_basic_format(df: pd.DataFrame):
    errors = []
    if "hora" in df.columns:
        try:
            pd.to_datetime(df["hora"], format="%H:%M", errors="raise")
        except Exception:
            errors.append("La columna 'hora' debe tener formato HH:MM.")
    if "fecha" in df.columns:
        try:
            pd.to_datetime(df["fecha"], errors="raise")
        except Exception:
            errors.append("La columna 'fecha' tiene valores inválidos.")
    return len(errors) == 0, errors


def validate_numeric_columns(df: pd.DataFrame):
    errors = []
    for col in NUMERIC_COLUMNS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"La columna '{col}' debe ser numérica.")
    return len(errors) == 0, errors