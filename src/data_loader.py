import pandas as pd

def load_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()
    return df


def preprocess_data(df: pd.DataFrame):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    for col in ["precio_lista", "descuento", "precio_final", "visita", "venta"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "hora" in df.columns:
        df["hora_dt"] = pd.to_datetime(df["hora"], format="%H:%M", errors="coerce")
        df = df.dropna(subset=["hora_dt"])
        df["hora_int"] = df["hora_dt"].dt.hour
        df["hora_str"] = df["hora_dt"].dt.strftime("%H:%M")

    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df = df.dropna(subset=["fecha"])
        df["fecha_str"] = df["fecha"].dt.strftime("%Y-%m-%d")

    for col in ["concesionario", "vendedor", "marca", "modelo", "pais_origen", "tipo_auto", "color", "estado_lead", "canal"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df