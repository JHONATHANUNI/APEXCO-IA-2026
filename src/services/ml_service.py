import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

MODEL_PATH = "models/sales_model.pkl"


def train_sales_model(df):
    """
    Entrena un modelo de predicción de ventas por hora.

    FIX CLAVE: Agrupa los datos por hora antes de entrenar.
    El dataset tiene una fila por visita (venta=0 o 1), por lo que
    entrenar fila por fila hace que el modelo aprenda a predecir
    0.0–0.4 siempre. Al agrupar, el modelo ve el total real de
    ventas y visitas por franja horaria y predice volúmenes útiles.
    """

    data = df.copy()

    # Columnas mínimas requeridas en el df original
    required_cols = ["hora_int", "venta", "visita"]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Columnas faltantes para ML: {', '.join(missing)}")

    data = data.dropna(subset=required_cols)

    if len(data) < 10:
        raise ValueError("No hay suficientes registros para entrenar el modelo.")

    # ── AGREGACIÓN POR HORA ────────────────────────────────────────────────────
    # Cada fila del dataset = 1 visita individual.
    # Agrupamos para obtener totales reales por franja horaria.
    hourly = (
        data
        .groupby("hora_int", as_index=False)
        .agg(
            ventas=("venta",   "sum"),
            visitas=("visita", "sum"),
        )
    )
    hourly["conversion"] = hourly["ventas"] / hourly["visitas"].clip(lower=1)

    if len(hourly) < 5:
        raise ValueError("No hay suficientes franjas horarias para entrenar (mínimo 5).")

    # Features: visitas por hora + hora del día (patrón temporal)
    X = hourly[["visitas", "hora_int"]].values
    y = hourly["ventas"].values

    # ── MODELO ────────────────────────────────────────────────────────────────
    # Con pocos puntos (13 horas) usamos Leave-One-Out CV para evaluar
    # sin sobreajustar el split train/test.
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=4,           # evita overfitting con dataset pequeño
        min_samples_leaf=2,
        random_state=42,
    )

    # Evaluación con LOO-CV (más robusta con n=13)
    loo = LeaveOneOut()
    y_pred_cv = cross_val_predict(model, X, y, cv=loo)

    mae  = mean_absolute_error(y, y_pred_cv)
    r2   = r2_score(y, y_pred_cv)

    # Entrenar modelo final con todos los datos
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # Predicciones finales para visualización (real vs predicho por hora)
    y_pred_final = model.predict(X)

    prediction_rows = []
    for i in range(len(hourly)):
        prediction_rows.append({
            "hora":        int(hourly.iloc[i]["hora_int"]),
            "ventas":      float(hourly.iloc[i]["ventas"]),
            "ventas_pred": float(y_pred_final[i]),
            "visitas":     float(hourly.iloc[i]["visitas"]),
            "conversion":  float(hourly.iloc[i]["conversion"]),
        })

    # Ordenar por hora para gráfica coherente
    prediction_rows.sort(key=lambda x: x["hora"])

    return {
        "model":       model,
        "mae":         mae,
        "r2":          r2,
        "predictions": prediction_rows,
        "hourly_df":   hourly,
    }


def predict_sales(visitas: float, hora: int) -> float:
    """Predicción puntual para una franja horaria con N visitas esperadas."""
    try:
        model = joblib.load(MODEL_PATH)
        input_data = pd.DataFrame([{"visitas": visitas, "hora_int": hora}])
        prediction = model.predict(input_data)[0]
        return round(float(prediction), 1)
    except Exception as e:
        return f"Error ML: {str(e)}"