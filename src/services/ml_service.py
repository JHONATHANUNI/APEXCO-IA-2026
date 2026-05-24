import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


MODEL_PATH = "models/sales_model.pkl"


def train_sales_model(df):

    data = df.copy()

    required_cols = [
        "visitas",
        "ventas",
        "hora_int"
    ]

    missing = [
        col for col in required_cols
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Columnas faltantes para ML: {', '.join(missing)}"
        )

    data = data.dropna(subset=required_cols)

    if len(data) < 10:
        raise ValueError(
            "No hay suficientes registros para entrenar el modelo."
        )

    X = data[["visitas", "hora_int"]]
    y = data["ventas"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model,
        MODEL_PATH
    )

    prediction_rows = []

    for i in range(len(X_test)):
        prediction_rows.append({
            "hora": int(X_test.iloc[i]["hora_int"]),
            "ventas": float(y_test.iloc[i]),
            "ventas_pred": float(predictions[i])
        })

    return {
        "model": model,
        "mae": mae,
        "predictions": prediction_rows
    }


def predict_sales(visitas, hora):

    try:

        model = joblib.load(MODEL_PATH)

        input_data = pd.DataFrame([{
            "visitas": visitas,
            "hora_int": hora
        }])

        prediction = model.predict(input_data)[0]

        return round(float(prediction), 2)

    except Exception as e:
        return f"Error ML: {str(e)}"