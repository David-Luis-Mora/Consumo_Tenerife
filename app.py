from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Cargar modelo
model = joblib.load("model/model.pkl")

# ===== Schema entrada =====
class InputData(BaseModel):
    dia: int
    mes: int
    cups_municipio: str
    cups_distribuidor: str

# ===== Health check =====
@app.get("/health")
def health():
    try:
        return {"status": "ok"}
    except:
        return {"status": "ko"}

# ===== Predicción =====
@app.post("/predict")
def predict(data: InputData):

    df = pd.DataFrame([data.dict()])

    pred_log = model.predict(df)[0]
    pred_real = np.expm1(pred_log)

    return {
        "prediccion_kWh": float(pred_real)
    }
