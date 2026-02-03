from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="API Consumo Energético Canarias",
    description="Predicción de consumo energético diario",
    version="1.0"
)

# Cargar modelo
try:
    model = joblib.load("random_forest_model.pkl")
except:
    model = None

# ---------- Pydantic ----------
class InputData(BaseModel):
    dia: int
    mes: int
    dia_semana: int
    fin_de_semana: int
    cups_municipio: int
    cups_distribuidor: int

# ---------- Health ----------
@app.get("/health")
def health():
    if model is None:
        return {"status": "ko"}
    return {"status": "ok"}

# ---------- Predict ----------
@app.post("/predict")
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")

    features = np.array([[
        data.dia,
        data.mes,
        data.dia_semana,
        data.fin_de_semana,
        data.cups_municipio,
        data.cups_distribuidor
    ]])

    prediction = model.predict(features)

    return {
        "prediccion_consumo": float(prediction[0])
    }
