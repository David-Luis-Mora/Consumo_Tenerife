from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np


URL_DB = "sqlite:///mlflow.db"

mlflow.set_tracking_uri("http://127.0.0.1:5000")# la URL de MLFLOW. Aquí es local, pero si estuviese en la nube...
mlflow.set_tracking_uri(URL_DB)

URL_MODELO = "/home/inta@informatica.edu/Escritorio/Consumo_Tenerife/mlruns/2/models/m-e12109d1bb05444dace5cffa5ce3680d/artifacts/"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Antes del yield se lanzará al iniciar el server
    # cargamos el modelo con pickle
    try:
        app.state.modelo = mlflow.sklearn.load_model(URL_MODELO)
        print("Modelo cargado desde MLflow")
    except Exception as e:
        print("Error cargando el modelo:", e)
        app.state.modelo = None

    yield

    # Esto se lanzará cuando apaguemos el server.
    print("Aplicación detenida")


app = FastAPI(
    title="API Cáncer de mama",
    lifespan=lifespan
)


class PredictRequest(BaseModel):
    dia: int
    mes: int
    cups_municipio: str
    cups_distribuidor: str


# @app.post("/predict")
# def predict(data: PredictRequest):
#     if app.state.modelo is None:
#         raise HTTPException(status_code=500, detail="No se ha cargado el modelo :(")

#     X = pd.DataFrame([{
#         "mean radius": data.mean_radius,
#         "mean texture": data.mean_texture,
#         "mean perimeter": data.mean_perimeter,
#         "mean area": data.mean_area,
#         "mean smoothness": data.mean_smoothness
#     }])

#     proba = app.state.modelo.predict(X)
#     p = float(proba[0])

#     return {
#         "prediccion": 1 if p >= 0.5 else 0,
#         "probabilidad_benigno": p,
#         "probabilidad_maligno": 1 - p
#     }


@app.post("/predict")
def predict(data: PredictRequest):

    if app.state.modelo is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")

    X = pd.DataFrame([{
        "dia": data.dia,
        "mes": data.mes,
        "cups_municipio": data.cups_municipio,
        "cups_distribuidor": data.cups_distribuidor
    }])

    pred_log = app.state.modelo.predict(X)[0]

    # Volver de log a kWh
    pred_kwh = float(pd.np.expm1(pred_log))

    return {"prediccion_kWh": pred_kwh}
