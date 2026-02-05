from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import pandas as pd

URL_DB = "sqlite:///mlflow.db"

mlflow.set_tracking_uri("http://127.0.0.1:5000")# la URL de MLFLOW. Aquí es local, pero si estuviese en la nube...
mlflow.set_tracking_uri(URL_DB)

URL_MODELO = "models:/NOMBRE MODELO/VERSION"


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

# -----------------------
# App
# -----------------------
app = FastAPI(
    title="API Cáncer de mama",
    lifespan=lifespan
)


class PredictRequest(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float


@app.post("/predict")
def predict(data: PredictRequest):
    if app.state.modelo is None:
        raise HTTPException(status_code=500, detail="No se ha cargado el modelo :(")

    X = pd.DataFrame([{
        "mean radius": data.mean_radius,
        "mean texture": data.mean_texture,
        "mean perimeter": data.mean_perimeter,
        "mean area": data.mean_area,
        "mean smoothness": data.mean_smoothness
    }])

    proba = app.state.modelo.predict(X)
    p = float(proba[0])

    return {
        "prediccion": 1 if p >= 0.5 else 0,
        "probabilidad_benigno": p,
        "probabilidad_maligno": 1 - p
    }