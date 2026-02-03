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


import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error
import numpy as np

mlflow.set_experiment("consumo_energetico_canarias")

with mlflow.start_run():

    # Entrenar modelo
    best_rf.fit(X_train, y_train)

    # Predicciones
    y_pred = best_rf.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Log de métricas
    mlflow.log_metric("rmse_test", rmse)

    # Log de parámetros
    mlflow.log_params(best_rf.get_params())

    # Log del modelo
    mlflow.sklearn.log_model(
        best_rf,
        artifact_path="model",
        registered_model_name="random_forest_consumo"
    )

print("Modelo y métricas registrados en MLflow")


# class PredictRequest(BaseModel):
#     mean_radius: float
#     mean_texture: float
#     mean_perimeter: float
#     mean_area: float
#     mean_smoothness: float


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
