# 🧠 Proyecto: MLflow + FastAPI  
## Clasificación de cáncer de mama (Breast Cancer Dataset)

Este proyecto muestra un **flujo completo de Machine Learning**, desde el **entrenamiento y registro de modelos con MLflow** hasta su **despliegue como API REST con FastAPI**, usando el dataset clásico de **cáncer de mama** de `scikit-learn`.

---

## 📁 Estructura del proyecto

```text
.
├── EntrenoMLFlow.py   # Entrenamiento + experimentación con MLflow
├── APICancer.py       # API REST (FastAPI) para servir el modelo
├── mlflow.db          # Base de datos local de MLflow (SQLite)
└── mlruns/            # Artefactos y runs de MLflow
```

---

---

## Instalar dependencias

```bash
pip install -r requirements.txt
```
---

## 🧪 1. Entrenamiento y experimentación (`EntrenoMLFlow.py`)

Este script:

- Carga el dataset **Breast Cancer** de `sklearn`
- Usa **solo 5 variables** relevantes:
  - `mean radius`
  - `mean texture`
  - `mean perimeter`
  - `mean area`
  - `mean smoothness`
- Entrena **varios Árboles de Decisión** con distintas combinaciones de hiperparámetros
- Registra **parámetros, métricas y modelos** en **MLflow**

### Métricas registradas
- Accuracy
- F1-score
- Precision
- Recall
- ROC AUC
- Matriz de confusión (tn, fp, fn, tp)
- Overfitting (`accuracy_gap`)

### Cómo ejecutarlo

```bash
pip install scikit-learn mlflow
python EntrenoMLFlow.py
```

Para visualizar los experimentos:

```bash
mlflow ui
```

Abrir en el navegador:
```
http://127.0.0.1:5000
```

---

## 📦 2. API de predicción (`APICancer.py`)

Este fichero implementa una **API REST con FastAPI** que:

- Carga un modelo entrenado desde **MLflow**
- Usa el **ciclo de vida (`lifespan`)**, patrón recomendado oficialmente
- Expone un endpoint `/predict` para hacer inferencias
- Devuelve **probabilidades**, no solo la clase

### Variables de entrada (JSON)

```json
{
  "mean_radius": 14.5,
  "mean_texture": 18.2,
  "mean_perimeter": 95.0,
  "mean_area": 680.0,
  "mean_smoothness": 0.097
}
```

### Respuesta de la API

```json
{
  "prediccion": 1,
  "probabilidad_benigno": 0.56,
  "probabilidad_maligno": 0.44
}
```

> `1 = benigno`  
> `0 = maligno`

---

## 🚀 Cómo ejecutar la API

### 1️⃣ Asegúrate de que MLflow está activo
```bash
mlflow ui
```

### 2️⃣ Ajusta la URI del modelo en `APICancer.py`

```python
URL_MODELO = "models:/NOMBRE MODELO/VERSION"
```

Ejemplo:
```python
URL_MODELO = "models:/modelo/1"
```

### 3️⃣ Arranca la API
```bash
pip install fastapi uvicorn pandas mlflow
uvicorn APICancer:app --reload
```

Swagger automático:
```
http://127.0.0.1:8000/docs
```

---

## 🧠 Decisiones técnicas importantes

- ✔ Se usa **MLflow** para experimentación y versionado
- ✔ El modelo se carga **una sola vez** al arrancar la API
- ✔ Se devuelven **probabilidades**, no solo clases
- ✔ Arquitectura preparada para frontend externo (Angular, HTML, Gradio)

---
