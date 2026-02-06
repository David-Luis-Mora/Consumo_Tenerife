import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

# Cojo datos y los divido para entrenar.


data = pd.read_csv('consumo-energetico-2025.csv')

data['fecha'] = pd.to_datetime(data['fecha'])

features = ['dia', 'mes','cups_municipio', 'cups_distribuidor']
data['dia'] = data['fecha'].dt.day
data['mes'] = data['fecha'].dt.month
data['consumo_log'] = np.log1p(data['consumo'])
data.drop(columns=['fecha'],inplace=True)

features = ['dia', 'mes','cups_municipio', 'cups_distribuidor']

X = data[features]
y = data['consumo_log']


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    # stratify=y
)

# Aquí establezco los parámetros de MLFlow.
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("GradientBoostingRegressor")

# Al igual que vimos con el CVGridSearch, pongo unos cuántos valores de hiperparámetros.
max_depths = [None, 2]
min_splits = [2, 5]

max_depth = 10
min_samples_split = 5
n_estimators = 50

model = GradientBoostingRegressor(
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    n_estimators=n_estimators,
    random_state=42
)

num_features = ['dia', 'mes']
cat_features = ['cups_municipio', 'cups_distribuidor']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ]
)

gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(random_state=42))
])

# Ahora simulo un poco lo que hace dicha clase.
# Lanzo todas las combinaciones...
# ... y las logueo.

run_name = (
    f"Lanzamiento de árbol de decisión, con profundidad_hoja={max_depth} y "
    f" split={min_samples_split}"
)

gb_pipeline.fit(X_train, y_train)

y_pred = gb_pipeline.predict(X_test)
y_train_pred = gb_pipeline.predict(X_train)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

r2_gap = train_r2 - r2
rmse_gap = train_rmse - rmse

mlflow.sklearn.log_model(
    gb_pipeline,
    name="GradientBoostingRegressor",
    input_example=X_test.sample()
)
# mlflow.set_tracking_uri("databricks")
# mlflow.set_experment("Users/davidluismora@gmail.com/ÁrbolDecisión")


with mlflow.start_run(run_name=run_name):
    mlflow.log_param("modelo", "GradientBoostingRegressor")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    # métricas test
    mlflow.log_metric("test_mae", float(mae))
    mlflow.log_metric("test_mse", float(mse))
    mlflow.log_metric("test_rmse", float(rmse))
    mlflow.log_metric("test_r2", float(r2))

    # métricas train
    mlflow.log_metric("train_mae", float(train_mae))
    mlflow.log_metric("train_rmse", float(train_rmse))
    mlflow.log_metric("train_r2", float(train_r2))

    # overfitting
    mlflow.log_metric("r2_gap", float(r2_gap))
    mlflow.log_metric("rmse_gap", float(rmse_gap))

    # guardar modelo
    print(
        run_name,
        f"RMSE={rmse:.4f}",
        f"R2={r2:.4f}",
        f"gap={r2_gap:.4f}"
    )


