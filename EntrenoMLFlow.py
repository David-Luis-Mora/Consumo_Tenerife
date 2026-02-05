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

# Cojo datos y los divido para entrenar.
data = pd.read_csv('consumo-energetico-2025.csv')

data['fecha'] = pd.to_datetime(data['fecha'])

features = ['dia', 'mes','cups_municipio', 'cups_distribuidor']
data['dia'] = data['fecha'].dt.day
data['mes'] = data['fecha'].dt.month
data['consumo_log'] = np.log1p(data['consumo'])
data.drop(columns=['fecha'],inplace=True)

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
mlflow.set_experiment("Cáncer mama - Árbol de decisión")

# Al igual que vimos con el CVGridSearch, pongo unos cuántos valores de hiperparámetros.
max_depths = [None, 2]
min_splits = [2, 5]

model_n_estimators =  [50, 100],
model_max_depth = [None, 10, 20],
model_min_samples_split = [2, 5]

gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(random_state=42))
])

# Ahora simulo un poco lo que hace dicha clase.
# Lanzo todas las combinaciones...
for max_depth in model_n_estimators:
    for min_samples_split in model_min_samples_split:

                # ... y las logueo.
                run_name = (
                    f"Lanzamiento de árbol de decisión, con profundidad_hoja={max_depth} y "
                    f" split={min_samples_split}"
                )

                # Iniciamos experimento
                with mlflow.start_run(run_name=run_name):

                    model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )

                    model.fit(X_train, y_train)

                    # Aquí predigo
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]

                    # Saco las métricas
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred)
                    rec = recall_score(y_test, y_pred)
                    auc = roc_auc_score(y_test, y_proba)

                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                    # Miro a ver si hay sobreajuste
                    train_acc = accuracy_score(y_train, model.predict(X_train))
                    acc_gap = train_acc - acc

                    # Logueo estos parámetros
                    mlflow.log_param("modelo", "Árbol de decisión")
                    mlflow.log_param("max_depth", str(max_depth)) # así evito que pete, porque puede ser None
                    mlflow.log_param("min_samples_split", min_samples_split)
                    mlflow.log_param("test_size", 0.2)
                    mlflow.log_param("random_state", 42)

                    # Logueo las métricas
                    mlflow.log_metric("test_accuracy", float(acc))
                    mlflow.log_metric("test_f1", float(f1))
                    mlflow.log_metric("test_precision", float(prec))
                    mlflow.log_metric("test_recall", float(rec))
                    mlflow.log_metric("test_roc_auc", float(auc))

                    # Guardo la matriz de confusión
                    mlflow.log_metric("tn", float(tn))
                    mlflow.log_metric("fp", float(fp))
                    mlflow.log_metric("fn", float(fn))
                    mlflow.log_metric("tp", float(tp))

                    # Miro el sobreajuste (error entreno - error validación)
                    mlflow.log_metric("train_accuracy", float(train_acc))
                    mlflow.log_metric("accuracy_gap", float(acc_gap))

                    # Se guarda el modelo como artefacto (.pkl + metadatos...)
                    # Aquí NO LO REGISTRO. Lo podría registrar con registered_model_name
                    mlflow.sklearn.log_model(model, name="modelo", input_example=X_test.sample())

                    print(
                        run_name,
                        f"test_acc={acc:.4f}",
                        f"auc={auc:.4f}",
                        f"gap={acc_gap:.4f}"
                    )

