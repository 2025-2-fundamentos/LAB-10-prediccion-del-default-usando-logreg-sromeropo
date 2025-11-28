# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import os
import gzip
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix
)


# =====================================================================
# CARGA Y LIMPIEZA DE DATOS
# =====================================================================

def load_dataset(path: str) -> pd.DataFrame:
    """Carga un archivo CSV comprimido en ZIP."""
    return pd.read_csv(path, index_col=False, compression='zip')


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas, corrige EDUCATION y elimina valores inválidos."""
    cleaned = df.copy()
    cleaned = cleaned.rename(columns={'default payment next month': 'default'})
    cleaned = cleaned.drop(columns=['ID'])

    # Eliminar valores inválidos
    cleaned = cleaned.loc[cleaned["MARRIAGE"] != 0]
    cleaned = cleaned.loc[cleaned["EDUCATION"] != 0]

    # Colapsar valores mayores a 4
    cleaned["EDUCATION"] = cleaned["EDUCATION"].apply(lambda x: x if x < 4 else 4)

    return cleaned


# =====================================================================
# PIPELINE
# =====================================================================

def create_pipeline(x: pd.DataFrame) -> Pipeline:
    """Construye un pipeline con escalado, OHE, selección K-best y Regresión Logística."""

    categorical = ["SEX", "EDUCATION", "MARRIAGE"]
    numerical = list(set(x.columns) - set(categorical))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="passthrough"
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("k_best", SelectKBest(f_classif)),
            ("model", LogisticRegression(random_state=42))
        ]
    )


def create_estimator(pipeline: Pipeline, x: pd.DataFrame) -> GridSearchCV:
    """Optimiza K-best y C con GridSearchCV usando balanced_accuracy."""
    param_grid = {
        "k_best__k": range(1, len(x.columns) + 1),
        "model__C": np.logspace(-3, 3, 7),
    }

    return GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=2,
        refit=True,
    )


# =====================================================================
# MODELO Y MÉTRICAS
# =====================================================================

def save_model(path: str, estimator: GridSearchCV):
    """Guarda el modelo optimizado en formato gzip."""
    with gzip.open(path, "wb") as f:
        pickle.dump(estimator, f)


def calculate_precision_metrics(dataset_name: str, y_true, y_pred) -> dict:
    """Devuelve métricas principales del dataset."""
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def calculate_confusion_metrics(dataset_name: str, y_true, y_pred) -> dict:
    """Devuelve matriz de confusión en formato dict."""
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {
            "predicted_0": int(cm[0][0]),
            "predicted_1": int(cm[0][1])
        },
        "true_1": {
            "predicted_0": int(cm[1][0]),
            "predicted_1": int(cm[1][1])
        },
    }


# =====================================================================
# PROGRAMA PRINCIPAL
# =====================================================================

def main():
    input_dir = "./files/input"
    model_dir = "./files/models"

    # ---------------------------------------------------------
    # 1. Carga y limpieza
    # ---------------------------------------------------------
    train_df = load_dataset(os.path.join(input_dir, "train_data.csv.zip"))
    test_df = load_dataset(os.path.join(input_dir, "test_data.csv.zip"))

    train_df = clean_dataset(train_df)
    test_df = clean_dataset(test_df)

    # ---------------------------------------------------------
    # 2. División X / y
    # ---------------------------------------------------------
    X_train = train_df.drop(columns=["default"])
    y_train = train_df["default"]

    X_test = test_df.drop(columns=["default"])
    y_test = test_df["default"]

    # ---------------------------------------------------------
    # 3. Pipeline
    # ---------------------------------------------------------
    pipeline = create_pipeline(X_train)

    # ---------------------------------------------------------
    # 4. GridSearch
    # ---------------------------------------------------------
    estimator = create_estimator(pipeline, X_train)
    estimator.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 5. Guardar modelo
    # ---------------------------------------------------------
    os.makedirs(model_dir, exist_ok=True)
    save_model(
        os.path.join(model_dir, "model.pkl.gz"),
        estimator
    )

    # ---------------------------------------------------------
    # 6. Métricas de precisión
    # ---------------------------------------------------------
    y_train_pred = estimator.predict(X_train)
    y_test_pred = estimator.predict(X_test)

    train_metrics = calculate_precision_metrics("train", y_train, y_train_pred)
    test_metrics = calculate_precision_metrics("test", y_test, y_test_pred)

    # ---------------------------------------------------------
    # 7. Matriz de confusión
    # ---------------------------------------------------------
    train_cm = calculate_confusion_metrics("train", y_train, y_train_pred)
    test_cm = calculate_confusion_metrics("test", y_test, y_test_pred)

    # ---------------------------------------------------------
    # 8. Guardar salida
    # ---------------------------------------------------------
    os.makedirs("files/output", exist_ok=True)
    with open("files/output/metrics.json", "w") as f:
        for entry in [train_metrics, test_metrics, train_cm, test_cm]:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()

