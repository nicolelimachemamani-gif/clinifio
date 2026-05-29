import pandas as pd
import numpy as np
import json
import joblib
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier

# ==========================================
# MODO DE EJECUCIÓN: --nuevo para CD pipeline
# ==========================================
# Si se pasa el argumento --nuevo, los artefactos se guardan
# con prefijo "nuevo_" para que el hot-swap sea seguro y atómico.
NUEVO_MODE = "--nuevo" in sys.argv
COUNTER_FILE = "query_counter.json"

if NUEVO_MODE:
    print("⚙️  [MLOps] Modo CD activado: guardando como 'nuevo_model.pkl' (hot-swap seguro)")
    MODEL_OUT   = "nuevo_model.pkl"
    SCALER_OUT  = "nuevo_scaler.pkl"
    METRICS_OUT = "nuevo_metrics.json"
else:
    print("⚙️  [MLOps] Modo entrenamiento inicial: guardando artefactos de producción.")
    MODEL_OUT   = "model.pkl"
    SCALER_OUT  = "scaler.pkl"
    METRICS_OUT = "metrics.json"

# ==========================================
# CONFIGURACIÓN Y CONSTANTES DEL ENTORNO
# ==========================================
DATASET_PATH = "heart_2020_cleaned.csv"
TARGET = "HeartDisease"

FEATURES = [
    "BMI", "Smoking", "AlcoholDrinking", "Stroke", 
    "PhysicalHealth", "MentalHealth", "DiffWalking", 
    "Sex", "AgeCategory", "GenHealth"
]

AGE_ORDER = [
    "18-24", "25-29", "30-34", "35-39", "40-44", "45-49", 
    "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"
]
HEALTH_ORDER = ["Excellent", "Very good", "Good", "Fair", "Poor"]

# ==========================================
# PIPELINE DE PREPROCESAMIENTO CLÍNICO
# ==========================================
def preprocess_clinical_data(path_csv):
    print("⏳ Cargando y limpiando conjunto de datos tabular masivo...")
    df = pd.read_csv(path_csv)
    df = df.dropna(subset=[TARGET] + FEATURES)
    df_proc = df.copy()
    
    # Mapeo binario estándar (Yes=1, No=0)
    binary_cols = [TARGET, "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking"]
    for col in binary_cols:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].map({"Yes": 1, "No": 0})
            
    # Mapeo de sexo biológico
    df_proc["Sex"] = df_proc["Sex"].map({"Male": 1, "Female": 0})
    
    # Mapeos ordinales explícitos exigidos
    df_proc["AgeCategory"] = df_proc["AgeCategory"].apply(
        lambda x: AGE_ORDER.index(x) if x in AGE_ORDER else 0
    )
    df_proc["GenHealth"] = df_proc["GenHealth"].apply(
        lambda x: HEALTH_ORDER.index(x) if x in HEALTH_ORDER else 2
    )
    
    return df_proc

# Ejecutar proceso
df_clean = preprocess_clinical_data(DATASET_PATH)
X = df_clean[FEATURES]
y = df_clean[TARGET]

# ==========================================
# SPLIT DE DATOS Y ESCALADO ROBUSTO
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# ENTRENAMIENTO DE ESTIMADOR ALTO POTENCIAL
# ==========================================
print("🚀 Entrenando ensamble potente XGBoost...")
base_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

# Calibración Isotónica para ajustar probabilidades reales para perfiles críticos
print("⚖️ Calibrando probabilidades clínicas (Isotonic CV)...")
calibrated_model = CalibratedClassifierCV(estimator=base_model, method="isotonic", cv=3)
calibrated_model.fit(X_train_scaled, y_train)

# ==========================================
# EVALUACIÓN DE MÉTRICAS COMPLETA
# ==========================================
y_pred = calibrated_model.predict(X_test_scaled)
y_prob = calibrated_model.predict_proba(X_test_scaled)[:, 1]

metrics_report = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred)),
    "recall": float(recall_score(y_test, y_pred)),
    "f1_score": float(f1_score(y_test, y_pred)),
    "auc_roc": float(roc_auc_score(y_test, y_prob)),
    "nombre_modelo_ganador": "XGBoost_Calibrated",
    "fecha_despliegue": "2026-05-29"
}

print("\n📊 REPORTE TÉCNICO PARA EL JURADO:")
print(json.dumps(metrics_report, indent=4))
print("\n🧱 MATRIZ DE CONFUSIÓN:")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# SERIALIZACIÓN Y EXPORTACIÓN DE ARTEFACTOS
# ==========================================
joblib.dump(calibrated_model, MODEL_OUT)
joblib.dump(scaler, SCALER_OUT)

# Solo actualizar features.json y métricas en modo inicial (no CD)
if not NUEVO_MODE:
    features_metadata = [
        {"name": "BMI", "type": "numeric", "min": 10, "max": 60, "default": 25},
        {"name": "Smoking", "type": "categorical", "options": ["Yes", "No"], "default": "No"},
        {"name": "AlcoholDrinking", "type": "categorical", "options": ["Yes", "No"], "default": "No"},
        {"name": "Stroke", "type": "categorical", "options": ["Yes", "No"], "default": "No"},
        {"name": "PhysicalHealth", "type": "numeric", "min": 0, "max": 30, "default": 0},
        {"name": "MentalHealth", "type": "numeric", "min": 0, "max": 30, "default": 0},
        {"name": "DiffWalking", "type": "categorical", "options": ["Yes", "No"], "default": "No"},
        {"name": "Sex", "type": "categorical", "options": ["Male", "Female"], "default": "Female"},
        {"name": "AgeCategory", "type": "categorical", "options": AGE_ORDER, "default": "40-44"},
        {"name": "GenHealth", "type": "categorical", "options": HEALTH_ORDER, "default": "Good"}
    ]
    with open("features.json", "w", encoding="utf-8") as f:
        json.dump(features_metadata, f, indent=4)

# Guardar métricas (siempre, ya sea como metrics.json o nuevo_metrics.json)
with open(METRICS_OUT, "w", encoding="utf-8") as f:
    json.dump(metrics_report, f, indent=4)

# ==========================================
# ACTUALIZAR ESTADO DEL CONTADOR (MODO CD)
# ==========================================
if NUEVO_MODE:
    # Marcar el reentrenamiento como completado en el contador
    if os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE, 'r', encoding='utf-8') as f:
            state = json.load(f)
        state["status"] = "idle"
        with open(COUNTER_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4)
        print("\n✅ [MLOps] Estado del contador actualizado a 'idle'. Ciclo de mantenimiento completado.")

print(f"\n✅ [MLOps] Artefactos guardados: '{MODEL_OUT}', '{SCALER_OUT}', '{METRICS_OUT}'.")
print("✅ [MLOps] Listos para hot-swap en FastAPI y Streamlit.")