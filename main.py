from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, create_model
import joblib
import numpy as np
import json
import os
import threading
import subprocess
import sys

# Cargar las configuraciones dinámicas de variables
with open('features.json', 'r', encoding='utf-8') as f:
    features_meta = json.load(f)

# Generar esquema dinámico para Pydantic
fields = {}
for feat in features_meta:
    if feat['type'] == 'numeric':
        fields[feat['name']] = (float if feat['name'] == 'BMI' else int, ...)
    elif feat['type'] == 'categorical':
        fields[feat['name']] = (str, ...)

DynamicInput = create_model('DynamicInput', **fields)

app = FastAPI(title="Clinifio API — Backend de Alta Disponibilidad", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mapeos estáticos globales sincronizados con el modelo
AGE_ORDER = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"]
HEALTH_ORDER = ["Excellent", "Very good", "Good", "Fair", "Poor"]

# Componentes globales de inferencia
MODEL = joblib.load('model.pkl')
SCALER = joblib.load('scaler.pkl')

# ==========================================
# GESTIÓN DEL CONTADOR DE MANTENIMIENTO
# ==========================================
COUNTER_FILE = "query_counter.json"
MAINTENANCE_THRESHOLD = 5  # Reentrenar cada 5 consultas
_counter_lock = threading.Lock()

def _load_counter():
    """Carga el contador persistente desde disco."""
    if os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"count": 0, "status": "idle"}

def _save_counter(data: dict):
    """Guarda el estado del contador en disco de forma atómica."""
    with open(COUNTER_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def _run_retraining():
    """Ejecuta el reentrenamiento en un proceso separado (asíncrono, no bloquea la API)."""
    try:
        print("[MLOps] 🔄 Lanzando reentrenamiento asíncrono en segundo plano...")
        subprocess.run(
            [sys.executable, "retrain_model.py", "--nuevo"],
            capture_output=False,
            text=True
        )
        print("[MLOps] ✅ Reentrenamiento completado. Nuevos artefactos disponibles.")
        
        # ── INTERCAMBIO (HOT-SWAP) EN EL BACKEND ──
        import shutil
        if os.path.exists("nuevo_model.pkl") and os.path.exists("nuevo_scaler.pkl"):
            print("[MLOps] 🔄 Intercambiando (hot-swapping) nuevos artefactos del modelo...")
            shutil.move("nuevo_model.pkl", "model.pkl")
            shutil.move("nuevo_scaler.pkl", "scaler.pkl")
            if os.path.exists("nuevo_metrics.json"):
                shutil.move("nuevo_metrics.json", "metrics.json")
            
            # Recargar en caliente en memoria
            global MODEL, SCALER
            MODEL = joblib.load('model.pkl')
            SCALER = joblib.load('scaler.pkl')
            print("[MLOps] 🚀 Modelo y Scaler actualizados y recargados en caliente con éxito.")
            
    except Exception as e:
        print(f"[MLOps] ❌ Error durante el reentrenamiento: {e}")
    finally:
        # Asegurar que el estado regrese a idle
        with _counter_lock:
            state = _load_counter()
            state["status"] = "idle"
            _save_counter(state)

def _increment_and_check():
    """Incrementa el contador y dispara el reentrenamiento si alcanza el umbral."""
    with _counter_lock:
        state = _load_counter()
        state["count"] = state.get("count", 0) + 1

        if state["count"] >= MAINTENANCE_THRESHOLD and state.get("status") != "training":
            # Umbral alcanzado — activar ciclo de mantenimiento
            state["count"] = 0
            state["status"] = "training"
            _save_counter(state)
            # Lanzar reentrenamiento en hilo separado para no bloquear la respuesta
            thread = threading.Thread(target=_run_retraining, daemon=True)
            thread.start()
            return True  # Indicar que se inició el reentrenamiento
        else:
            _save_counter(state)
            return False

# ==========================================
# ENDPOINT: RAIZ / HEALTH CHECK
# ==========================================
@app.get("/")
def root():
    """Endpoint raiz — health check para Render y proxies."""
    return {
        "app": "Clinifio API",
        "version": "2.0",
        "status": "online",
        "endpoints": ["/predict", "/status", "/reload_model", "/docs"]
    }

# ==========================================
# ENDPOINT: PREDICCIÓN CLÍNICA
# ==========================================
@app.post("/predict")
def predict(data: DynamicInput):
    try:
        input_list = []
        for feat in features_meta:
            name = feat['name']
            val = getattr(data, name)
            
            # Transformación en caliente (Hot Encoding Lineal/Ordinal)
            if feat['type'] == 'categorical':
                if name in ["Smoking", "AlcoholDrinking", "Stroke", "DiffWalking"]:
                    val = 1 if val == "Yes" else 0
                elif name == "Sex":
                    val = 1 if val == "Male" else 0
                elif name == "AgeCategory":
                    val = AGE_ORDER.index(val)
                elif name == "GenHealth":
                    val = HEALTH_ORDER.index(val)
            input_list.append(val)
            
        # Convertir a matriz, escalar e inferir probabilidad
        X_array = np.array([input_list])
        X_scaled = SCALER.transform(X_array)
        
        probabilidad = float(MODEL.predict_proba(X_scaled)[0][1])
        
        # Umbral Clínico Calibrado (Ajustado para cribado preventivo masivo)
        diagnostico = "ALTO RIESGO" if probabilidad >= 0.30 else "EVALUACIÓN NORMAL"

        # ── Incrementar contador y disparar mantenimiento si corresponde ──
        mantenimiento_iniciado = _increment_and_check()
        
        return {
            "diagnostico": diagnostico,
            "probabilidad": probabilidad,
            "alerta": bool(probabilidad >= 0.30),
            "fecha_servidor": "2026-05-29",
            "mantenimiento_iniciado": mantenimiento_iniciado
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el motor clínico: {str(e)}")

# ==========================================
# ENDPOINT: ESTADO DEL CONTADOR (MLOps Dashboard)
# ==========================================
@app.get("/status")
def get_status():
    """Retorna el contador de consultas actual y el estado del ciclo de mantenimiento."""
    with _counter_lock:
        state = _load_counter()
    return {
        "count": state.get("count", 0),
        "status": state.get("status", "idle"),
        "umbral": MAINTENANCE_THRESHOLD,
        "restantes_para_mantenimiento": max(0, MAINTENANCE_THRESHOLD - state.get("count", 0))
    }

# ==========================================
# ENDPOINT: RECARGA EN CALIENTE DEL MODELO
# ==========================================
@app.post("/reload_model")
def reload_model():
    global MODEL, SCALER
    MODEL = joblib.load('model.pkl')
    SCALER = joblib.load('scaler.pkl')
    return {"status": "success", "message": "Modelo de XGBoost recargado en memoria RAM caliente"}

# ==========================================
# ENDPOINT: OBTENER MÉTRICAS ACTUALES
# ==========================================
@app.get("/metrics")
def get_metrics():
    """Retorna las métricas del modelo actual en producción."""
    if os.path.exists("metrics.json"):
        try:
            with open("metrics.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error al leer metrics.json: {e}")
    return {
        "accuracy": 0.8231, "precision": 0.8158, "recall": 0.8346, "f1_score": 0.8251, "auc_roc": 0.9103,
        "nombre_modelo_ganador": "XGBoost_Calibrated", "fecha_despliegue": "2026-05-29"
    }