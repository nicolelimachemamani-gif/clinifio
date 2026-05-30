import streamlit as st
import requests
import json
import os
import shutil
import time

# ── URL del microservicio FastAPI ──────────────────────────────────────────────
# En producción (Render) se define la variable de entorno API_URL.
# En desarrollo local, apunta a localhost.
API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Clinifio", layout="centered")

# Estilos CSS Avanzados Minimalista Pastel
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&display=swap');

/* Global settings */
html, body, [class*="css"] { 
    font-family: 'Inter', sans-serif;
    color: #657166;
}

/* Background Gradient (stApp) */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #FDE8D3 0%, #DAEBE3 100%);
}

/* Sidebar Gradient */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.4);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255, 255, 255, 0.3);
}

/* Main Container / Glassmorphism Cards */
.main-title { 
    font-family: 'Poppins', sans-serif;
    font-size: 3em; 
    font-weight: 700; 
    color: #99CDD8; 
    text-align: center;
    margin-bottom: 25px;
    letter-spacing: -1px;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.05);
}

.alert-box { 
    padding: 16px 20px; 
    border-radius: 12px; 
    margin: 20px 0; 
    font-weight: 400; 
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.03);
    backdrop-filter: blur(5px);
}
.alert-danger  { background-color: rgba(243, 195, 178, 0.8); color: #657166; border: 1px solid rgba(255,255,255,0.5); }
.alert-success { background-color: rgba(207, 214, 196, 0.8); color: #657166; border: 1px solid rgba(255,255,255,0.5); }
.alert-info    { background-color: rgba(153, 205, 216, 0.8); color: #657166; border: 1px solid rgba(255,255,255,0.5); }
.alert-warning { background-color: rgba(253, 232, 211, 0.8); color: #657166; border: 1px solid rgba(255,255,255,0.5); }

/* Buttons */
.stButton > button {
    background-color: #99CDD8;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 10px rgba(153, 205, 216, 0.3);
}
.stButton > button:hover {
    background-color: #CFD6C4;
    box-shadow: 0 6px 15px rgba(207, 214, 196, 0.4);
    transform: translateY(-1px);
    color: #657166;
}

/* Inputs and Selectboxes */
.stTextInput > div > div > input, .stSelectbox > div > div > div {
    border-radius: 8px;
    border: 1px solid rgba(101, 113, 102, 0.2);
    background-color: rgba(255, 255, 255, 0.7);
}

/* Cycle Bar */
.cycle-bar-bg {
    background: rgba(255, 255, 255, 0.5);
    border-radius: 8px;
    height: 8px;
    width: 100%;
    margin-top: 8px;
    overflow: hidden;
}
.cycle-bar-fill {
    height: 8px;
    border-radius: 8px;
    background: #99CDD8;
    transition: width 0.4s ease;
}
.cycle-bar-fill-training {
    height: 8px;
    border-radius: 8px;
    background: #F3C3B2;
    animation: pulse-bar 1.5s infinite alternate;
}
@keyframes pulse-bar {
    from { opacity: 0.6; }
    to   { opacity: 1.0; }
}

.badge-training {
    display: inline-block;
    background: #F3C3B2;
    color: #657166;
    border-radius: 12px;
    padding: 4px 12px;
    font-size: 0.82em;
    font-weight: 600;
    margin-left: 6px;
    box-shadow: 0 2px 5px rgba(243, 195, 178, 0.4);
}
.badge-idle {
    display: inline-block;
    background: #CFD6C4;
    color: #657166;
    border-radius: 12px;
    padding: 4px 12px;
    font-size: 0.82em;
    font-weight: 600;
    box-shadow: 0 2px 5px rgba(207, 214, 196, 0.4);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Clinifio 2.0</div>", unsafe_allow_html=True)

# Carga dinámica de metadatos
with open('features.json', 'r', encoding='utf-8') as f:
    features_meta = json.load(f)

# Carga de métricas desde el microservicio FastAPI o archivo local como fallback
metrics = {
    "accuracy": 0.8231, "precision": 0.8158, "recall": 0.8346, "f1_score": 0.8251, "auc_roc": 0.9103,
    "nombre_modelo_ganador": "XGBoost_Calibrated", "fecha_despliegue": "2026-05-29"
}
try:
    metrics_res = requests.get(f"{API_URL}/metrics", timeout=5)
    if metrics_res.status_code == 200:
        metrics = metrics_res.json()
except Exception as e:
    if os.path.exists('metrics.json'):
        try:
            with open('metrics.json', 'r', encoding='utf-8') as f:
                metrics = json.load(f)
        except Exception:
            pass

# ─────────────────────────────────────────────
# PANEL LATERAL: Ciclo de Mantenimiento Continuo
# ─────────────────────────────────────────────
st.sidebar.markdown("#### Mantenimiento Continuo")

try:
    status_res = requests.get(f"{API_URL}/status", timeout=30)
    if status_res.status_code == 200:
        status_data = status_res.json()
        count     = status_data.get("count", 0)
        umbral    = status_data.get("umbral", 5)
        estado    = status_data.get("status", "idle")
        restantes = status_data.get("restantes_para_mantenimiento", umbral)

        # Detectar transición de "training" a "idle" para notificar éxito en UI
        if "prev_estado" not in st.session_state:
            st.session_state["prev_estado"] = estado
        
        if st.session_state["prev_estado"] == "training" and estado == "idle":
            st.sidebar.success("🎉 ¡Modelo reentrenado con éxito!")
            st.toast("🚀 ¡El pipeline MLOps ha actualizado el modelo en caliente!")
            
        st.session_state["prev_estado"] = estado

        pct = int((count / umbral) * 100)

        if estado == "training":
            st.sidebar.markdown(
                f"**Estado:** <span class='badge-training'>Reentrenando...</span>",
                unsafe_allow_html=True
            )
            st.sidebar.markdown(
                "<div class='cycle-bar-bg'><div class='cycle-bar-fill-training' style='width:100%'></div></div>",
                unsafe_allow_html=True
            )
            st.sidebar.markdown(
                "<div class='alert-box alert-warning'>Reentrenamiento automático en progreso. El nuevo modelo se cargará en caliente al finalizar.</div>",
                unsafe_allow_html=True
            )
        else:
            st.sidebar.markdown(
                f"**Estado:** <span class='badge-idle'>Activo</span>",
                unsafe_allow_html=True
            )
            st.sidebar.markdown(
                f"<div class='cycle-bar-bg'><div class='cycle-bar-fill' style='width:{pct}%'></div></div>",
                unsafe_allow_html=True
            )

        st.sidebar.metric(
            "Consultas en ciclo actual",
            f"{count} / {umbral}",
            delta=f"Faltan {restantes} para mantenimiento" if estado != "training" else "Reentrenando…"
        )

        # ── EXPOSITOR DE MÉTRICAS DEL MODELO ACTIVO ──
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### Métricas del Modelo Activo")
        st.sidebar.markdown(f"**Ganador:** `{metrics.get('nombre_modelo_ganador', 'XGBoost_Calibrated')}`")
        st.sidebar.markdown(f"**F1-Score:** `{metrics.get('f1_score', 0.8251):.4f}`")
        st.sidebar.markdown(f"**Exactitud:** `{metrics.get('accuracy', 0.8231):.4f}`")
        st.sidebar.markdown(f"**Precisión:** `{metrics.get('precision', 0.8158):.4f}`")
        st.sidebar.markdown(f"**Sensibilidad:** `{metrics.get('recall', 0.8346):.4f}`")
        st.sidebar.markdown(f"**AUC-ROC:** `{metrics.get('auc_roc', 0.9103):.4f}`")
        st.sidebar.caption(f"Último despliegue: {metrics.get('fecha_despliegue', '2026-05-29')}")
    else:
        st.sidebar.warning(f"API inaccesible. Status code: {status_res.status_code}")
except Exception as e:
    st.sidebar.warning(f"⚠️ Sin conexión al API en: {API_URL} (Detalle: {e})")

# ─────────────────────────────────────────────
# FORMULARIO CLÍNICO
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("#### Ficha del Paciente")

user_inputs = {}

# Diccionarios de traducción visual
tr_opciones = {
    "Yes": "Sí", "No": "No", "Male": "Masculino", "Female": "Femenino",
    "Excellent": "Excelente", "Very good": "Muy Bueno", "Good": "Bueno",
    "Fair": "Regular", "Poor": "Malo", "80 or older": "80 años o más"
}

tr_labels = {
    "Smoking": "¿Fuma actualmente?",
    "AlcoholDrinking": "¿Consume alcohol frecuente?",
    "Stroke": "¿Ha tenido un ACV (Derrame)?",
    "PhysicalHealth": "Días de mala salud física",
    "MentalHealth": "Días de mala salud mental",
    "DiffWalking": "¿Dificultad para caminar?",
    "Sex": "Género del paciente",
    "AgeCategory": "Rango de Edad",
    "GenHealth": "Estado general de salud",
    "BMI": "Índice de Masa Corporal (BMI)"
}

with st.form("clinical_form"):
    col1, col2 = st.columns(2)
    for idx, feat in enumerate(features_meta):
        target_col = col1 if idx % 2 == 0 else col2
        name = feat['name']
        label_es = tr_labels.get(name, name)
        
        if feat['type'] == 'numeric':
            if name == 'BMI':
                user_inputs[name] = target_col.slider(label_es, float(feat['min']), float(feat['max']), float(feat['default']), 0.1)
            else:
                user_inputs[name] = target_col.slider(label_es, int(feat['min']), int(feat['max']), int(feat['default']))
        elif feat['type'] == 'categorical':
            user_inputs[name] = target_col.selectbox(
                label_es, 
                options=feat['options'], 
                index=feat['options'].index(feat['default']),
                format_func=lambda x: tr_opciones.get(x, x.replace("-", " a ") + " años" if "-" in x else x)
            )

    submit_btn = st.form_submit_button("Realizar Predicción")

# ─────────────────────────────────────────────
# PROCESAMIENTO DE RESPUESTA FASTAPI
# ─────────────────────────────────────────────
if submit_btn:
    try:
        res = requests.post(f"{API_URL}/predict", json=user_inputs, timeout=30)
        if res.status_code == 200:
            resultado = res.json()
            prob = resultado['probabilidad']
            
            st.markdown("### Resultado del Diagnóstico")
            st.write(f"**Probabilidad de riesgo:** `{prob * 100:.2f}%`")
            st.progress(int(prob * 100))
            
            if resultado['alerta']:
                st.markdown(
                    "<div class='alert-box alert-danger'>ALTO RIESGO: El paciente requiere priorización diagnóstica.</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div class='alert-box alert-success'>EVALUACIÓN NORMAL: El paciente se encuentra dentro de los parámetros aceptables.</div>",
                    unsafe_allow_html=True
                )

            # Notificación si esta consulta disparó el mantenimiento automático
            if resultado.get("mantenimiento_iniciado"):
                st.markdown(
                    "<div class='alert-box alert-info'>Ciclo de mantenimiento iniciado (5 consultas). El modelo se actualizará en segundo plano.</div>",
                    unsafe_allow_html=True
                )
                # Rerun para actualizar el panel lateral inmediatamente
                time.sleep(0.5)
                st.rerun()
        else:
            st.error(f"Error {res.status_code} en la API: {res.text}")
    except Exception as e:
        st.error(f"El microservicio en {API_URL} no responde. Detalle: {e}")

# ─────────────────────────────────────────────
# PIPELINE DE MANTENIMIENTO CONTINUO (CD) — HOT-SWAP
# ─────────────────────────────────────────────
def check_maintenance_pipeline():
    """
    Sincronización del ciclo de mantenimiento. El backend gestiona el hot-swap
    de manera interna y atómica para evitar conflictos multihilo y de contenedores.
    """
    pass

check_maintenance_pipeline()