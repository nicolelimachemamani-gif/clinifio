import streamlit as st
import requests
import json
import os
import shutil
import time

# ── URL del microservicio FastAPI ──────────────────────────────────────────────
# En producción (Render) se define la variable de entorno API_URL.
# En desarrollo local, apunta a localhost.
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Clinifio — Inferencia de Riesgo", page_icon="🩺", layout="centered")

# Estilos CSS Avanzados para entorno clínico
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-title { font-size:2.4em; font-weight:700; color:#1a5276; text-align:center; }

.alert-box { padding: 15px; border-radius: 8px; margin: 15px 0; font-weight: bold; text-align: center; }
.alert-danger  { background-color: #fadbd8; color: #78281f; border: 1px solid #f5b7b1; font-size:1.2em; }
.alert-success { background-color: #d4efdf; color: #145a32; border: 1px solid #abebc6; font-size:1.2em; }
.alert-info    { background-color: #d6eaf8; color: #1a5276; border: 1px solid #7fb3d3; font-size:1.0em; }
.alert-warning { background-color: #fef9e7; color: #7d6608; border: 1px solid #f7dc6f; font-size:1.0em; }

/* Barra de progreso del ciclo de mantenimiento */
.cycle-bar-bg {
    background: #e8ecef;
    border-radius: 8px;
    height: 12px;
    width: 100%;
    margin-top: 4px;
    overflow: hidden;
}
.cycle-bar-fill {
    height: 12px;
    border-radius: 8px;
    background: linear-gradient(90deg, #2ecc71, #1abc9c);
    transition: width 0.4s ease;
}
.cycle-bar-fill-training {
    height: 12px;
    border-radius: 8px;
    background: linear-gradient(90deg, #e67e22, #e74c3c);
    animation: pulse-bar 1.5s infinite alternate;
}
@keyframes pulse-bar {
    from { opacity: 0.7; }
    to   { opacity: 1.0; }
}

.badge-training {
    display: inline-block;
    background: linear-gradient(90deg, #e67e22, #e74c3c);
    color: white;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.82em;
    font-weight: 600;
    animation: pulse-badge 1.2s infinite alternate;
    margin-left: 6px;
}
@keyframes pulse-badge {
    from { box-shadow: 0 0 4px #e67e22; }
    to   { box-shadow: 0 0 14px #e74c3c; }
}

.badge-idle {
    display: inline-block;
    background: linear-gradient(90deg, #27ae60, #2ecc71);
    color: white;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.82em;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🏥 Sistema de Cribado Clínico Clinifio</div>", unsafe_allow_html=True)
st.write("Aplicación Web Médica Conectada en Localhost. Grupo: **Jhon Clinton, Garen y Fiorela**.")

# Carga dinámica de metadatos
with open('features.json', 'r', encoding='utf-8') as f:
    features_meta = json.load(f)

with open('metrics.json', 'r', encoding='utf-8') as f:
    metrics = json.load(f)

# ─────────────────────────────────────────────
# PANEL LATERAL: Tablero de Control MLOps
# ─────────────────────────────────────────────
st.sidebar.markdown("### 📊 Tablero de Control TI (MLOps)")
st.sidebar.metric("Modelo Operativo",        metrics.get("nombre_modelo_ganador", "XGBoost"))
st.sidebar.metric("Área bajo la curva (AUC)", f"{metrics.get('auc_roc', 0.85)*100:.2f}%")
st.sidebar.metric("Sensibilidad (Recall)",   f"{metrics.get('recall', 0.80)*100:.2f}%")

# Obtener estado del ciclo de mantenimiento desde el API
st.sidebar.markdown("---")
st.sidebar.markdown("#### 🔧 Ciclo de Mantenimiento Continuo")

try:
    status_res = requests.get(f"{API_URL}/status", timeout=3)
    if status_res.status_code == 200:
        status_data = status_res.json()
        count     = status_data.get("count", 0)
        umbral    = status_data.get("umbral", 5)
        estado    = status_data.get("status", "idle")
        restantes = status_data.get("restantes_para_mantenimiento", umbral)

        pct = int((count / umbral) * 100)

        if estado == "training":
            st.sidebar.markdown(
                f"**Estado:** <span class='badge-training'>🔄 Reentrenando…</span>",
                unsafe_allow_html=True
            )
            st.sidebar.markdown(
                "<div class='cycle-bar-bg'><div class='cycle-bar-fill-training' style='width:100%'></div></div>",
                unsafe_allow_html=True
            )
            st.sidebar.markdown(
                "<div class='alert-box alert-warning'>⚙️ Reentrenamiento automático en progreso. El nuevo modelo se cargará en caliente al finalizar.</div>",
                unsafe_allow_html=True
            )
        else:
            st.sidebar.markdown(
                f"**Estado:** <span class='badge-idle'>✅ Activo</span>",
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
    else:
        st.sidebar.warning("API inaccesible — estado no disponible.")
except Exception:
    st.sidebar.warning("⚠️ Sin conexión al API. Inicia FastAPI primero.")

# ─────────────────────────────────────────────
# FORMULARIO CLÍNICO
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 🩺 Ficha Epidemiológica Digital del Paciente")

user_inputs = {}
with st.form("clinical_form"):
    col1, col2 = st.columns(2)
    for idx, feat in enumerate(features_meta):
        target_col = col1 if idx % 2 == 0 else col2
        name = feat['name']
        
        if feat['type'] == 'numeric':
            if name == 'BMI':
                user_inputs[name] = target_col.slider("Índice de Masa Corporal (BMI)", float(feat['min']), float(feat['max']), float(feat['default']), 0.1)
            else:
                user_inputs[name] = target_col.slider(f"Días con afección: {name}", int(feat['min']), int(feat['max']), int(feat['default']))
        elif feat['type'] == 'categorical':
            user_inputs[name] = target_col.selectbox(f"Indicador: {name}", options=feat['options'], index=feat['options'].index(feat['default']))

    submit_btn = st.form_submit_button("🔴 ACCIONAR PREDICCIÓN CLÍNICA")

# ─────────────────────────────────────────────
# PROCESAMIENTO DE RESPUESTA FASTAPI
# ─────────────────────────────────────────────
if submit_btn:
    try:
        res = requests.post(f"{API_URL}/predict", json=user_inputs)
        if res.status_code == 200:
            resultado = res.json()
            prob = resultado['probabilidad']
            
            st.markdown("### 🧬 Resultado del Diagnóstico de Inferencia")
            st.write(f"**Probabilidad de riesgo calculada:** `{prob * 100:.2f}%` (Prevalencia ajustada)")
            st.progress(int(prob * 100))
            
            if resultado['alerta']:
                st.markdown(
                    "<div class='alert-box alert-danger'>🚨 ALTO RIESGO CORONARIO: El paciente requiere priorización diagnóstica inmediata debido a la presencia de multiplicadores de riesgo severos.</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div class='alert-box alert-success'>✅ EVALUACIÓN NORMAL: El paciente se encuentra dentro de los parámetros de riesgo aceptables para la ventana epidemiológica base.</div>",
                    unsafe_allow_html=True
                )

            # Notificación si esta consulta disparó el mantenimiento automático
            if resultado.get("mantenimiento_iniciado"):
                st.markdown(
                    "<div class='alert-box alert-info'>⚙️ ¡5 consultas completadas! El pipeline de <strong>Mantenimiento Continuo (CD)</strong> se ha activado automáticamente. El modelo se reentrenará en segundo plano.</div>",
                    unsafe_allow_html=True
                )
                # Rerun para actualizar el panel lateral inmediatamente
                time.sleep(0.5)
                st.rerun()
        else:
            st.error("Error en la predicción. Verifica que el servidor de FastAPI esté respondiendo correctamente.")
    except Exception as e:
        st.error(f"El microservicio de FastAPI está apagado o inaccesible. Asegúrate de ejecutar uvicorn primero. Detalle: {e}")

# ─────────────────────────────────────────────
# PIPELINE DE MANTENIMIENTO CONTINUO (CD) — HOT-SWAP
# ─────────────────────────────────────────────
def check_maintenance_pipeline():
    """
    Detecta nuevos artefactos generados por el reentrenamiento asíncrono
    y los reemplaza en caliente de forma segura y atómica.
    Reemplaza: nuevo_model.pkl → model.pkl
               nuevo_scaler.pkl → scaler.pkl
               nuevo_metrics.json → metrics.json
    """
    nuevo_model   = 'nuevo_model.pkl'
    nuevo_scaler  = 'nuevo_scaler.pkl'
    nuevo_metrics = 'nuevo_metrics.json'

    if os.path.exists(nuevo_model) and os.path.exists(nuevo_scaler):
        st.markdown(
            "<div class='alert-box alert-info'>🆕 [Pipeline CD] ¡Nuevo modelo detectado! Reemplazando artefactos en caliente…</div>",
            unsafe_allow_html=True
        )
        # Reemplazar artefactos de forma segura
        shutil.move(nuevo_model,   'model.pkl')
        shutil.move(nuevo_scaler,  'scaler.pkl')
        if os.path.exists(nuevo_metrics):
            shutil.move(nuevo_metrics, 'metrics.json')

        # Recargar el modelo en el microservicio FastAPI en caliente
        try:
            r = requests.post(f"{API_URL}/reload_model", timeout=5)
            if r.status_code == 200:
                st.success("✅ [Pipeline CD] Microservicio actualizado. Nuevo modelo en producción en caliente.")
            else:
                st.warning("⚠️ El modelo fue reemplazado pero el API no respondió al reload. Reinicia FastAPI.")
        except Exception as ex:
            st.warning(f"⚠️ Artefactos copiados, pero no se pudo notificar al API: {ex}")

        # Forzar recarga de la página para mostrar las nuevas métricas
        time.sleep(1)
        st.rerun()

check_maintenance_pipeline()