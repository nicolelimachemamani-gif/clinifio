# 🏥 Clinifio — Sistema de Cribado Clínico con MLOps

Aplicación de inferencia de riesgo cardíaco con pipeline de **Mantenimiento Continuo (CD)** automatizado cada 5 consultas.

**Grupo:** Jhon Clinton, Garen y Fiorela

---

## 🏗️ Arquitectura

```
clinifio-api  (FastAPI)   →  Motor de inferencia + contador de mantenimiento
clinifio-app  (Streamlit) →  Interfaz clínica + dashboard MLOps
```

---

## 🚀 Despliegue en Render

### Paso 1 — Requisito: modelo pre-entrenado
Los archivos `model.pkl` y `scaler.pkl` deben estar en el repositorio para que el servidor de la API los cargue al arrancar.

### Paso 2 — Subir a GitHub
```bash
git add .
git commit -m "Deploy: Clinifio MLOps app"
git push origin main
```

### Paso 3 — Crear servicios en Render

#### 🔧 Servicio 1: FastAPI (API de Inferencia)
1. Ir a [render.com](https://render.com) → **New Web Service**
2. Conectar tu repositorio de GitHub
3. Configurar:
   - **Name:** `clinifio-api`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Copiar la URL generada (ej: `https://clinifio-api.onrender.com`)

#### 🖥️ Servicio 2: Streamlit (Frontend)
1. **New Web Service** → conectar el mismo repositorio
2. Configurar:
   - **Name:** `clinifio-app`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
3. Agregar variable de entorno:
   - **Key:** `API_URL`
   - **Value:** `https://clinifio-api.onrender.com` ← URL del paso anterior

---

## 💻 Ejecución Local

```bash
# Terminal 1: FastAPI
uvicorn main:app --reload

# Terminal 2: Streamlit
streamlit run app.py
```

---

## ⚙️ Pipeline MLOps — Mantenimiento Continuo

| Consultas | Acción |
|---|---|
| 1-4 | Predicción normal, contador incrementa |
| **5** | 🔄 Reentrenamiento automático en segundo plano |
| Tras reentrenar | Hot-swap del modelo sin apagar el servidor |

---

## 📁 Estructura del Proyecto

```
clinifio/
├── main.py              # FastAPI — Motor de inferencia + MLOps counter
├── app.py               # Streamlit — Frontend clínico
├── retrain_model.py     # Script de reentrenamiento (normal y --nuevo para CD)
├── features.json        # Metadatos dinámicos de variables clínicas
├── metrics.json         # Métricas del modelo en producción
├── query_counter.json   # Contador persistente de consultas (ciclo CD)
├── model.pkl            # Modelo XGBoost calibrado serializado
├── scaler.pkl           # StandardScaler serializado
├── requirements.txt     # Dependencias del proyecto
└── render.yaml          # Configuración de despliegue en Render
```
