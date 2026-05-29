# Clinifio — Sistema de Cribado Clinico con MLOps

Aplicacion de inferencia de riesgo cardiaco con pipeline de Mantenimiento Continuo (CD) automatizado cada 5 consultas.

**Grupo:** Jhon Clinton, Garen y Fiorela

---

## Arquitectura

```
clinifio-api  (FastAPI)   ->  Motor de inferencia + contador de mantenimiento
clinifio-app  (Streamlit) ->  Interfaz clinica + dashboard MLOps
```

---

## Despliegue en Render

### Paso 1 — Requisito: modelo pre-entrenado
Los archivos `model.pkl` y `scaler.pkl` deben estar en el repositorio para que el servidor de la API los cargue al arrancar.

### Paso 2 — Subir a GitHub
```bash
git add .
git commit -m "Deploy: Clinifio MLOps app"
git push origin main
```

### Paso 3 — Crear servicios en Render

#### Servicio 1: FastAPI (API de Inferencia)
1. Ir a [render.com](https://render.com) -> **New Web Service**
2. Conectar tu repositorio de GitHub
3. Configurar:
   - **Name:** `clinifio-api`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Copiar la URL generada (ej: `https://clinifio-api.onrender.com`)

#### Servicio 2: Streamlit (Frontend)
1. **New Web Service** -> conectar el mismo repositorio
2. Configurar:
   - **Name:** `clinifio-app`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
3. Agregar variable de entorno:
   - **Key:** `API_URL`
   - **Value:** `https://clinifio-api.onrender.com` <- URL del paso anterior

---

## Ejecucion Local

```bash
# Terminal 1: FastAPI
uvicorn main:app --reload

# Terminal 2: Streamlit
streamlit run app.py
```

---

## Pipeline MLOps — Mantenimiento Continuo

| Consultas | Accion |
|---|---|
| 1-4 | Prediccion normal, contador incrementa |
| 5 | Reentrenamiento automatico en segundo plano |
| Tras reentrenar | Hot-swap del modelo sin apagar el servidor |

---

## Estructura del Proyecto

```
clinifio/
├── main.py              # FastAPI — Motor de inferencia + MLOps counter
├── app.py               # Streamlit — Frontend clinico
├── retrain_model.py     # Script de reentrenamiento (normal y --nuevo para CD)
├── features.json        # Metadatos dinamicos de variables clinicas
├── metrics.json         # Metricas del modelo en produccion
├── query_counter.json   # Contador persistente de consultas (ciclo CD)
├── model.pkl            # Modelo XGBoost calibrado serializado
├── scaler.pkl           # StandardScaler serializado
├── requirements.txt     # Dependencias del proyecto
└── render.yaml          # Configuracion de despliegue en Render
```
