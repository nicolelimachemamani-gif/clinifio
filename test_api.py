import os
import json
import pytest
import requests

# =========================
# Test 1: Artefactos existen
# =========================
def test_artifacts_exist():
    for fname in ['model.pkl', 'scaler.pkl', 'features.json', 'metrics.json']:
        assert os.path.exists(fname), f"Falta el archivo: {fname}"

# =========================
# Test 2: Features.json tiene 10 features
# =========================
def test_features_json():
    with open('features.json', 'r', encoding='utf-8') as f:
        features = json.load(f)
    assert isinstance(features, list), "features.json debe ser una lista"
    assert len(features) == 10, "features.json debe tener exactamente 10 características"

# =========================
# Test 3: Petición HTTP válida a /predict
# =========================
def test_predict_endpoint():
    with open('features.json', 'r', encoding='utf-8') as f:
        features = json.load(f)
    # Generar datos de prueba válidos
    data = {}
    for feature in features:
        if feature['type'] == 'numeric':
            data[feature['name']] = feature.get('default', feature.get('min', 0))
        elif feature['type'] == 'categorical':
            data[feature['name']] = feature.get('default', feature.get('options', [''])[0])
    # Realizar petición
    response = requests.post("http://localhost:8000/predict", json=data, timeout=10)
    assert response.status_code == 200, f"Status code: {response.status_code}"
    result = response.json()
    assert 'diagnostico' in result
    assert 'probabilidad' in result
    assert 'fecha' in result