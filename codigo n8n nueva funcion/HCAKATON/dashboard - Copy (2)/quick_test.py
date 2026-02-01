import requests
import json
from datetime import datetime

N8N_WEBHOOK_URL = "https://anitra-nonpaid-shavon.ngrok-free.dev/webhook/ee3226db-2aba-4316-a3e1-b18bb9d1c042"

print("TEST 1: Conexión Básica...")
try:
    response = requests.post(N8N_WEBHOOK_URL, json={"test": "conexion"}, timeout=10)
    print(f"✅ Conexión exitosa - Status: {response.status_code}")
except Exception as e:
    print(f"❌ ERROR: {e}")

print("\nTEST 2: Enviar Alerta...")
try:
    payload = {
        "timestamp": datetime.now().isoformat(),
        "sector": "Laboratorios",
        "consumo": 1200,
        "alerta_type": "Consumo Crítico",
        "mensaje": "Prueba de alerta",
        "urgencia": "alta"
    }
    response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=15)
    print(f"✅ Alerta enviada - Status: {response.status_code}")
    print("💬 Revisa Telegram en 3-5 segundos")
except Exception as e:
    print(f"❌ ERROR: {e}")
