"""
Script para verificar la conexión con Gemini y listar modelos disponibles
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Obtener API key
api_key = os.getenv("GOOGLE_API_KEY")
print(f"API Key encontrada: {'Sí' if api_key else 'No'}")
print(f"Longitud de la key: {len(api_key) if api_key else 0}")
print(f"Primeros 10 caracteres: {api_key[:10] if api_key else 'N/A'}...")

# Limpiar comillas si las tiene
api_key = api_key.strip().strip('"').strip("'") if api_key else None
print(f"Key limpia - Primeros 10: {api_key[:10] if api_key else 'N/A'}...")

# Configurar
genai.configure(api_key=api_key)

print("\n=== MODELOS DISPONIBLES ===")
try:
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"✓ {model.name}")
except Exception as e:
    print(f"Error al listar modelos: {e}")

print("\n=== PRUEBA DE GENERACIÓN ===")
# Probar con diferentes nombres de modelos
test_models = [
    "models/gemini-1.5-flash",
    "models/gemini-1.5-pro",
    "models/gemini-pro",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-pro",
    "models/gemini-2.0-flash-exp",
]

for model_name in test_models:
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Di hola")
        print(f"✅ {model_name} - FUNCIONA")
        print(f"   Respuesta: {response.text[:50]}...")
        break
    except Exception as e:
        print(f"❌ {model_name} - {str(e)[:60]}")
