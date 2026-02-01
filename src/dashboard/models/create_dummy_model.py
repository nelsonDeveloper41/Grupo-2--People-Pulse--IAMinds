"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         Generador de Modelos XGBoost Dummy - EcoCampus UPTC                  ║
║                                                                              ║
║  Estructura EXACTA de Sebastián:                                             ║
║  - 36 modelos (9 targets × 4 sedes)                                          ║
║  - Features: hora_sin, hora_cos, binarias, one-hot, temperatura, etc.        ║
║                                                                              ║
║  IMPORTANTE: Reemplazar con modelos reales de Sebastián cuando estén listos  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import joblib
from pathlib import Path
import json

# ==============================================================================
# CONFIGURACIÓN - EXACTA DE SEBASTIÁN
# ==============================================================================

# Features de entrada (12 features)
FEATURE_NAMES = [
    'hora_sin',                      # sin(2π × hora/24)
    'hora_cos',                      # cos(2π × hora/24)
    'es_fin_semana',                 # 0 o 1
    'es_festivo',                    # 0 o 1
    'semana_parciales',              # 0 o 1
    'semana_final',                  # 0 o 1
    'periodo_academico_semestre1',   # One-hot
    'periodo_academico_semestre2',   # One-hot
    'periodo_academico_vacaciones',  # One-hot
    'temperatura_exterior',          # °C
    'ocupacion_pct',                 # 0-100
    'co2',                           # kg (entrada, no confundir con target)
]

# Targets (9 predicciones por sede)
TARGET_NAMES = [
    'energia_total_kwh',
    'energia_comedor_kwh',
    'energia_salones_kwh',
    'energia_laboratorios_kwh',
    'energia_auditorios_kwh',
    'energia_oficinas_kwh',
    'potencia_total_kw',
    'agua_litros',
    'co2_kg',
]

# Sedes (4)
SEDES = ['UPTC_TUN', 'UPTC_SOG', 'UPTC_DUI', 'UPTC_CHI']

# R² aproximados de Sebastián (para calibrar el ruido del dummy)
# Mayor R² = menos ruido, predicción más precisa
R2_SCORES = {
    'energia_total_kwh': 0.95,
    'energia_comedor_kwh': 0.95,
    'energia_salones_kwh': 0.93,
    'energia_laboratorios_kwh': 0.85,
    'energia_auditorios_kwh': 0.06,   # Débil - más variabilidad
    'energia_oficinas_kwh': 0.06,     # Débil - más variabilidad
    'potencia_total_kw': 0.97,
    'agua_litros': 0.80,
    'co2_kg': 0.90,
}

# ==============================================================================
# CLASE DUMMY MODEL - Simula comportamiento de XGBoost
# ==============================================================================

class DummyXGBoostModel:
    """
    Modelo dummy que simula predicciones realistas.
    Tiene la misma interfaz que un modelo XGBoost real.
    """

    def __init__(self, target_name: str, sede: str):
        self.target_name = target_name
        self.sede = sede
        self.r2 = R2_SCORES.get(target_name, 0.5)

        # Parámetros base por target (valores típicos)
        self.base_values = {
            'energia_total_kwh': 2500,
            'energia_comedor_kwh': 400,
            'energia_salones_kwh': 350,
            'energia_laboratorios_kwh': 800,
            'energia_auditorios_kwh': 150,
            'energia_oficinas_kwh': 300,
            'potencia_total_kw': 450,
            'agua_litros': 15000,
            'co2_kg': 410,
        }

        # Ajuste por sede (Tunja es la más grande)
        self.sede_factors = {
            'UPTC_TUN': 1.0,
            'UPTC_SOG': 0.7,
            'UPTC_DUI': 0.5,
            'UPTC_CHI': 0.3,
        }

    def predict(self, X):
        """
        Predice valores simulando comportamiento realista.
        X: array de shape (n_samples, 12) con las features
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)

        base = self.base_values[self.target_name]
        sede_factor = self.sede_factors[self.sede]

        for i in range(n_samples):
            # Extraer features
            hora_sin = X[i, 0]
            hora_cos = X[i, 1]
            es_fin_semana = X[i, 2]
            es_festivo = X[i, 3]
            semana_parciales = X[i, 4]
            semana_final = X[i, 5]
            semestre1 = X[i, 6]
            semestre2 = X[i, 7]
            vacaciones = X[i, 8]
            temperatura = X[i, 9]
            ocupacion = X[i, 10]

            # Reconstruir hora aproximada desde sin/cos
            hora_aprox = np.arctan2(hora_sin, hora_cos) * 24 / (2 * np.pi)
            if hora_aprox < 0:
                hora_aprox += 24

            # Factor hora (pico 10-14h y 18-20h)
            if 10 <= hora_aprox <= 14:
                factor_hora = 1.4
            elif 18 <= hora_aprox <= 20:
                factor_hora = 1.2
            elif 0 <= hora_aprox <= 6:
                factor_hora = 0.25
            elif 22 <= hora_aprox <= 24:
                factor_hora = 0.3
            else:
                factor_hora = 1.0

            # Factor ocupación
            factor_ocupacion = 0.3 + (ocupacion / 100) * 0.9

            # Factor día (fin de semana/festivo = menos consumo)
            if es_festivo:
                factor_dia = 0.2
            elif es_fin_semana:
                factor_dia = 0.35
            else:
                factor_dia = 1.0

            # Factor periodo académico
            if vacaciones:
                factor_periodo = 0.25
            elif semana_final:
                factor_periodo = 1.3  # Más actividad en finales
            elif semana_parciales:
                factor_periodo = 1.15
            else:
                factor_periodo = 1.0

            # Factor temperatura (más consumo si hace calor por A/C)
            factor_temp = 1.0 + max(0, (temperatura - 22)) * 0.03

            # Calcular predicción base
            pred = base * sede_factor * factor_hora * factor_ocupacion * factor_dia * factor_periodo * factor_temp

            # Agregar ruido según R² (menor R² = más ruido)
            noise_level = (1 - self.r2) * 0.5  # 0% a 50% de ruido
            noise = np.random.normal(0, pred * noise_level)
            pred += noise

            # Especial para auditorios y oficinas (muy variables según Sebastián)
            if self.target_name in ['energia_auditorios_kwh', 'energia_oficinas_kwh']:
                # Agregar variabilidad extra aleatoria
                pred *= np.random.uniform(0.3, 1.8)

            predictions[i] = max(0, pred)

        return predictions


# ==============================================================================
# GENERAR Y GUARDAR MODELOS DUMMY
# ==============================================================================

def create_all_dummy_models():
    """Crea los 36 modelos dummy (9 targets × 4 sedes)"""

    models_dir = Path(__file__).parent

    print("=" * 60)
    print("  GENERADOR DE MODELOS DUMMY - EcoCampus UPTC")
    print("  Estructura de Sebastián: 36 modelos (9×4)")
    print("=" * 60)
    print()

    all_models = {}

    for sede in SEDES:
        print(f"\n📍 Sede: {sede}")
        sede_models = {}

        for target in TARGET_NAMES:
            model = DummyXGBoostModel(target, sede)
            model_key = f"{sede}_{target}"
            sede_models[target] = model

            r2 = R2_SCORES.get(target, 0.5)
            status = "✅" if r2 > 0.7 else "⚠️" if r2 > 0.3 else "❌"
            print(f"   {status} {target:25} (R² ≈ {r2:.2f})")

        all_models[sede] = sede_models

    # Guardar todos los modelos en un solo archivo
    models_path = models_dir / "modelos_xgboost_dummy.pkl"
    joblib.dump(all_models, models_path)
    print(f"\n✅ Modelos guardados en: {models_path}")

    # Guardar configuración
    config = {
        'features': FEATURE_NAMES,
        'targets': TARGET_NAMES,
        'sedes': SEDES,
        'r2_scores': R2_SCORES,
        'is_dummy': True,
        'version': '1.0',
        'author': 'Dummy generator - Reemplazar con modelos de Sebastián'
    }

    config_path = models_dir / "model_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"✅ Configuración guardada en: {config_path}")

    # Test rápido
    print("\n" + "=" * 60)
    print("  TEST RÁPIDO - Predicción para UPTC_TUN")
    print("=" * 60)

    # Simular features para las 10 AM, día normal, semestre 1
    hora = 10
    test_features = np.array([[
        np.sin(2 * np.pi * hora / 24),  # hora_sin
        np.cos(2 * np.pi * hora / 24),  # hora_cos
        0,    # es_fin_semana
        0,    # es_festivo
        0,    # semana_parciales
        0,    # semana_final
        1,    # periodo_academico_semestre1
        0,    # periodo_academico_semestre2
        0,    # periodo_academico_vacaciones
        18,   # temperatura_exterior
        70,   # ocupacion_pct
        400,  # co2 (entrada)
    ]])

    print(f"\nFeatures de prueba (10 AM, día normal, 70% ocupación):")
    for sede in ['UPTC_TUN']:
        print(f"\n📍 {sede}:")
        for target in TARGET_NAMES:
            pred = all_models[sede][target].predict(test_features)[0]
            unit = 'kWh' if 'kwh' in target else 'kW' if 'kw' in target else 'L' if 'litros' in target else 'kg'
            print(f"   {target:25}: {pred:10,.1f} {unit}")

    return all_models


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    create_all_dummy_models()
    print("\n🎯 Listo para integrar en la aplicación.")
    print("📝 Cuando Sebastián envíe los modelos reales, reemplaza modelos_xgboost_dummy.pkl")
