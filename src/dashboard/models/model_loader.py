"""
Cargador de Modelos XGBoost v3 - EcoCampus UPTC
================================================
Carga los 36 modelos de Sebastian (9 targets x 4 sedes)
Formato: xgb_v3_{SEDE}_{TARGET}.pkl
"""

import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# ==============================================================================
# CONFIGURACION
# ==============================================================================

MODELS_DIR = Path(__file__).parent
XGBOOST_DIR = MODELS_DIR / "models_xgboost"

# Sedes disponibles
SEDES = ['UPTC_TUN', 'UPTC_SOG', 'UPTC_DUI', 'UPTC_CHI']

# Targets disponibles
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

# Features que esperan los modelos v3 (26 features)
FEATURE_NAMES = [
    'hora_sin', 'hora_cos',
    'dia_sem_sin', 'dia_sem_cos',
    'mes_sin', 'mes_cos',
    'es_fin_semana', 'es_festivo',
    'periodo_academico_semestre_1', 'periodo_academico_semestre_2',
    'periodo_academico_vacaciones_fin', 'periodo_academico_vacaciones_mitad',
    'temperatura_exterior_c', 'ocupacion_pct',
    'energia_total_kwh_lag_1h', 'energia_total_kwh_lag_24h', 'energia_total_kwh_lag_168h',
    'energia_total_kwh_rolling_mean_24h',
    'temp_hace_1h', 'temp_hace_3h',
    'ocupacion_hace_1h', 'ocupacion_hace_3h',
    'cambio_temp_1h', 'cambio_temp_3h',
    'cambio_ocupacion_1h',
    'energia_total_kwh_velocidad_1h',
]

# Mapeo de nombres
SEDE_NOMBRES = {
    'UPTC_TUN': 'Tunja',
    'UPTC_SOG': 'Sogamoso',
    'UPTC_DUI': 'Duitama',
    'UPTC_CHI': 'Chiquinquira',
}

TARGET_NOMBRES = {
    'energia_total_kwh': 'Energia Total',
    'energia_comedor_kwh': 'Comedores',
    'energia_salones_kwh': 'Salones',
    'energia_laboratorios_kwh': 'Laboratorios',
    'energia_auditorios_kwh': 'Auditorios',
    'energia_oficinas_kwh': 'Oficinas',
    'potencia_total_kw': 'Potencia Total',
    'agua_litros': 'Agua',
    'co2_kg': 'CO2',
}

TARGET_UNIDADES = {
    'energia_total_kwh': 'kWh',
    'energia_comedor_kwh': 'kWh',
    'energia_salones_kwh': 'kWh',
    'energia_laboratorios_kwh': 'kWh',
    'energia_auditorios_kwh': 'kWh',
    'energia_oficinas_kwh': 'kWh',
    'potencia_total_kw': 'kW',
    'agua_litros': 'L',
    'co2_kg': 'kg',
}

# Promedios por sede para estimar lags (basados en datos de entrenamiento)
PROMEDIOS_SEDE = {
    'UPTC_TUN': {'energia': 3.5, 'temp': 13.0, 'ocupacion': 35},
    'UPTC_SOG': {'energia': 8.5, 'temp': 14.0, 'ocupacion': 35},
    'UPTC_DUI': {'energia': 7.2, 'temp': 15.0, 'ocupacion': 35},
    'UPTC_CHI': {'energia': 2.8, 'temp': 14.0, 'ocupacion': 35},
}


# ==============================================================================
# CLASE PRINCIPAL
# ==============================================================================

class EnergyPredictor:
    """
    Predictor de consumo energetico usando modelos XGBoost v3.
    Soporta 36 modelos (9 targets x 4 sedes).
    """

    def __init__(self):
        """Inicializa el predictor cargando los modelos."""
        self.models = {}
        self.model_info = {}
        self.model_loaded = False
        self.is_dummy = False

        self._load_models()

    def _load_models(self):
        """Carga todos los modelos desde archivos individuales."""
        if not XGBOOST_DIR.exists():
            print(f"[WARN] Carpeta de modelos no encontrada: {XGBOOST_DIR}")
            return

        loaded_count = 0

        for sede in SEDES:
            self.models[sede] = {}
            self.model_info[sede] = {}

            for target in TARGET_NAMES:
                filename = f"xgb_v3_{sede}_{target}.pkl"
                filepath = XGBOOST_DIR / filename

                if filepath.exists():
                    try:
                        with open(filepath, 'rb') as f:
                            data = pickle.load(f)

                        # Extraer el modelo y metadata
                        self.models[sede][target] = data['model']
                        self.model_info[sede][target] = {
                            'feature_names': data.get('feature_names', FEATURE_NAMES),
                            'version': data.get('version', 'v3'),
                            'best_params': data.get('best_params', {}),
                        }
                        loaded_count += 1
                    except Exception as e:
                        print(f"[ERROR] Cargando {filename}: {e}")
                else:
                    print(f"[WARN] Modelo no encontrado: {filename}")

        if loaded_count > 0:
            self.model_loaded = True
            print(f"[OK] Modelos XGBoost cargados: {loaded_count}/36")
        else:
            print("[WARN] No se cargaron modelos")

    def build_features(
        self,
        hora: int = None,
        dia_semana: int = None,
        mes: int = None,
        es_fin_semana: bool = None,
        es_festivo: bool = False,
        periodo: str = 'semestre_1',
        temperatura: float = 15.0,
        ocupacion_pct: float = 50.0,
        sede: str = 'UPTC_TUN',
        # Valores historicos (para lags)
        energia_lag_1h: float = None,
        energia_lag_24h: float = None,
        energia_lag_168h: float = None,
        temp_hace_1h: float = None,
        temp_hace_3h: float = None,
        ocupacion_hace_1h: float = None,
        ocupacion_hace_3h: float = None,
    ) -> np.ndarray:
        """
        Construye el vector de 26 features para el modelo v3.
        """
        now = datetime.now()

        # Defaults
        if hora is None:
            hora = now.hour
        if dia_semana is None:
            dia_semana = now.weekday()
        if mes is None:
            mes = now.month
        if es_fin_semana is None:
            es_fin_semana = dia_semana >= 5

        # Promedios para estimar lags si no se proporcionan
        prom = PROMEDIOS_SEDE.get(sede, PROMEDIOS_SEDE['UPTC_TUN'])

        if energia_lag_1h is None:
            energia_lag_1h = prom['energia'] * (0.8 + 0.4 * ocupacion_pct / 100)
        if energia_lag_24h is None:
            energia_lag_24h = prom['energia']
        if energia_lag_168h is None:
            energia_lag_168h = prom['energia']
        if temp_hace_1h is None:
            temp_hace_1h = temperatura - 0.5
        if temp_hace_3h is None:
            temp_hace_3h = temperatura - 1.0
        if ocupacion_hace_1h is None:
            ocupacion_hace_1h = ocupacion_pct * 0.95
        if ocupacion_hace_3h is None:
            ocupacion_hace_3h = ocupacion_pct * 0.85

        # Codificaciones ciclicas
        hora_sin = np.sin(2 * np.pi * hora / 24)
        hora_cos = np.cos(2 * np.pi * hora / 24)
        dia_sem_sin = np.sin(2 * np.pi * dia_semana / 7)
        dia_sem_cos = np.cos(2 * np.pi * dia_semana / 7)
        mes_sin = np.sin(2 * np.pi * (mes - 1) / 12)
        mes_cos = np.cos(2 * np.pi * (mes - 1) / 12)

        # One-hot periodo academico
        periodo_s1 = 1 if periodo == 'semestre_1' else 0
        periodo_s2 = 1 if periodo == 'semestre_2' else 0
        periodo_vac_fin = 1 if periodo == 'vacaciones_fin' else 0
        periodo_vac_mit = 1 if periodo == 'vacaciones_mitad' else 0

        # Calcular features derivadas
        rolling_mean_24h = (energia_lag_1h + energia_lag_24h) / 2
        cambio_temp_1h = temperatura - temp_hace_1h
        cambio_temp_3h = temperatura - temp_hace_3h
        cambio_ocupacion_1h = ocupacion_pct - ocupacion_hace_1h
        velocidad_1h = energia_lag_1h - energia_lag_24h  # Simplificado

        # Construir array de features (26 features)
        features = np.array([[
            hora_sin, hora_cos,
            dia_sem_sin, dia_sem_cos,
            mes_sin, mes_cos,
            1 if es_fin_semana else 0,
            1 if es_festivo else 0,
            periodo_s1, periodo_s2,
            periodo_vac_fin, periodo_vac_mit,
            temperatura,
            ocupacion_pct,
            energia_lag_1h,
            energia_lag_24h,
            energia_lag_168h,
            rolling_mean_24h,
            temp_hace_1h,
            temp_hace_3h,
            ocupacion_hace_1h,
            ocupacion_hace_3h,
            cambio_temp_1h,
            cambio_temp_3h,
            cambio_ocupacion_1h,
            velocidad_1h,
        ]])

        return features

    def predict(
        self,
        sede: str = 'UPTC_TUN',
        target: str = 'energia_total_kwh',
        **kwargs
    ) -> float:
        """
        Predice un target especifico para una sede.
        """
        if not self.model_loaded:
            return self._fallback_prediction(sede, target)

        # Asegurar que sede esta en kwargs para build_features
        kwargs['sede'] = sede
        features = self.build_features(**kwargs)

        try:
            model = self.models[sede][target]
            prediction = model.predict(features)[0]
            return max(0, float(prediction))
        except KeyError:
            print(f"[WARN] Modelo no encontrado: {sede}/{target}")
            return self._fallback_prediction(sede, target)
        except Exception as e:
            print(f"[ERROR] Prediccion: {e}")
            return self._fallback_prediction(sede, target)

    def predict_all_targets(self, sede: str = 'UPTC_TUN', **kwargs) -> Dict[str, float]:
        """Predice todos los targets para una sede."""
        predictions = {}
        for target in TARGET_NAMES:
            predictions[target] = self.predict(sede=sede, target=target, **kwargs)
        return predictions

    def predict_all_sedes(self, target: str = 'energia_total_kwh', **kwargs) -> Dict[str, float]:
        """Predice un target para todas las sedes."""
        predictions = {}
        for sede in SEDES:
            predictions[sede] = self.predict(sede=sede, target=target, **kwargs)
        return predictions

    def predict_sectores(self, sede: str = 'UPTC_TUN', **kwargs) -> Dict[str, float]:
        """Predice los 5 sectores de energia."""
        return {
            'laboratorios': self.predict(sede=sede, target='energia_laboratorios_kwh', **kwargs),
            'oficinas': self.predict(sede=sede, target='energia_oficinas_kwh', **kwargs),
            'salones': self.predict(sede=sede, target='energia_salones_kwh', **kwargs),
            'comedores': self.predict(sede=sede, target='energia_comedor_kwh', **kwargs),
            'auditorios': self.predict(sede=sede, target='energia_auditorios_kwh', **kwargs),
        }

    def predict_hourly(self, sede: str = 'UPTC_TUN', target: str = 'energia_total_kwh', **kwargs) -> List[float]:
        """Predice valores para las 24 horas del dia."""
        hourly = []
        for hora in range(24):
            pred = self.predict(sede=sede, target=target, hora=hora, **kwargs)
            hourly.append(pred)
        return hourly

    def _fallback_prediction(self, sede: str, target: str) -> float:
        """Prediccion de fallback cuando no hay modelo."""
        prom = PROMEDIOS_SEDE.get(sede, PROMEDIOS_SEDE['UPTC_TUN'])
        base = prom['energia']

        factors = {
            'energia_total_kwh': 1.0,
            'energia_comedor_kwh': 0.12,
            'energia_salones_kwh': 0.25,
            'energia_laboratorios_kwh': 0.30,
            'energia_auditorios_kwh': 0.08,
            'energia_oficinas_kwh': 0.25,
            'potencia_total_kw': 1.3,
            'agua_litros': 500,
            'co2_kg': 0.2,
        }
        return base * factors.get(target, 1.0)

    def get_model_info(self) -> dict:
        """Retorna informacion sobre los modelos cargados."""
        return {
            'loaded': self.model_loaded,
            'is_dummy': self.is_dummy,
            'sedes': SEDES,
            'targets': TARGET_NAMES,
            'num_features': len(FEATURE_NAMES),
            'version': 'v3_magic_features',
        }


# ==============================================================================
# SINGLETON
# ==============================================================================
_predictor_instance = None

def get_predictor() -> EnergyPredictor:
    """Obtiene la instancia unica del predictor."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = EnergyPredictor()
    return _predictor_instance


# ==============================================================================
# FUNCIONES DE CONVENIENCIA
# ==============================================================================

def predict(sede: str = 'UPTC_TUN', target: str = 'energia_total_kwh', **kwargs) -> float:
    """Funcion rapida para una prediccion."""
    return get_predictor().predict(sede=sede, target=target, **kwargs)

def predict_sectores(sede: str = 'UPTC_TUN', **kwargs) -> Dict[str, float]:
    """Funcion rapida para predecir los 5 sectores."""
    return get_predictor().predict_sectores(sede=sede, **kwargs)

def predict_all(sede: str = 'UPTC_TUN', **kwargs) -> Dict[str, float]:
    """Funcion rapida para todos los targets."""
    return get_predictor().predict_all_targets(sede=sede, **kwargs)


# ==============================================================================
# TEST
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  TEST - Cargador de Modelos XGBoost v3")
    print("=" * 60)

    predictor = EnergyPredictor()
    info = predictor.get_model_info()

    print(f"\nModelos cargados: {info['loaded']}")
    print(f"Version: {info['version']}")
    print(f"Features: {info['num_features']}")

    if predictor.model_loaded:
        print("\n" + "-" * 60)
        print("Predicciones para UPTC Tunja (10 AM, 70% ocupacion)")
        print("-" * 60)

        predictions = predictor.predict_all_targets(
            sede='UPTC_TUN',
            hora=10,
            ocupacion_pct=70,
            temperatura=18
        )

        for target, valor in predictions.items():
            unidad = TARGET_UNIDADES[target]
            nombre = TARGET_NOMBRES[target]
            print(f"   {nombre:20}: {valor:10,.2f} {unidad}")

        print("\n" + "-" * 60)
        print("Comparacion entre sedes (energia total, 10 AM)")
        print("-" * 60)

        for sede in SEDES:
            valor = predictor.predict(sede=sede, target='energia_total_kwh', hora=10, ocupacion_pct=70)
            nombre = SEDE_NOMBRES[sede]
            print(f"   {nombre:15}: {valor:10,.2f} kWh")
