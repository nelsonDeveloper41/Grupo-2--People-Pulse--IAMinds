"""
Módulo de modelos de predicción - EcoCampus UPTC
36 modelos XGBoost (9 targets × 4 sedes)
"""

from .model_loader import (
    # Clase principal
    EnergyPredictor,
    get_predictor,

    # Funciones de conveniencia
    predict,
    predict_sectores,
    predict_all,

    # Constantes
    FEATURE_NAMES,
    TARGET_NAMES,
    SEDES,
    SEDE_NOMBRES,
    TARGET_NOMBRES,
    TARGET_UNIDADES,
)

__all__ = [
    'EnergyPredictor',
    'get_predictor',
    'predict',
    'predict_sectores',
    'predict_all',
    'FEATURE_NAMES',
    'TARGET_NAMES',
    'SEDES',
    'SEDE_NOMBRES',
    'TARGET_NOMBRES',
    'TARGET_UNIDADES',
]
