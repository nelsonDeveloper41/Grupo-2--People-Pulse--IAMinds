# Modelos de Predicción - EcoCampus UPTC

## Estructura de Sebastián

**36 modelos XGBoost** (9 targets × 4 sedes)

### Sedes (4)
| Código | Nombre |
|--------|--------|
| UPTC_TUN | Tunja |
| UPTC_SOG | Sogamoso |
| UPTC_DUI | Duitama |
| UPTC_CHI | Chiquinquirá |

### Targets (9 por sede)
| Target | Descripción | R² Sebastián |
|--------|-------------|--------------|
| energia_total_kwh | Energía total | ~0.95 |
| energia_comedor_kwh | Comedores | ~0.95 ✅ |
| energia_salones_kwh | Salones | ~0.93 ✅ |
| energia_laboratorios_kwh | Laboratorios | ~0.77-0.97 |
| energia_auditorios_kwh | Auditorios | ~0.04-0.08 ⚠️ |
| energia_oficinas_kwh | Oficinas | ~0.04-0.08 ⚠️ |
| potencia_total_kw | Potencia total | ~0.97 ✅ |
| agua_litros | Consumo agua | ~0.80 |
| co2_kg | Emisiones CO₂ | ~0.90 |

### Features de Entrada (12)
```python
FEATURES = [
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
    'co2',                           # kg (entrada)
]
```

## Archivos

```
models/
├── README.md                    # Este archivo
├── __init__.py                  # Exports del módulo
├── create_dummy_model.py        # Genera modelos simulados
├── model_loader.py              # API para usar los modelos
├── modelos_xgboost_dummy.pkl    # Modelos dummy (generado)
├── modelos_xgboost_real.pkl     # Modelos de Sebastián (PENDIENTE)
└── model_config.json            # Configuración (generado)
```

## Uso Rápido

### 1. Generar modelos dummy

```bash
cd src/dashboard/models
pip install numpy joblib
python create_dummy_model.py
```

### 2. Usar en la aplicación

```python
from models import predict, predict_sectores, predict_all

# Predecir un target específico
energia = predict(
    sede='UPTC_TUN',
    target='energia_total_kwh',
    hora=10,
    ocupacion_pct=70,
    temperatura=18
)

# Predecir los 5 sectores de Tunja (para el dashboard actual)
sectores = predict_sectores(hora=10, ocupacion_pct=70)
# Returns: {'laboratorios': X, 'oficinas': Y, 'salones': Z, ...}

# Predecir todos los targets
todos = predict_all(hora=10, ocupacion_pct=70)
```

### 3. Uso avanzado con clase

```python
from models import EnergyPredictor

predictor = EnergyPredictor()

# Predicción horaria (24 valores)
hourly = predictor.predict_hourly(
    sede='UPTC_TUN',
    target='energia_total_kwh',
    ocupacion_pct=70
)

# Comparar todas las sedes
por_sede = predictor.predict_all_sedes(target='energia_total_kwh')

# Info del modelo
info = predictor.get_model_info()
print(f"Usando modelo: {'DUMMY' if info['is_dummy'] else 'REAL'}")
```

## Integrar Modelos Reales de Sebastián

Cuando Sebastián entregue los modelos:

1. **Renombrar** su archivo a `modelos_xgboost_real.pkl`
2. **Colocar** en esta carpeta (`src/dashboard/models/`)
3. **Verificar** que la estructura sea:
   ```python
   {
       'UPTC_TUN': {
           'energia_total_kwh': modelo,
           'energia_comedor_kwh': modelo,
           ...
       },
       'UPTC_SOG': {...},
       'UPTC_DUI': {...},
       'UPTC_CHI': {...},
   }
   ```
4. El loader **automáticamente** prefiere el modelo real sobre el dummy

## Métricas de Sebastián (Referencia)

```
RMSE: 39.1361
MAE:  11.7025
R²:   0.5372 (promedio, ACEPTABLE)
MAPE: 14.44%

Top 5 Features:
1. es_fin_semana
2. ocupacion_pct
3. periodo_academico_vacaciones
4. periodo_academico_semestre
5. hora (codificada como sin/cos)
```

## Notas

- Los modelos de **auditorios** y **oficinas** tienen R² bajo (~0.06), el dummy agrega más variabilidad para simular esto
- El modelo dummy NO requiere XGBoost instalado (usa clases Python simples)
- Cuando llegue el modelo real, SÍ se necesita `xgboost` instalado
