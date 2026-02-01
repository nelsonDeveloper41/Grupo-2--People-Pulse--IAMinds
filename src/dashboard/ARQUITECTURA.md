# Arquitectura EcoCampus UPTC

## Estructura de Archivos

```
src/dashboard/
├── app.py                 # App principal Streamlit
├── claude_client.py       # Chatbot con Claude API
├── triggers.py            # Motor de alertas
├── data_simulator.py      # Simulador (backup)
├── models/
│   ├── model_loader.py    # Carga modelos XGBoost
│   ├── modelos_xgboost_real.pkl   # <-- MODELOS DE SEBASTIAN
│   └── modelos_xgboost_dummy.pkl  # Fallback
└── data/input/
    ├── demo_sensores.csv  # Datos de sensores simulados
    └── demo_periodos.csv  # Datos agregados por periodo
```

## Flujo de Datos

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  CSV Demo       │────▶│  app.py          │────▶│  Dashboard      │
│  (sensores)     │     │  (Streamlit)     │     │  (Visuales)     │
└─────────────────┘     └────────┬─────────┘     └─────────────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │  models/         │
                        │  XGBoost (36)    │
                        └──────────────────┘
```

## Conexion con XGBoost

### 1. Estructura del .pkl de Sebastian

```python
# El archivo modelos_xgboost_real.pkl debe tener esta estructura:
{
    'UPTC_TUN': {
        'energia_total_kwh': <modelo_xgboost>,
        'energia_comedor_kwh': <modelo_xgboost>,
        'energia_salones_kwh': <modelo_xgboost>,
        'energia_laboratorios_kwh': <modelo_xgboost>,
        'energia_auditorios_kwh': <modelo_xgboost>,
        'energia_oficinas_kwh': <modelo_xgboost>,
        'potencia_total_kw': <modelo_xgboost>,
        'agua_litros': <modelo_xgboost>,
        'co2_kg': <modelo_xgboost>,
    },
    'UPTC_SOG': { ... },  # 9 modelos
    'UPTC_DUI': { ... },  # 9 modelos
    'UPTC_CHI': { ... },  # 9 modelos
}
# Total: 36 modelos (9 targets x 4 sedes)
```

### 2. Como Usar los Modelos

```python
from models import predict, predict_all, get_predictor

# Opcion 1: Prediccion simple
energia = predict(
    sede='UPTC_TUN',
    target='energia_total_kwh',
    hora=10,
    ocupacion_pct=70,
    temperatura=18
)

# Opcion 2: Todos los targets de una sede
resultados = predict_all(
    sede='UPTC_TUN',
    hora=10,
    ocupacion_pct=70,
    temperatura=18
)
# Retorna: {'energia_total_kwh': 245.5, 'agua_litros': 1200, ...}

# Opcion 3: Usar la clase directamente
predictor = get_predictor()
hourly = predictor.predict_hourly('UPTC_TUN', 'energia_total_kwh', ocupacion_pct=70)
```

### 3. Features de Entrada (12)

```python
# El modelo espera estas 12 features en este orden:
FEATURES = [
    'hora_sin',                      # sin(2*pi*hora/24)
    'hora_cos',                      # cos(2*pi*hora/24)
    'es_fin_semana',                 # 0 o 1
    'es_festivo',                    # 0 o 1
    'semana_parciales',              # 0 o 1
    'semana_final',                  # 0 o 1
    'periodo_academico_semestre1',   # 0 o 1 (one-hot)
    'periodo_academico_semestre2',   # 0 o 1 (one-hot)
    'periodo_academico_vacaciones',  # 0 o 1 (one-hot)
    'temperatura_exterior',          # grados C
    'ocupacion_pct',                 # 0-100
    'co2',                           # kg (entrada)
]
```

### 4. Integracion en app.py

```python
# En app.py, el flujo seria:

# 1. Cargar datos del CSV (simulan sensores)
df = pd.read_csv('data/input/demo_sensores.csv')

# 2. Filtrar por sede y escenario
datos = df[(df['sede'] == 'UPTC_TUN') & (df['escenario'] == 'dia_normal')]

# 3. Preparar features para el modelo
features = datos[['hora_sin', 'hora_cos', 'es_fin_semana', ...]]

# 4. Hacer prediccion
from models import get_predictor
predictor = get_predictor()
prediccion = predictor.predict('UPTC_TUN', 'energia_total_kwh', features)

# 5. Comparar prediccion vs real
real = datos['energia_total_kwh'].values
diferencia = real - prediccion  # Si > 0, hay exceso

# 6. Mostrar en dashboard
st.metric("Consumo Real", f"{real.sum():.0f} kWh")
st.metric("Prediccion", f"{prediccion.sum():.0f} kWh")
```

## Componentes

### app.py
- Streamlit UI
- Selectores: Sede, Periodo
- Metricas, Graficos, Alertas
- Chatbot

### models/model_loader.py
- Carga .pkl automaticamente
- Prefiere modelo real sobre dummy
- Transforma features (hora -> sin/cos)

### triggers.py
- Detecta anomalias
- Genera alertas con nombre y costo
- Ej: "Vampiro Nocturno", "Pico Inesperado"

### claude_client.py
- Chatbot "Consejero del Rector"
- Recibe datos actuales
- Genera recomendaciones

## Instalar Modelos de Sebastian

1. Recibir archivo `.pkl` de Sebastian
2. Renombrar a `modelos_xgboost_real.pkl`
3. Copiar a `src/dashboard/models/`
4. Reiniciar la app

El loader detecta automaticamente el modelo real y lo usa.

## Variables de Entorno

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-api03-xxx  # Para chatbot Claude
```

## Ejecutar

```bash
cd src/dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Resumen Rapido

| Componente | Funcion |
|------------|---------|
| CSV Demo | Simula lecturas de sensores |
| XGBoost | Predice consumo esperado |
| Dashboard | Compara Real vs Prediccion |
| Triggers | Detecta anomalias |
| Claude | Explica y recomienda |
