# EcoCampus UPTC

**Sistema inteligente de monitoreo y prediccion de consumo energetico para la Universidad Pedagogica y Tecnologica de Colombia (UPTC).**

Desarrollado por **Equipo NovaIA** - Hackathon IAMinds 2026

---

## Descripcion

EcoCampus UPTC es una plataforma que utiliza **Machine Learning (XGBoost)** para predecir el consumo energetico esperado y compararlo con datos reales de sensores, permitiendo:

- Detectar anomalias de consumo (picos, vampiros nocturnos, fugas)
- Generar alertas en tiempo real
- Visualizar consumo por sede, sector y periodo
- Estimar impacto economico y huella de carbono
- Recibir recomendaciones de un asistente IA (Claude)

---

## Screenshots

| Vista Dia | Vista Semana |
|-----------|--------------|
| Curva horaria con picos detectados | Consumo diario con anomalias |

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│                   STREAMLIT DASHBOARD                    │
│   Metricas  │  Graficas Real vs Esperado  │  Alertas   │
└─────────────────────────────────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
   │  CSV Data   │   │  XGBoost    │   │  Claude API │
   │  (Sensores) │   │  36 Modelos │   │  (Chatbot)  │
   └─────────────┘   └─────────────┘   └─────────────┘
```

### Modelos XGBoost

- **36 modelos** = 9 targets × 4 sedes
- **26 features** de entrada (hora, temperatura, ocupacion, lags temporales)
- **R² = 0.95** de precision

| Sedes | Targets |
|-------|---------|
| Tunja, Sogamoso, Duitama, Chiquinquira | Energia total, Sectores (5), Potencia, Agua, CO2 |

---

## Instalacion

### Requisitos
- Python 3.9+
- pip

### Pasos

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/ecocampus-uptc.git
cd ecocampus-uptc

# Instalar dependencias
cd src/dashboard
pip install -r requirements.txt

# Configurar API Key (opcional, para chatbot)
# Crear archivo .env con:
# ANTHROPIC_API_KEY=sk-ant-api03-xxx

# Ejecutar
streamlit run app.py
```

La app estara disponible en `http://localhost:8501`

---

## Estructura del Proyecto

```
src/dashboard/
├── app.py                    # App principal Streamlit
├── claude_client.py          # Chatbot con Claude API
├── triggers.py               # Motor de alertas
├── models/
│   ├── model_loader.py       # Cargador de modelos XGBoost
│   └── models_xgboost/       # 36 modelos .pkl
└── data/input/
    ├── demo_sensores.csv     # Datos de sensores simulados
    ├── demo_periodos.csv     # Datos agregados por periodo
    └── generate_*.py         # Scripts de generacion
```

---

## Uso

### Seleccionar Sede
Elige entre las 4 sedes de la UPTC en el panel lateral.

### Seleccionar Periodo
Usa los botones para ver consumo por:
- **Dia**: 24 horas
- **Semana**: 7 dias
- **Mes**: 30 dias
- **Semestre**: 26 semanas
- **Año**: 12 meses

### Interpretar Graficas
- **Linea azul (Real)**: Consumo medido por sensores
- **Linea gris (Esperado)**: Prediccion del modelo XGBoost
- **Area sombreada**: Diferencia (exceso o ahorro)
- **Puntos rojos**: Anomalias detectadas

---

## Tecnologias

| Componente | Tecnologia |
|------------|------------|
| Frontend | Streamlit, Plotly |
| Machine Learning | XGBoost, scikit-learn |
| Datos | Pandas, NumPy |
| Chatbot IA | Claude API (Anthropic) |
| Lenguaje | Python 3.9+ |

---

## Equipo NovaIA

Proyecto desarrollado para el **Hackathon IAMinds 2026**.

### Integrantes

| Nombre | Rol |
|--------|-----|
| **Nelson Rodriguez Silva** | Desarrollo Frontend & Integracion |
| **Walter Pelaez** | Machine Learning & Modelos XGBoost |
| **Juan Sebastian Urbina Silva** | Data Engineering & Backend |

---

## Licencia

MIT License - Ver archivo LICENSE para mas detalles.
