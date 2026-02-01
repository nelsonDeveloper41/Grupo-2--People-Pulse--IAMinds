# EcoCampus UPTC - Arquitectura de la App

## Resumen Ejecutivo

Sistema de monitoreo energetico que compara consumo **Real (sensores)** vs **Esperado (IA)** para detectar anomalias y generar alertas de ahorro.

---

## Arquitectura

```
┌──────────────────────────────────────────────────────────────────┐
│                        DASHBOARD (Streamlit)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │
│  │  Metricas   │  │  Graficas   │  │   Alertas   │               │
│  │  Consumo    │  │  Real vs    │  │  Anomalias  │               │
│  │  CO2, Agua  │  │  Esperado   │  │  Detectadas │               │
│  └─────────────┘  └─────────────┘  └─────────────┘               │
└──────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
┌───────────────────┐ ┌───────────────┐ ┌───────────────────┐
│   CSV Demo Data   │ │ XGBoost (36)  │ │   Claude API      │
│   (Sensores)      │ │ (Prediccion)  │ │   (Chatbot IA)    │
└───────────────────┘ └───────────────┘ └───────────────────┘
```

---

## Componentes Principales

### 1. Modelos XGBoost (36 modelos)
- **9 targets** x **4 sedes** = 36 modelos
- Targets: energia_total, sectores (5), potencia, agua, CO2
- Sedes: Tunja, Sogamoso, Duitama, Chiquinquira
- **26 features** de entrada (hora, temperatura, ocupacion, lags...)

### 2. Datos Demo
| Archivo | Contenido | Uso |
|---------|-----------|-----|
| `demo_periodos.csv` | Curvas pre-calculadas | Vistas Dia/Semana/Mes/Semestre/Año |
| `demo_sensores.csv` | Datos horarios | Selector de escenarios |

### 3. Flujo de Datos

```
generate_periods_data.py          app.py (Streamlit)
         │                              │
         ▼                              ▼
┌─────────────────┐            ┌─────────────────┐
│ XGBoost.predict │            │ Lee CSV         │
│ para cada hora  │            │ (pre-calculado) │
└────────┬────────┘            └────────┬────────┘
         │                              │
         ▼                              ▼
┌─────────────────┐            ┌─────────────────┐
│ Guarda en CSV:  │            │ Muestra grafica │
│ - curva_valores │ ────────►  │ Real vs Esperado│
│ - curva_esperada│            │                 │
└─────────────────┘            └─────────────────┘
```

---

## Deteccion de Anomalias

La app compara **Real** vs **Esperado**:

| Situacion | Significado | Accion |
|-----------|-------------|--------|
| Real > Esperado | Sobreconsumo | Alerta roja |
| Real ≈ Esperado | Normal | OK |
| Real < Esperado | Ahorro | Alerta verde |

**Tipos de anomalias detectadas:**
- Picos inesperados (consumo 2-3x mayor)
- Vampiros nocturnos (consumo alto de noche)
- Fugas (consumo constante anormal)

---

## Precision del Modelo

- **R² = 0.95** (coeficiente de determinacion)
- El modelo explica el 95% de la variabilidad del consumo
- Entrenado con datos 2018-2024

---

## Stack Tecnologico

| Componente | Tecnologia |
|------------|------------|
| Frontend | Streamlit + Plotly |
| Modelos ML | XGBoost (sklearn) |
| Chatbot | Claude API (Anthropic) |
| Datos | CSV / Pandas |

---

## Ejecucion

```bash
cd src/dashboard
pip install -r requirements.txt
streamlit run app.py
```

---

## Equipo NovaIA - Hackathon IAMinds 2026
