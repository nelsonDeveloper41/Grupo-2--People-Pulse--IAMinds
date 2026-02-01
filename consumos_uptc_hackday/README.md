# 🏫 Modelo Predictivo de Consumos - UPTC HackDay

## 📋 Descripción del Proyecto

Este proyecto implementa un **sistema de Machine Learning** para predecir el consumo de energía eléctrica, agua y emisiones de CO2 en las 4 sedes de la **Universidad Pedagógica y Tecnológica de Colombia (UPTC)**.

El modelo utiliza **XGBoost** y fue desarrollado durante el **UPTC HackDay 2026**.

---

## 🎯 Objetivo

Desarrollar un modelo predictivo que permita:
- **Anticipar consumos** de energía y agua por hora
- **Detectar anomalías** en los patrones de consumo
- **Optimizar recursos** basándose en predicciones precisas
- **Reducir costos** operativos de la universidad

---

## 🏢 Sedes Analizadas

| Código | Sede | Características |
|--------|------|-----------------|
| `UPTC_TUN` | Tunja (Central) | Residencias estudiantiles + Comedor masivo |
| `UPTC_SOG` | Sogamoso | Industrial pesado (laboratorios de maquinaria) |
| `UPTC_DUI` | Duitama | Industrial/Técnico |
| `UPTC_CHI` | Chiquinquirá | Académico/Administrativo |

---

## 📊 Variables Predichas

El modelo predice **9 variables objetivo** para cada sede:

| Variable | Descripción |
|----------|-------------|
| `energia_total_kwh` | Consumo eléctrico total por hora |
| `energia_comedor_kwh` | Consumo del comedor universitario |
| `energia_salones_kwh` | Consumo de aulas de clase |
| `energia_laboratorios_kwh` | Consumo de laboratorios |
| `energia_auditorios_kwh` | Consumo de auditorios |
| `energia_oficinas_kwh` | Consumo de oficinas |
| `potencia_total_kw` | Potencia eléctrica instantánea |
| `agua_litros` | Consumo de agua |
| `co2_kg` | Emisiones de CO2 |

---

## 🔄 Pipeline de Procesamiento

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATOS CRUDOS                                      │
│                consumos_uptc.csv (~47MB, 400K+ registros)                   │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PASO 1-4: LIMPIEZA DE DATOS (Limpieza_Datos.py)                           │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Partición por sede (TUN, SOG, DUI, CHI)                                 │
│  • Auditoría forense (detecta flickering, gaps, inconsistencias)           │
│  • Saneamiento de metadatos (festivos Colombia, días semana)               │
│  • Detección de outliers (negativos, >1M, límites por sede)                │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SALIDA: Archivos preprocesados por sede                                   │
│  • preprocesado_UPTC_TUN.csv                                               │
│  • preprocesado_UPTC_SOG.csv                                               │
│  • preprocesado_UPTC_DUI.csv                                               │
│  • preprocesado_UPTC_CHI.csv                                               │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PASO 5-6: ENTRENAMIENTO XGBOOST (xgboost_training.py)                     │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Feature Engineering (lags, rolling, variables cíclicas)                 │
│  • Variables mágicas (inercia térmica, velocidad de cambio)                │
│  • Validación con TimeSeriesSplit (respeta orden temporal)                 │
│  • Detección de overfitting                                                │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SALIDA: 36 modelos entrenados (4 sedes × 9 targets)                       │
│  • MODELOS_XGBOOST/xgb_v3_{SEDE}_{TARGET}.pkl                              │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PASO 7: VISUALIZACIÓN (visualizar_arboles.py)                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Feature Importance por modelo                                           │
│  • Visualización de árboles de decisión                                    │
│  • Interpretabilidad del modelo                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Estructura del Proyecto

```
consumos_uptc_hackday/
│
├── 📄 Limpieza_Datos.py          # Pipeline de limpieza (Pasos 1-4)
├── 📄 xgboost_training.py        # Entrenamiento de modelos (Pasos 5-6)
├── 📄 visualizar_arboles.py      # Visualización e interpretabilidad (Paso 7)
│
├── 📂 MODELOS_XGBOOST/           # 36 modelos entrenados (.pkl)
├── 📂 RESULTADOS_ENTRENAMIENTO/  # Métricas y resultados (.json)
├── 📂 VISUALIZACIONES_ARBOLES/   # Gráficos generados (.png)
│
├── 📄 consumos_uptc.csv          # Datos crudos (~47MB)
├── 📄 sedes_uptc.csv             # Catálogo de sedes
├── 📄 preprocesado_UPTC_*.csv    # Datos limpios por sede
│
├── 📄 FASE 1-MODELO-PREDICTIVO.md  # Documentación del proyecto
├── 📄 CODEBOOK_UPTC.md             # Diccionario de datos
└── 📄 README.md                    # Este archivo
```

---

## 🚀 Cómo Ejecutar

### 1️⃣ Instalar Dependencias

```bash
pip install pandas numpy xgboost scikit-learn joblib matplotlib seaborn holidays
```

### 2️⃣ Ejecutar Limpieza de Datos

```bash
python Limpieza_Datos.py
```
**Salida:** `preprocesado_UPTC_*.csv` (4 archivos)

### 3️⃣ Entrenar Modelos

```bash
# Modo completo (con búsqueda de hiperparámetros)
python xgboost_training.py

# Modo rápido (sin búsqueda)
python xgboost_training.py --fast
```
**Salida:** 36 modelos en `MODELOS_XGBOOST/`

### 4️⃣ Visualizar Resultados

```bash
python visualizar_arboles.py --ejemplo
```

---

## 📈 Resultados del Modelo

### Rendimiento por Variable (R² Promedio)

| Variable | R² | Interpretación |
|----------|:--:|----------------|
| energia_oficinas_kwh | **0.97** | ✅ Excelente |
| energia_comedor_kwh | **0.96** | ✅ Excelente |
| energia_salones_kwh | **0.96** | ✅ Excelente |
| potencia_total_kw | **0.94** | ✅ Muy bueno |
| energia_laboratorios_kwh | **0.81** | ✅ Bueno |
| energia_total_kwh | 0.61 | ⚠️ Moderado |
| co2_kg | 0.60 | ⚠️ Moderado |
| agua_litros | 0.58 | ⚠️ Moderado |
| energia_auditorios_kwh | 0.06 | ❌ No recomendado |

---

## 🧠 Arquitectura del Modelo

El modelo utiliza **18+ features** organizadas en 5 categorías:

| Categoría | Features | Descripción |
|-----------|----------|-------------|
| **Temporales Cíclicas** | hora_sin/cos, dia_sem_sin/cos | Codificación circular |
| **Calendario** | es_festivo, es_fin_semana | Contexto operativo |
| **Exógenas** | temperatura, ocupacion | Factores físicos |
| **Memoria** | lag_1h, lag_24h, lag_168h | Valores históricos |
| **Variables Mágicas** | temp_hace_1h, cambio_temp | Inercia y velocidad |

---

## 📚 Documentación Adicional

- **[FASE 1-MODELO-PREDICTIVO.md](FASE%201-MODELO-PREDICTIVO.md)** - Plan detallado del proyecto
- **[CODEBOOK_UPTC.md](CODEBOOK_UPTC.md)** - Diccionario de variables

---

## 👥 Equipo

**Equipo UPTC HackDay 2026**

Universidad Pedagógica y Tecnológica de Colombia

---

## 📝 Licencia

Este proyecto fue desarrollado con fines académicos durante el UPTC HackDay 2026.
