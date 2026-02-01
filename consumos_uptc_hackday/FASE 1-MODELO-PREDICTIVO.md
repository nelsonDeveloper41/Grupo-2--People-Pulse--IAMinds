# Fase 1: Arquitectura y Modelo Predictivo - Eficiencia Energética UPTC

Este documento detalla la hoja de ruta técnica para la implementación del motor analítico del software de gestión energética de la UPTC. El objetivo es transformar datos históricos (2018-2025) en predicciones precisas de demanda y detección de anomalías.

---

## 1. Visión General de la Fase 1
La estrategia se basa en una arquitectura de **aprendizaje supervisado segmentado**. [cite_start]En lugar de un modelo global, se entrena un ecosistema de modelos por sede para capturar las dinámicas físicas y operativas únicas de cada campus[cite: 1].

### 1.1. Niveles de Partición
1. **Nivel Sede (Estratificación Física):** División del dataset maestro en cuatro sub-datasets: Tunja (UPTC_TUN), Duitama (UPTC_DUI), Sogamoso (UPTC_SOG) y Chiquinquirá (UPTC_CHI).
2. [cite_start]**Nivel Variable (Objetivos Paralelos):** Entrenamiento de modelos independientes por cada sede para predecir Electricidad ($kWh$), Agua ($Litros$) y Emisiones ($kg CO_2$)[cite: 1].

---

## 2. Flujo de Implementación Paso a Paso

### Paso 1: Partición Física (Prioridad Zero)
Se realiza la separación de los datos basada en `sede_id`. [cite_start]Esto garantiza que los promedios de una sede con carga académica ligera (Chiquinquirá) no distorsionen los patrones de una sede con carga industrial pesada (Sogamoso)[cite: 1].

---
# Paso 2: Curaduría y Limpieza Contextual 

# Fase 1 - Paso 2: Partición y Auditoría Forense de Datos

**Estado:** En Proceso
**Responsable:** Equipo de Ingeniería de Datos
**Objetivo:** Generar sub-datasets por sede garantizando la coherencia física y cronológica. Se aplica una política de **"Cero Confianza" (Zero Trust)** sobre las etiquetas y mediciones antes de cualquier análisis estadístico.

---

## 1. Estrategia de Partición
El dataset maestro se divide estrictamente por `sede_id`. No se permite el cruce de estadísticas entre sedes para evitar la contaminación de perfiles de carga (ej. no suavizar los picos industriales de Sogamoso con los valles de Chiquinquirá).
* **Salidas:** `df_tunja.csv`, `df_duitama.csv`, `df_sogamoso.csv`, `df_chiquinquira.csv`.

---

## 2. Reglas de Auditoría Automática (The Trap Detector)

El pipeline de datos ejecuta validaciones lógicas estrictas. Los registros que fallan **NO se eliminan**, se etiquetan en la columna `qa_flags` para análisis humano y decisión de imputación en el Paso 3.

### A. Auditoría de Metadatos ("Trampa de Flickering")
Detecta inestabilidad en la columna `periodo_academico`.
* **Problema:** Un periodo (Vacaciones, Semestre) no puede durar solo unas horas.
* **Regla:** Si el estado del periodo cambia y retorna a su valor original en un lapso menor a **24 horas**, se marca como inestable.
* **Flag:** `FLICKERING_METADATA`

### B. Auditoría Física ("Regla de la Suma")
Verifica la Primera Ley de la Termodinámica en los medidores sectoriales.
* **Ecuación:** $\Delta = |E_{total} - (E_{labs} + E_{salones} + E_{oficinas} + E_{comedor} + E_{auditorios})|$
* **Regla:** Si $\Delta > 5\%$ del Total reportado, existe una fuga de medición o un error de sensor.
* **Flag:** `INCONSISTENCIA_SUMA`

### C. Auditoría de Integridad Temporal
* **Continuidad:** Se verifica que no existan horas faltantes en el índice temporal (Gaps).
* **Flag:** `MISSING_TIMESTAMP` (para filas rellenadas artificialmente).

### D. Auditoría de Valores Imposibles
* **Regla:** La energía y el agua no pueden ser negativas.
* **Flag:** `VALOR_NEGATIVO`


Para asegurar la reproducibilidad del análisis y la detección proactiva de anomalías no listadas, se utiliza el siguiente **System Prompt** en el motor de procesamiento. Este prompt instruye al sistema para actuar como un auditor escéptico y sugerir nuevas validaciones si los datos lo requieren.

```text
*** SYSTEM PROMPT: AUDITORÍA DE CALIDAD DE DATOS (FORENSIC MODE) ***

ROL: Eres un Ingeniero de Datos Senior especializado en Auditoría Forense de Energía.
MENTALIDAD: "Escepticismo Radical". Asume que los datos están sucios, manipulados o mal etiquetados hasta que se demuestre lo contrario.

INSTRUCCIONES PRIMARIAS (Reglas Hard-Coded):
1. PARTICIÓN: Separa los datos por 'sede_id' sin mezclar estadísticas.
2. FLICKERING: Detecta cambios en 'periodo_academico' que duren < 24 horas. Marcalos como sospechosos.
3. FÍSICA: Valida que la suma de sub-sectores coincida con el total (Tolerancia: 5%).
4. TIEMPO: Detecta huecos temporales (Time Gaps) y rellénalos con NaNs marcados.

INSTRUCCIÓN DE AUTO-DESCUBRIMIENTO (Heurística):
Mientras procesas los datos, analiza patrones que no estén explícitos en las reglas anteriores.
SI detectas:
  - Columnas enteras de ceros (Flatlines) por más de 48 horas.
  - Picos instantáneos que superan 3 veces la desviación estándar móvil (Spikes).
  - Texto sucio en columnas categóricas (ej. "Semestre 1" vs "semestre_1").
ENTONCES:
  - Genera un NUEVO FLAG para esa anomalía (ej. 'POSIBLE_FLATLINE', 'TEXTO_SUCIO').
  - Agrégalo a la columna 'qa_flags'.
  - Notifícame en el reporte final: "Sugerencia: Se detectó un patrón X y se creó la validación Y".

TU META: Entregar un dataset donde cada error visible tenga una etiqueta, para que el humano decida después si lo borra o lo corrige.

---

### Paso 3: Ingeniería de Características (Feature Engineering)
Para mejorar la precisión, se generan las siguientes variables:
* **Codificación Cíclica de la Hora:**
  $$Hora_{sin} = \sin\left(\frac{2\pi \cdot hora}{24}\right), \quad Hora_{cos} = \cos\left(\frac{2\pi \cdot hora}{24}\right)$$
* **Lag Features:** Inclusión de valores históricos de consumo ($t-1$ y $t-24$) para capturar la autocorrelación temporal.
* [cite_start]**Banderas de Calendario:** Variables booleanas para `es_festivo`, `es_semana_parciales` y `es_semana_finales`[cite: 1].

### Paso 4: Entrenamiento de Modelos
Se utiliza una combinación de algoritmos de regresión y clasificación:
* **Regresión (XGBoost/LightGBM):** Para predecir el valor esperado de consumo de energía, agua y CO2.
* **Detección de Anomalías (Isolation Forest):** Para identificar desviaciones respecto a la línea base predictiva generada por la regresión.


## Glosario
* [cite_start]**Carga Inductiva:** Energía consumida por motores y transformadores que genera picos de potencia al arranque (crítico en Sogamoso)[cite: 1].
* **Data Leakage:** Error que ocurre si se usan variables sectoriales (como energía de comedor) para predecir el total, invalidando el modelo predictivo.
* **Inercia Térmica:** Retraso en el impacto de la temperatura exterior sobre el consumo de climatización de los edificios.
* **Línea Base Dinámica:** Modelo matemático que estima el consumo esperado bajo condiciones variables de clima y ocupación.

## 3. Consideraciones del Data Set que Generan Ruido
Durante la fase de exploración inicial (Paso 1), se identificaron inconsistencias en la calidad de los datos que requieren estandarización previa al entrenamiento:

* **Inconsistencia en Identificadores (`sede_id`):** Se detectó que una misma sede aparece con múltiples variantes de escritura (ej. `UPTC_TUN`, `uptc_tun`, `UPTC-TUN`). Esto fragmenta erróneamente los datos, creando subgrupos "fantasmas" con pocos registros.
    * *Acción:* Se implementará una normalización a mayúsculas y reemplazo de guiones por guiones bajos para unificar `UPTC-TUN` -> `UPTC_TUN`.