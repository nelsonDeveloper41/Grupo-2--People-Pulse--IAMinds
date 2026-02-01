import pandas as pd
import os
import numpy as np
import holidays
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
# ==========================================
# Algoritmo-UPTC
# Fase 1: Arquitectura y Modelo Predictivo
# ==========================================

# Definir rutas de archivos
BASE_DIR = r"c:\Users\POWER\OneDrive\Escritorio\consumos_uptc_hackday"
OUTPUT_DIR = BASE_DIR  # Directorio de salida para archivos generados
files = {
    "consumos": os.path.join(BASE_DIR, "consumos_uptc.csv"),
    "sedes": os.path.join(BASE_DIR, "sedes_uptc.csv")
}

def cargar_datos():
    """Carga los datasets de consumos y sedes."""
    print(">>> Cargando datasets...")
    try:
        df_consumos = pd.read_csv(files["consumos"])
        df_sedes = pd.read_csv(files["sedes"])
        print(f"Datos de consumos cargados: {df_consumos.shape}")
        print(f"Datos de sedes cargados: {df_sedes.shape}")
        return df_consumos, df_sedes
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo - {e}")
        return None, None

def paso_1_particion_fisica(df_consumos, df_sedes):
    """
    PASO 1: Partición Física (Prioridad Zero)
    Se realiza la separación de los datos basada en sede_id.
    Objetivo: Ver los datos de cada sede individualmente.
    """
    print("\n" + "="*50)
    print(" EJECUTANDO PASO 1: PARTICIÓN FÍSICA POR SEDE")
    print("="*50)

    # Unir con nombres de sedes para mayor claridad (si aplica)
    # Asumimos que hay una columna común, revisaremos las columnas primero
    print("Columnas en consumos:", df_consumos.columns.tolist())
    print("Columnas en sedes:", df_sedes.columns.tolist())

    # Lista de sedes únicas en el dataset de consumos
    sedes_unicas = df_consumos['sede_id'].unique()
    print(f"\nSedes encontradas (Original): {sedes_unicas}")

    # --- ESTANDARIZACIÓN DE SEDE_ID ---
    print("\n>>> Aplicando corrección de inconsistencias en sede_id...")
    # 1. Convertir a mayúsculas
    df_consumos['sede_id'] = df_consumos['sede_id'].str.upper()
    # 2. Reemplazar guiones por guiones bajos
    df_consumos['sede_id'] = df_consumos['sede_id'].str.replace('-', '_')
    
    sedes_unicas_clean = df_consumos['sede_id'].unique()
    print(f"Sedes encontradas (Corregido): {sedes_unicas_clean}")
    print("-" * 30)

    datos_por_sede = {}

    for sede in sedes_unicas_clean:
        print(f"\n--- Procesando Sede: {sede} ---")
        
        # Filtro por sede
        df_sede = df_consumos[df_consumos['sede_id'] == sede].copy()
        datos_por_sede[sede] = df_sede
        
        # Mostrar resumen de datos para esta sede
        print(f"Registros encontrados: {len(df_sede)}")
        print("Primeras 5 filas:")
        print(df_sede.head())
        print("-" * 30)

    return datos_por_sede

def paso_2_limpieza_contextual(datos_por_sede):
    """
    PASO 2: Partición y Auditoría Forense de Datos (Trap Detector)
    Objetivo: Generar sub-datasets por sede y aplicar política 'Zero Trust' con flags de calidad.
    """
    print("\n" + "="*80)
    print(" EJECUTANDO PASO 2: AUDITORÍA FORENSE DE DATOS (THE TRAP DETECTOR)")
    print("="*80)

    datos_curados = {}

    for sede, df in datos_por_sede.items():
        print(f"\n>>> Procesando Sede: {sede}")
        # Ordenar por tiempo para asegurar integridad temporal
        if 'fecha_hora' in df.columns:
             df['fecha_hora'] = pd.to_datetime(df['fecha_hora'])
             df = df.sort_values('fecha_hora').reset_index(drop=True)
        
        df_clean = df.copy()
        
        # 1. Inicializar columna de auditoría (qa_flags)
        # Usaremos una lista para acumular flags y luego uniremos con pipe '|'
        df_clean['qa_flags'] = ''
        
        # =========================================================================
        # REGLA A: Auditoría de Metadatos ("Trampa de Flickering")
        # Detecta cambios en periodo_academico < 24 horas
        # =========================================================================
        print("   -> (A) Ejecutando Trampa de Flickering...")
        if 'periodo_academico' in df_clean.columns and 'fecha_hora' in df_clean.columns:
            # Crear grupos consecutivos de periodo_academico
            df_clean['grp_periodo'] = (df_clean['periodo_academico'] != df_clean['periodo_academico'].shift()).cumsum()
            
            # Calcular duración de cada grupo
            grp_duracion = df_clean.groupby('grp_periodo')['fecha_hora'].agg(['min', 'max'])
            grp_duracion['duracion_horas'] = (grp_duracion['max'] - grp_duracion['min']).dt.total_seconds() / 3600
            
            # Identificar grupos con duración < 24h
            grps_flickering = grp_duracion[grp_duracion['duracion_horas'] < 24].index
            
            mask_flickering = df_clean['grp_periodo'].isin(grps_flickering)
            df_clean.loc[mask_flickering, 'qa_flags'] = df_clean.loc[mask_flickering, 'qa_flags'].apply(
                lambda x: x + 'FLICKERING_METADATA|' if x == '' else x + 'FLICKERING_METADATA|'
            )
            
            # Limpieza auxiliar
            df_clean.drop(columns=['grp_periodo'], inplace=True)

        # =========================================================================
        # REGLA B: Auditoría Física ("Regla de la Suma")
        # Delta > 5% del Total
        # =========================================================================
        print("   -> (B) Verificando Primera Ley de la Termodinámica...")
        cols_subsectores = ['energia_laboratorios_kwh', 'energia_salones_kwh', 
                            'energia_oficinas_kwh', 'energia_comedor_kwh', 'energia_auditorios_kwh']
        
        # Verificar existencia de columnas
        cols_presentes = [c for c in cols_subsectores if c in df_clean.columns]
        
        if 'energia_total_kwh' in df_clean.columns and cols_presentes:
            suma_sub = df_clean[cols_presentes].sum(axis=1)
            delta = abs(df_clean['energia_total_kwh'] - suma_sub)
            delta_pct = (delta / df_clean['energia_total_kwh']).fillna(0) # Evitar div por 0
            
            mask_fisica = delta_pct > 0.05 # 5% tolerancia
            
            # Solo marcar si la energia total no es insignificante (ej > 1 kWh) para evitar ruido en ceros
            mask_fisica = mask_fisica & (df_clean['energia_total_kwh'] > 1)
            
            df_clean.loc[mask_fisica, 'qa_flags'] = df_clean.loc[mask_fisica, 'qa_flags'].apply(
                lambda x: x + 'INCONSISTENCIA_SUMA|' if x == '' else x + 'INCONSISTENCIA_SUMA|'
            )

        # =========================================================================
        # REGLA C: Auditoría de Integridad Temporal (Gaps)
        # =========================================================================
        print("   -> (C) Buscando Gaps Temporales...")
        if 'fecha_hora' in df_clean.columns:
            # Crear rango completo esperado
            min_date = df_clean['fecha_hora'].min()
            max_date = df_clean['fecha_hora'].max()
            if pd.notnull(min_date) and pd.notnull(max_date):
                full_range = pd.date_range(start=min_date, end=max_date, freq='h')
                
                # Reindexar para exponer gaps
                df_clean = df_clean.set_index('fecha_hora').reindex(full_range)
                
                # Los nuevos índices creados tendrán NaNs en todas las columnas. Marcar.
                mask_gap = df_clean['energia_total_kwh'].isna() # Usamos una columna clave para detectar
                
                # Reset index para recuperar fecha_hora como columna
                df_clean = df_clean.reset_index().rename(columns={'index': 'fecha_hora'})
                
                # Marcar Gaps
                # Nota: Al reindexar, la columna qa_flags será NaN en las nuevas filas.
                # Inicializamos qa_flags en esas filas como '' antes de agregar el flag
                df_clean.loc[mask_gap, 'qa_flags'] = ''
                df_clean.loc[mask_gap, 'qa_flags'] = df_clean.loc[mask_gap, 'qa_flags'].apply(
                    lambda x: x + 'MISSING_TIMESTAMP|'
                )

        # =========================================================================
        # REGLA D: Valores Imposibles
        # =========================================================================
        print("   -> (D) Detectando Valores Negativos...")
        cols_numericas = df_clean.select_dtypes(include=['float64', 'int64']).columns
        # Excluir lat, lon, id, y otras que pueden ser validamente negativas o no son consumo
        cols_consumo = [c for c in cols_numericas if 'energia' in c or 'agua' in c]
        
        mask_neg = pd.Series(False, index=df_clean.index)
        for col in cols_consumo:
            mask_neg = mask_neg | (df_clean[col] < 0)
            
        df_clean.loc[mask_neg, 'qa_flags'] = df_clean.loc[mask_neg, 'qa_flags'].apply(
             lambda x: x + 'VALOR_NEGATIVO|' if pd.notnull(x) else 'VALOR_NEGATIVO|'
        )

        # =========================================================================
        # HEURÍSTICAS (System Prompt)
        # =========================================================================
        print("   -> Ejecutando Heurísticas auto-descubiertas...")
        
        # 1. Flatlines (Ceros constantes > 48h)
        # Simplificación: Rolling sum sobre 48h. Si es 0 y el total acumulado no debería ser 0...
        # Mejor enfoque: Si la desviación estándar en ventana movil de 48h es 0.
        if 'energia_total_kwh' in df_clean.columns:
             # Rellenar NaNs temporalmente para el cálculo (los gaps ya tienen flag)
             temp_series = df_clean['energia_total_kwh'].fillna(0)
             rolling_std = temp_series.rolling(window=48).std()
             
             # Detectar donde std es 0 (o muy cercano) y el valor promedio es bajo/cero? 
             # La regla dice "Columnas enteras de ceros". O sea valor == 0.
             rolling_sum = temp_series.rolling(window=48).sum()
             mask_flatline = (rolling_sum == 0) & (temp_series == 0) # Ventana de 48h de ceros antecedente
             
             df_clean.loc[mask_flatline, 'qa_flags'] = df_clean.loc[mask_flatline, 'qa_flags'].apply(
                 lambda x: str(x) + 'POSIBLE_FLATLINE|' if pd.notnull(x) else 'POSIBLE_FLATLINE|'
             )

        # 2. Spikes (> 3 std dev)
        # Z-score robusto o simple sobre ventana móvil
        if 'energia_total_kwh' in df_clean.columns:
             temp_series = df_clean['energia_total_kwh'].ffill().bfill()
             # Usamos ventana amplia para la media base (ej. 1 semana)
             roll_mean = temp_series.rolling(window=24*7, min_periods=24).mean()
             roll_std = temp_series.rolling(window=24*7, min_periods=24).std()
             
             # Evitar division por cero
             z_score = (temp_series - roll_mean) / (roll_std + 1e-5) # epsilon
             
             mask_spike = z_score > 3
             df_clean.loc[mask_spike, 'qa_flags'] = df_clean.loc[mask_spike, 'qa_flags'].apply(
                 lambda x: str(x) + 'POSIBLE_SPIKE|' if pd.notnull(x) else 'POSIBLE_SPIKE|'
             )

        # 3. Limpieza de Flags
        # Eliminar el pipe final si existe y reemplazar vacios con 'OK' o dejar vacio
        df_clean['qa_flags'] = df_clean['qa_flags'].astype(str).str.rstrip('|')
        
        # =========================================================================
        # Reporte y Guardado
        # =========================================================================
        conteo_flags = df_clean['qa_flags'].replace('', 'OK').value_counts().head(10)
        print("\n   --- Top Flags Detectados ---")
        print(conteo_flags)
        
        # NOTA: Ya no se guarda CSV intermedio, se mantiene en memoria
        print(f"   -> Datos curados en memoria para: {sede}")

        # Guardar en diccionario de retorno
        datos_curados[sede] = df_clean

    return datos_curados

def paso_3_saneamiento(datasets_auditados):
    """
    PASO 3: Saneamiento de Metadatos (The Time Lord Protocol)
    Corrige festivos, dias y flickering de periodos.
    """
    print("\n>>> INICIANDO PASO 3: Saneamiento de Metadatos...")
    
    # 1. Instanciar Festivos Colombia una sola vez (Años del dataset)
    # Se asume rango 2018-2026 para cubrir todo
    co_holidays = holidays.CO(years=range(2018, 2027))
    
    datos_saneados = {}

    for sede, df in datasets_auditados.items():
        print(f"   Procesando saneamiento para: {sede}")
        df_san = df.copy()
        
        # Asegurar datetime
        df_san['timestamp'] = pd.to_datetime(df_san['timestamp'])
        
        # ---------------------------------------------------------
        # A. ESTANDARIZACIÓN LÉXICA (Snake Case)
        # ---------------------------------------------------------
        cols_texto = ['periodo_academico', 'dia_nombre']
        for col in cols_texto:
            if col in df_san.columns:
                df_san[col] = df_san[col].astype(str).str.lower().str.strip().str.replace(' ', '_')

        # ---------------------------------------------------------
        # B. RECONSTRUCCIÓN DE LA VERDAD TEMPORAL
        # ---------------------------------------------------------
        # 1. Recalcular Dia Semana (0=Lunes, 6=Domingo)
        df_san['dia_semana'] = df_san['timestamp'].dt.dayofweek
        
        # 2. Recalcular Nombre Día
        dias_map = {0:'lunes', 1:'martes', 2:'miercoles', 3:'jueves', 4:'viernes', 5:'sabado', 6:'domingo'}
        df_san['dia_nombre'] = df_san['dia_semana'].map(dias_map)
        
        # 3. Recalcular Festivos (Lógica Vectorizada)
        # Primero, extraemos solo la fecha (YYYY-MM-DD) para comparar
        fechas_unicas = df_san['timestamp'].dt.date
        
        # Verificar si la fecha está en la lista de festivos
        es_festivo_col = fechas_unicas.isin(co_holidays)
        
        # Regla: Es festivo si está en holidays O si es Domingo (día 6)
        df_san['es_festivo'] = es_festivo_col | (df_san['dia_semana'] == 6)
        
        # Recalcular Fin de Semana (Sábado=5, Domingo=6)
        df_san['es_fin_semana'] = df_san['dia_semana'] >= 5

        # ---------------------------------------------------------
        # C. ESTABILIZACIÓN PERIODO (Anti-Flickering por DÍA EXACTO)
        # ---------------------------------------------------------
        # Agrupamos por la fecha exacta (Date) para sacar la moda de ESE día específico
        # Transform calcula la moda de ese grupo y la replica en todas las filas del grupo
        
        def get_mode(x):
            m = pd.Series.mode(x)
            return m[0] if not m.empty else np.nan

        # Creamos columna auxiliar de fecha
        df_san['fecha_solo'] = df_san['timestamp'].dt.date
        
        # Aplicamos corrección
        df_san['periodo_academico'] = df_san.groupby('fecha_solo')['periodo_academico'].transform(get_mode)
        
        # Eliminar columna auxiliar
        df_san.drop(columns=['fecha_solo'], inplace=True)

        # ---------------------------------------------------------
        # D. IMPUTACIÓN DE HUECOS (GAPS)
        # ---------------------------------------------------------
        # Rellenar metadatos faltantes por vecindad (limitado a 24h)
        cols_rellenar = ['periodo_academico', 'es_semana_parciales', 'es_semana_finales']
        # Solo rellenar si existen las columnas
        cols_existentes = [c for c in cols_rellenar if c in df_san.columns]
        
        if cols_existentes:
            df_san[cols_existentes] = df_san[cols_existentes].ffill(limit=24).bfill(limit=24)

        # ---------------------------------------------------------
        # NOTA: El CSV se genera al final del Paso 4
        # ---------------------------------------------------------
        print(f"   -> Datos saneados en memoria para: {sede}")
        
        datos_saneados[sede] = df_san

    return datos_saneados


def paso_4_deteccion_contextual_outliers(datasets_saneados):
    """
    PASO 4: Detección Contextual de Outliers (Z-Score por Contexto)
    
    Aplica reglas de negocio específicas por sede y contexto operacional:
    - Segmentación en "Cubos de Comportamiento"
    - Z-Score contextual con N variable
    - Reglas físicas por tipo de sede
    - Validación de TODAS las columnas numéricas relevantes
    - Detección de valores negativos y extremos (>1 millón)
    
    Outliers se convierten a NaN para posterior imputación.
    Se genera columna explicativa con motivo de eliminación.
    """
    print("\n" + "="*80)
    print(" PASO 4: DETECCIÓN CONTEXTUAL DE OUTLIERS (Z-Score por Contexto)")
    print("="*80)
    
    # =========================================================================
    # CONFIGURACIÓN DE SEDES (Reglas específicas por campus)
    # =========================================================================
    
    CONFIGURACION_SEDES = {
        'UPTC_TUN': {
            'descripcion': 'Residencial + Comedor Masivo',
            'limites_max': {
                'energia_total_kwh': 3500,       # Capacidad alta por residencias
                'energia_comedor_kwh': 800,      # Hornos/Marmitas (Picos altos permitidos)
                'energia_laboratorios_kwh': 500, # Labs normales
                'energia_salones_kwh': 500,
                'energia_auditorios_kwh': 300,
                'energia_oficinas_kwh': 300,
                'agua_litros': 50000,            # Consumo alto residencial
                'ocupacion_pct': 100
            },
            'reglas_especiales': {
                'permitir_cero_clases': False,   # EN CLASES: Residencias activas, NO ceros
                'permitir_cero_vacaciones': True, # EN VACACIONES: Sin estudiantes, SÍ ceros
                'borrar_negativos': True
            }
        },
        'UPTC_SOG': {
            'descripcion': 'Industrial Pesado',
            'limites_max': {
                'energia_total_kwh': 3000,       # Maquinaria pesada
                'energia_comedor_kwh': 200,      # Cafetería pequeña
                'energia_laboratorios_kwh': 1500,# MOTORES Y HORNOS (Prioridad)
                'energia_salones_kwh': 400,
                'energia_auditorios_kwh': 200,
                'energia_oficinas_kwh': 200,
                'agua_litros': 20000,
                'ocupacion_pct': 100
            },
            'reglas_especiales': {
                'permitir_cero_total': True,     # Pueden apagar todo en vacaciones
                'borrar_negativos': True
            }
        },
        'UPTC_DUI': {
            'descripcion': 'Industrial / Técnico',
            'limites_max': {
                'energia_total_kwh': 3000,
                'energia_comedor_kwh': 200,
                'energia_laboratorios_kwh': 1200,# Maquinaria técnica
                'energia_salones_kwh': 400,
                'energia_auditorios_kwh': 200,
                'energia_oficinas_kwh': 200,
                'agua_litros': 20000,
                'ocupacion_pct': 100
            },
            'reglas_especiales': {
                'permitir_cero_total': True,
                'borrar_negativos': True
            }
        },
        'UPTC_CHI': {
            'descripcion': 'Académico / Administrativo',
            'limites_max': {
                'energia_total_kwh': 1000,       # Bajo consumo (Oficinas/Aulas)
                'energia_comedor_kwh': 100,      # Cafetería básica
                'energia_laboratorios_kwh': 300, # Labs de computo/básicos
                'energia_salones_kwh': 300,
                'energia_auditorios_kwh': 150,
                'energia_oficinas_kwh': 200,
                'agua_litros': 10000,
                'ocupacion_pct': 100
            },
            'reglas_especiales': {
                'permitir_cero_total': True,
                'borrar_negativos': True
            }
        }
    }
    
    # =========================================================================
    # LÍMITES GLOBALES (Aplican a todos si no se especifican arriba)
    # =========================================================================
    LIMITES_GLOBALES = {
        'temperatura_exterior_c': {'min': -5, 'max': 45},  # Clima Colombia/Boyacá
        'potencia_total_kw': 5000,
        'co2_kg': 1000
    }
    
    # Columnas a validar
    COLUMNAS_VALIDAR = [
        'energia_total_kwh', 'energia_comedor_kwh', 'energia_salones_kwh',
        'energia_laboratorios_kwh', 'energia_auditorios_kwh', 'energia_oficinas_kwh',
        'potencia_total_kw', 'agua_litros', 'temperatura_exterior_c', 'ocupacion_pct','co2_kg' 
    ]
    
    # Columnas que NO pueden ser negativas (todas excepto temperatura)
    COLUMNAS_NO_NEGATIVAS = [
        'energia_total_kwh', 'energia_comedor_kwh', 'energia_salones_kwh',
        'energia_laboratorios_kwh', 'energia_auditorios_kwh', 'energia_oficinas_kwh',
        'potencia_total_kw', 'agua_litros', 'ocupacion_pct',
        'co2_kg'  # Agregado: emisiones CO2 nunca pueden ser negativas
    ]
    
    # Temperatura minima (Colombia/Boyaca)
    TEMP_MIN = LIMITES_GLOBALES['temperatura_exterior_c']['min']
    
    datasets_preprocesados = {}
    
    for sede, df in datasets_saneados.items():
        print(f"\n>>> Procesando: {sede}")
        df_proc = df.copy()
        
        # Asegurar columnas necesarias
        df_proc['timestamp'] = pd.to_datetime(df_proc['timestamp'])
        df_proc['hora'] = df_proc['timestamp'].dt.hour
        
        # Obtener configuracion de esta sede
        config_sede = CONFIGURACION_SEDES.get(sede, {
            'descripcion': 'Generico',
            'limites_max': {},
            'reglas_especiales': {'permitir_cero_total': True, 'borrar_negativos': True}
        })
        
        limites_sede = config_sede.get('limites_max', {})
        reglas_esp = config_sede.get('reglas_especiales', {})
        
        print(f"   Tipo: {config_sede.get('descripcion', 'Generico')}")
        
        # =================================================================
        # INICIALIZAR COLUMNA DE MOTIVO DE ELIMINACIÓN
        # =================================================================
        df_proc['motivo_eliminacion'] = ''
        
        # =================================================================
        # DETECCIÓN DE OUTLIERS EN TODAS LAS COLUMNAS
        # =================================================================
        print("   -> Detectando valores atípicos en todas las columnas...")
        
        # Contadores de outliers por tipo
        conteo_outliers = {
            'negativos': {},
            'extremos_altos': {},
            'extremos_millones': {},
            'temp_extrema': 0
        }
        
        # Columnas presentes en el dataset
        cols_presentes = [c for c in COLUMNAS_VALIDAR if c in df_proc.columns]
        print(f"      Columnas a validar: {cols_presentes}")
        
        # -----------------------------------------------------------------
        # REGLA A: VALORES NEGATIVOS (Física Imposible)
        # -----------------------------------------------------------------
        print("   -> [A] Detectando valores negativos...")
        
        for col in cols_presentes:
            if col in COLUMNAS_NO_NEGATIVAS:
                mask_negativo = df_proc[col] < 0
                n_negativos = mask_negativo.sum()
                
                if n_negativos > 0:
                    conteo_outliers['negativos'][col] = n_negativos
                    
                    # Marcar motivo y convertir a NaN
                    df_proc.loc[mask_negativo, 'motivo_eliminacion'] = df_proc.loc[mask_negativo, 'motivo_eliminacion'].apply(
                        lambda x: x + f'{col}:NEGATIVO; ' if x else f'{col}:NEGATIVO; '
                    )
                    df_proc.loc[mask_negativo, col] = np.nan
                    df_proc.loc[mask_negativo, 'qa_flags'] = df_proc.loc[mask_negativo, 'qa_flags'].astype(str) + f'NEGATIVO_{col.upper()}|'
        
        # Caso especial: Temperatura (puede ser negativa pero no extrema)
        if 'temperatura_exterior_c' in df_proc.columns:
            mask_temp_baja = df_proc['temperatura_exterior_c'] < TEMP_MIN
            n_temp_baja = mask_temp_baja.sum()
            
            if n_temp_baja > 0:
                conteo_outliers['temp_extrema'] = n_temp_baja
                df_proc.loc[mask_temp_baja, 'motivo_eliminacion'] = df_proc.loc[mask_temp_baja, 'motivo_eliminacion'].apply(
                    lambda x: x + f'temperatura_exterior_c:TEMP_EXTREMA_BAJA(<{TEMP_MIN}°C); ' if x else f'temperatura_exterior_c:TEMP_EXTREMA_BAJA(<{TEMP_MIN}°C); '
                )
                df_proc.loc[mask_temp_baja, 'temperatura_exterior_c'] = np.nan
                df_proc.loc[mask_temp_baja, 'qa_flags'] = df_proc.loc[mask_temp_baja, 'qa_flags'].astype(str) + 'TEMP_EXTREMA_BAJA|'
        
        # -----------------------------------------------------------------
        # REGLA B: VALORES EXTREMOS (Órdenes de Millones = Error de Sensor)
        # -----------------------------------------------------------------
        print("   -> [B] Detectando valores extremos (millones)...")
        
        UMBRAL_MILLONES = 1_000_000  # Cualquier valor > 1 millón es claramente erróneo
        
        for col in cols_presentes:
            mask_millones = df_proc[col].abs() > UMBRAL_MILLONES
            n_millones = mask_millones.sum()
            
            if n_millones > 0:
                conteo_outliers['extremos_millones'][col] = n_millones
                
                # Guardar valores para el reporte (antes de eliminar)
                valores_erroneos = df_proc.loc[mask_millones, col].head(3).tolist()
                
                df_proc.loc[mask_millones, 'motivo_eliminacion'] = df_proc.loc[mask_millones, 'motivo_eliminacion'].apply(
                    lambda x: x + f'{col}:VALOR_MILLON(>{UMBRAL_MILLONES}); ' if x else f'{col}:VALOR_MILLON(>{UMBRAL_MILLONES}); '
                )
                df_proc.loc[mask_millones, col] = np.nan
                df_proc.loc[mask_millones, 'qa_flags'] = df_proc.loc[mask_millones, 'qa_flags'].astype(str) + f'EXTREMO_MILLON_{col.upper()}|'
                
                print(f"      [WARN] {col}: {n_millones} valores > 1M (ej: {valores_erroneos})")
        
        # -----------------------------------------------------------------
        # REGLA C: LIMITES FISICOS MAXIMOS POR SEDE
        # -----------------------------------------------------------------
        print("   -> [C] Validando limites fisicos maximos por sede...")
        
        for col in cols_presentes:
            # Buscar limite en configuracion de sede, luego en globales
            limite = limites_sede.get(col, LIMITES_GLOBALES.get(col))
            
            if limite is not None:
                # Si es dict (como temperatura), tomar max
                if isinstance(limite, dict):
                    limite = limite.get('max')
                
                if limite is not None:
                    mask_excede = df_proc[col] > limite
                    n_excede = mask_excede.sum()
                    
                    if n_excede > 0:
                        conteo_outliers['extremos_altos'][col] = n_excede
                        
                        df_proc.loc[mask_excede, 'motivo_eliminacion'] = df_proc.loc[mask_excede, 'motivo_eliminacion'].apply(
                            lambda x: x + f'{col}:EXCEDE_LIMITE(>{limite}); ' if x else f'{col}:EXCEDE_LIMITE(>{limite}); '
                        )
                        df_proc.loc[mask_excede, col] = np.nan
                        df_proc.loc[mask_excede, 'qa_flags'] = df_proc.loc[mask_excede, 'qa_flags'].astype(str) + f'EXCEDE_{col.upper()}|'
        
        # -----------------------------------------------------------------
        # REGLA D: CEROS EN CONTEXTO (Vacaciones vs Clases)
        # -----------------------------------------------------------------
        # REGLA CLAVE:
        # - En VACACIONES: NO hay estudiantes en NINGUNA sede (incluida TUNJA)
        #   por lo tanto, ceros son NORMALES y NO se eliminan
        # - En CLASES: Ceros son sospechosos, especialmente en UPTC_TUN 
        #   donde las residencias estudiantiles están activas
        # 
        # UPTC_TUN específico:
        #   - VACACIONES: Ceros permitidos (sin estudiantes en residencias)
        #   - CLASES: Ceros ELIMINADOS (residencias ocupadas 24/7)
        
        print("   -> [D] Analizando ceros en contexto academico...")
        
        if 'energia_total_kwh' in df_proc.columns and 'periodo_academico' in df_proc.columns:
            # Detectar periodo academico
            es_vacaciones = df_proc['periodo_academico'].astype(str).str.lower().str.contains(
                'vacacion|receso|intersemestral', na=False
            )
            es_clases = df_proc['periodo_academico'].astype(str).str.lower().str.contains(
                'semestre|clases|parciales|finales', na=False
            )
            
            mask_cero = df_proc['energia_total_kwh'] == 0
            
            # Contar ceros por contexto
            n_ceros_vacaciones = (mask_cero & es_vacaciones).sum()
            n_ceros_clases = (mask_cero & es_clases).sum()
            
            print(f"      Ceros en VACACIONES: {n_ceros_vacaciones:,} (SIN estudiantes -> normales)")
            print(f"      Ceros en CLASES: {n_ceros_clases:,}")
            
            # =====================================================================
            # LÓGICA ESPECÍFICA POR SEDE
            # =====================================================================
            
            if sede == 'UPTC_TUN':
                # UPTC_TUN tiene residencias estudiantiles
                permitir_cero_clases = reglas_esp.get('permitir_cero_clases', False)
                permitir_cero_vacaciones = reglas_esp.get('permitir_cero_vacaciones', True)
                
                # EN CLASES: NO permitir ceros (residencias activas)
                if not permitir_cero_clases:
                    mask_cero_clases = mask_cero & es_clases
                    n_ceros_eliminar = mask_cero_clases.sum()
                    
                    if n_ceros_eliminar > 0:
                        df_proc.loc[mask_cero_clases, 'motivo_eliminacion'] = df_proc.loc[mask_cero_clases, 'motivo_eliminacion'].apply(
                            lambda x: x + 'energia_total_kwh:CERO_EN_CLASES(Residencias activas); ' if x else 'energia_total_kwh:CERO_EN_CLASES(Residencias activas); '
                        )
                        df_proc.loc[mask_cero_clases, 'energia_total_kwh'] = np.nan
                        df_proc.loc[mask_cero_clases, 'qa_flags'] = df_proc.loc[mask_cero_clases, 'qa_flags'].astype(str) + 'CERO_CLASES|'
                        print(f"      [UPTC_TUN] {n_ceros_eliminar} ceros en CLASES -> ELIMINADOS (residencias activas)")
                
                # EN VACACIONES: Permitir ceros (sin estudiantes)
                if permitir_cero_vacaciones:
                    mask_cero_vac = mask_cero & es_vacaciones
                    n_ceros_vac = mask_cero_vac.sum()
                    if n_ceros_vac > 0:
                        print(f"      [UPTC_TUN] {n_ceros_vac} ceros en VACACIONES -> PERMITIDOS (sin estudiantes)")
                        # Solo marcar informativamente, NO eliminar
                        df_proc.loc[mask_cero_vac, 'qa_flags'] = df_proc.loc[mask_cero_vac, 'qa_flags'].astype(str) + 'CERO_VACACIONES_OK|'
            
            else:
                # OTRAS SEDES (SOG, DUI, CHI): No tienen residencias
                # Ceros en vacaciones son completamente normales
                mask_cero_vac = mask_cero & es_vacaciones
                n_ceros_vac = mask_cero_vac.sum()
                if n_ceros_vac > 0:
                    print(f"      [{sede}] {n_ceros_vac} ceros en VACACIONES -> PERMITIDOS (sin actividad)")
                
                # Ceros en clases: solo marcar para revisión (no eliminar automáticamente)
                mask_cero_clases = mask_cero & es_clases
                n_ceros_clases_sede = mask_cero_clases.sum()
                if n_ceros_clases_sede > 0:
                    df_proc.loc[mask_cero_clases, 'qa_flags'] = df_proc.loc[mask_cero_clases, 'qa_flags'].astype(str) + 'CERO_CLASES_REVISAR|'
                    print(f"      [{sede}] {n_ceros_clases_sede} ceros en CLASES -> MARCADOS para revisión")
        
        # =================================================================
        # REPORTE DETALLADO
        # =================================================================
        print(f"\n   {'='*60}")
        print(f"   RESUMEN DE OUTLIERS DETECTADOS - {sede}")
        print(f"   {'='*60}")
        
        print("\n   [VALORES NEGATIVOS]")
        if conteo_outliers['negativos']:
            for col, n in conteo_outliers['negativos'].items():
                print(f"      - {col}: {n:,} valores negativos eliminados")
        else:
            print("      [OK] Sin valores negativos detectados")
        
        print("\n   [VALORES EXTREMOS (>1 MILLON)]")
        if conteo_outliers['extremos_millones']:
            for col, n in conteo_outliers['extremos_millones'].items():
                print(f"      - {col}: {n:,} valores > 1M eliminados")
        else:
            print("      [OK] Sin valores extremos (millones) detectados")
        
        print("\n   [EXCEDEN LIMITES FISICOS]")
        if conteo_outliers['extremos_altos']:
            for col, n in conteo_outliers['extremos_altos'].items():
                limite = limites_sede.get(col, LIMITES_GLOBALES.get(col, 'N/A'))
                if isinstance(limite, dict):
                    limite = limite.get('max', 'N/A')
                print(f"      - {col}: {n:,} valores > {limite}")
        else:
            print("      [OK] Sin valores que excedan limites fisicos")
        
        if conteo_outliers['temp_extrema'] > 0:
            print(f"\n   [TEMPERATURA EXTREMA]")
            print(f"      - temperatura_exterior_c: {conteo_outliers['temp_extrema']:,} valores < {TEMP_MIN}C")
        
        # Total filas afectadas
        filas_afectadas = (df_proc['motivo_eliminacion'] != '').sum()
        pct_afectadas = (filas_afectadas / len(df_proc)) * 100
        print(f"\n   TOTAL FILAS AFECTADAS: {filas_afectadas:,} ({pct_afectadas:.2f}%)")
        
        # -----------------------------------------------------------------
        # LIMPIEZA FINAL
        # -----------------------------------------------------------------
        # Limpiar columnas auxiliares
        cols_aux = ['mu', 'sigma', 'mediana', 'n_sigma', 'limite_sup']
        df_proc.drop(columns=cols_aux, inplace=True, errors='ignore')
        
        # Limpiar qa_flags y motivo_eliminacion
        df_proc['qa_flags'] = df_proc['qa_flags'].astype(str).str.rstrip('|')
        df_proc['motivo_eliminacion'] = df_proc['motivo_eliminacion'].str.rstrip('; ')
        
        # Si no hay motivo, dejar vacío (o podrías poner 'OK')
        df_proc['motivo_eliminacion'] = df_proc['motivo_eliminacion'].replace('', np.nan)
        
        # Guardar CSV
        nombre_archivo = f"preprocesado_{sede}.csv"
        path_archivo = os.path.join(OUTPUT_DIR, nombre_archivo)
        df_proc.to_csv(path_archivo, index=False)
        print(f"\n   -> CSV FINAL: {nombre_archivo}")
        
        datasets_preprocesados[sede] = df_proc
    
    return datasets_preprocesados


# =========================================================================
# MAIN - Ejecución del Pipeline
# =========================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("   ALGORITMO UPTC - Pipeline de Preprocesamiento de Datos")
    print("   Detección Contextual de Outliers con Z-Score por Sede")
    print("="*70)
    
    # 1. Cargar Datos
    df_main, df_meta = cargar_datos()
    
    if df_main is not None:
        # 2. Paso 1: Partición Física por Sede
        datasets_sedes = paso_1_particion_fisica(df_main, df_meta)
        
        # 3. Paso 2: Auditoría Forense (Flags de calidad)
        datasets_sedes_limpios = paso_2_limpieza_contextual(datasets_sedes)
        
        # 4. Paso 3: Saneamiento de Metadatos (Fechas, Festivos, Escritura)
        datasets_sedes_saneados = paso_3_saneamiento(datasets_sedes_limpios)
        
        # 5. Paso 4: Detección Contextual de Outliers (Z-Score)
        # Aplica reglas de negocio por sede y convierte outliers a NaN
        datasets_preprocesados = paso_4_deteccion_contextual_outliers(datasets_sedes_saneados)
        
        # -----------------------------------------------------------------
        # RESUMEN FINAL
        # -----------------------------------------------------------------
        print("\n" + "="*70)
        print("   PROCESO COMPLETADO")
        print("="*70)
        print("\n   Archivos generados:")
        for sede in datasets_preprocesados.keys():
            print(f"      - preprocesado_{sede}.csv")
        
        print("\n   Reglas aplicadas por sede:")
        print("      UPTC_TUN:  Residencial + Comedor | Ceros en CLASES eliminados | Limites: energia=3500, agua=50000")
        print("      UPTC_SOG:  Industrial Pesado    | Ceros permitidos vacaciones | Limites: energia=3000, labs=1500")
        print("      UPTC_DUI:  Industrial/Tecnico   | Ceros permitidos vacaciones | Limites: energia=3000, labs=1200")
        print("      UPTC_CHI:  Academico/Admin      | Ceros permitidos vacaciones | Limites: energia=1000, labs=300")
        print("\n   Reglas globales aplicadas:")
        print("      - Valores negativos -> NaN")
        print("      - Valores > 1,000,000 -> NaN (errores de sensor)")
        print("      - Valores que exceden limites fisicos por sede -> NaN")
        
        print("\n   Outliers convertidos a NaN para posterior imputación.")
        print("="*70)