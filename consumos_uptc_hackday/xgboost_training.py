# -*- coding: utf-8 -*-
"""
PASO 6 v3: Entrenamiento XGBoost con VARIABLES MÁGICAS + VALIDACIÓN ANTI-OVERFITTING
====================================================================================
Mejoras sobre v2:
1. Variables de INERCIA (lags de temperatura y ocupación)
2. Variables de VELOCIDAD DE CAMBIO (diff de temperatura y consumo)
3. RandomizedSearchCV con TimeSeriesSplit para evitar overfitting
4. Comparación Train vs Test para detectar memorización
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import json
import sys
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Fix encoding para Windows
sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# CONFIGURACIÓN
# ==========================================
BASE_DIR = r"c:\Users\POWER\OneDrive\Escritorio\consumos_uptc_hackday"
INPUT_DIR = os.path.join(BASE_DIR, "DATASETS_ENTRENAMIENTO_LISTOS")
MODEL_DIR = os.path.join(BASE_DIR, "MODELOS_XGBOOST_V3")
RESULTS_DIR = os.path.join(BASE_DIR, "RESULTADOS_ENTRENAMIENTO_V3")

for d in [MODEL_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Archivos de entrada
ARCHIVOS_SEDES = {
    'UPTC_TUN': 'train_ready_UPTC_TUN.csv',
    'UPTC_SOG': 'train_ready_UPTC_SOG.csv',
    'UPTC_DUI': 'train_ready_UPTC_DUI.csv',
    'UPTC_CHI': 'train_ready_UPTC_CHI.csv',
}

# ==========================================
# DEFINICIÓN DEL VECTOR DE ENTRADA (X)
# ==========================================

# A. Temporales Cíclicas - El "Reloj Matemático"
FEATURES_CICLICAS = [
    'hora_sin', 'hora_cos',           # Ciclo diario (0-24h)
    'dia_sem_sin', 'dia_sem_cos',     # Ciclo semanal (Lun-Dom)
    'mes_sin', 'mes_cos',             # Ciclo anual (Ene-Dic)
]

# B. Calendario Académico - Contexto Operativo
FEATURES_CALENDARIO = [
    'es_fin_semana',                  # 0/1 - Apaga labs, baja ocupación
    'es_festivo',                     # 0/1 - Apagado general (Baseload)
    'periodo_academico_semestre_1',   # 0/1 - Operación normal
    'periodo_academico_semestre_2',   # 0/1 - Operación normal
]

# C. Variables Exógenas (Físicas)
FEATURES_EXOGENAS = [
    'temperatura_exterior_c',         # Afecta calentadores y eficiencia
    'ocupacion_pct',                  # Proporcional a uso de agua y luz
]

# D. Memoria Histórica - CRÍTICO PARA XGBOOST
LAGS = [1, 24, 168]  # 1h, 24h (ayer misma hora), 168h (semana pasada mismo día/hora)
ROLLING_WINDOWS = [24]  # Promedio últimas 24h

# E. 🆕 VARIABLES MÁGICAS - Inercia y Velocidad de Cambio
# Estas se generan dinámicamente

# TARGETS a predecir
TARGETS = [
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

# ==========================================
# HIPERPARÁMETROS - ESPACIO DE BÚSQUEDA
# ==========================================

PARAM_DIST = {
    'n_estimators': [300, 500, 800, 1000],      # ¿Cuántos árboles?
    'max_depth': [3, 5, 7, 9],                   # ¿Complejidad de preguntas?
    'learning_rate': [0.01, 0.03, 0.05, 0.1],   # ¿Velocidad de aprendizaje?
    'subsample': [0.7, 0.8, 0.9],               # % datos por árbol (anti-overfit)
    'colsample_bytree': [0.7, 0.8, 0.9],        # % features por árbol
    'min_child_weight': [3, 5, 7],              # Mínimo por hoja (regularización)
    'reg_alpha': [0, 0.01, 0.1, 1],             # L1 regularization
    'reg_lambda': [1, 1.5, 2, 3],               # L2 regularization
}

# Configuración de validación cruzada
N_ITER_SEARCH = 15           # Combinaciones a probar (balance tiempo/exploración)
N_SPLITS_CV = 5              # Particiones temporales para TimeSeriesSplit
TEST_SIZE = 0.2              # Holdout final

# Umbral para detectar overfitting
OVERFITTING_THRESHOLD = 0.10  # Si R2_train - R2_test > 0.10, hay overfitting

# ==========================================
# FUNCIONES DE FEATURE ENGINEERING
# ==========================================

def agregar_variables_magicas(df):
    """
    Agrega variables de INERCIA y VELOCIDAD DE CAMBIO.
    
    INERCIA: El efecto retardado
    - La temperatura de hace 1h afecta el consumo de AHORA
    - La ocupación de hace 1h indica si la gente está llegando
    
    VELOCIDAD DE CAMBIO: Tendencias
    - ¿La temperatura está subiendo o bajando?
    - ¿El consumo está acelerando?
    
    Returns:
        DataFrame con nuevas columnas mágicas
    """
    df = df.copy()
    
    # =====================
    # 1. EFECTO RETARDADO (INERCIA)
    # =====================
    # La temperatura de hace 1 hora afecta el consumo de AHORA
    if 'temperatura_exterior_c' in df.columns:
        df['temp_hace_1h'] = df['temperatura_exterior_c'].shift(1)
        df['temp_hace_3h'] = df['temperatura_exterior_c'].shift(3)
    
    # La ocupación de hace 1 hora nos dice si la gente está llegando
    if 'ocupacion_pct' in df.columns:
        df['ocupacion_hace_1h'] = df['ocupacion_pct'].shift(1)
        df['ocupacion_hace_3h'] = df['ocupacion_pct'].shift(3)
    
    # =====================
    # 2. VELOCIDAD DE CAMBIO (TENDENCIA)
    # =====================
    # ¿La temperatura está subiendo o bajando?
    if 'temperatura_exterior_c' in df.columns:
        df['cambio_temp_1h'] = df['temperatura_exterior_c'].diff(1)
        df['cambio_temp_3h'] = df['temperatura_exterior_c'].diff(3)
    
    # ¿La ocupación está creciendo?
    if 'ocupacion_pct' in df.columns:
        df['cambio_ocupacion_1h'] = df['ocupacion_pct'].diff(1)
    
    return df


def generar_features_lag(df, target_col, lags=[1, 24, 168]):
    """
    Genera features de lag para una columna target.
    """
    df = df.copy()
    
    for lag in lags:
        col_name = f'{target_col}_lag_{lag}h'
        df[col_name] = df[target_col].shift(lag)
    
    return df


def generar_features_rolling(df, target_col, windows=[24]):
    """
    Genera features de rolling mean para una columna target.
    """
    df = df.copy()
    
    for window in windows:
        col_name = f'{target_col}_rolling_mean_{window}h'
        df[col_name] = df[target_col].rolling(window=window, min_periods=1).mean().shift(1)
    
    return df


def generar_velocidad_consumo(df, target_col):
    """
    Genera la velocidad de cambio del consumo (tendencia del target).
    """
    df = df.copy()
    col_name = f'{target_col}_velocidad_1h'
    df[col_name] = df[target_col].diff(1)
    return df


def preparar_features(df, target_col):
    """
    Prepara el DataFrame con TODAS las features, incluyendo variables mágicas.
    
    Args:
        df: DataFrame original
        target_col: Nombre del target
    
    Returns:
        X: Features
        y: Target  
        feature_names: Lista de nombres de features
        df_clean: DataFrame limpio
    """
    df = df.copy()
    
    # 1. Agregar VARIABLES MÁGICAS (inercia y velocidad)
    df = agregar_variables_magicas(df)
    
    # 2. Generar features de memoria histórica para este target
    df = generar_features_lag(df, target_col, LAGS)
    df = generar_features_rolling(df, target_col, ROLLING_WINDOWS)
    df = generar_velocidad_consumo(df, target_col)
    
    # Nombres de las features de memoria
    lag_features = [f'{target_col}_lag_{lag}h' for lag in LAGS]
    rolling_features = [f'{target_col}_rolling_mean_{w}h' for w in ROLLING_WINDOWS]
    velocidad_features = [f'{target_col}_velocidad_1h']
    
    # Lista de features mágicas
    magic_features = []
    
    # Inercia térmica
    if 'temp_hace_1h' in df.columns:
        magic_features.extend(['temp_hace_1h', 'temp_hace_3h'])
    
    # Inercia de ocupación  
    if 'ocupacion_hace_1h' in df.columns:
        magic_features.extend(['ocupacion_hace_1h', 'ocupacion_hace_3h'])
    
    # Velocidad de cambio
    if 'cambio_temp_1h' in df.columns:
        magic_features.extend(['cambio_temp_1h', 'cambio_temp_3h'])
    
    if 'cambio_ocupacion_1h' in df.columns:
        magic_features.append('cambio_ocupacion_1h')
    
    # Construir lista completa de features
    all_features = []
    
    # A. Temporales cíclicas
    for f in FEATURES_CICLICAS:
        if f in df.columns:
            all_features.append(f)
    
    # B. Calendario académico
    for f in FEATURES_CALENDARIO:
        if f in df.columns:
            all_features.append(f)
    
    # Buscar columnas de vacaciones
    vacaciones_cols = [c for c in df.columns if 'vacaciones' in c.lower()]
    for f in vacaciones_cols:
        if f not in all_features:
            all_features.append(f)
    
    # C. Variables exógenas
    for f in FEATURES_EXOGENAS:
        if f in df.columns:
            all_features.append(f)
    
    # D. Features de memoria histórica
    all_features.extend(lag_features)
    all_features.extend(rolling_features)
    
    # E. 🆕 VARIABLES MÁGICAS
    all_features.extend(magic_features)
    all_features.extend(velocidad_features)
    
    # Eliminar filas con NaN (las primeras 168 horas por el lag de 1 semana)
    df_clean = df.dropna(subset=all_features + [target_col])
    
    X = df_clean[all_features].values
    y = df_clean[target_col].values
    
    return X, y, all_features, df_clean


# ==========================================
# FUNCIONES DE MÉTRICAS Y EVALUACIÓN
# ==========================================

def calcular_metricas(y_true, y_pred):
    """Calcula métricas de evaluación."""
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.inf
    
    return {
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'R2': float(r2_score(y_true, y_pred)),
        'MAPE': float(min(mape, 999.99)),  # Cap para evitar infinitos en JSON
    }


def interpretar_r2(r2):
    """Interpreta el valor de R2."""
    if r2 >= 0.90:
        return "🏆 EXCELENTE"
    elif r2 >= 0.80:
        return "✅ MUY BUENO"
    elif r2 >= 0.70:
        return "👍 BUENO"
    elif r2 >= 0.50:
        return "⚠️ ACEPTABLE"
    else:
        return "❌ DEFICIENTE"


def detectar_overfitting(r2_train, r2_test, threshold=OVERFITTING_THRESHOLD):
    """
    Detecta si hay overfitting comparando R2 de train vs test.
    
    Un modelo que memoriza tendrá R2_train muy alto pero R2_test bajo.
    """
    gap = r2_train - r2_test
    
    if gap > threshold:
        return True, gap, f"⚠️ OVERFITTING (gap={gap:.3f})"
    elif gap > threshold/2:
        return False, gap, f"👀 Vigilar (gap={gap:.3f})"
    else:
        return False, gap, f"✅ OK (gap={gap:.3f})"


def obtener_feature_importance(model, feature_names, top_n=10):
    """Obtiene las características más importantes."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]
    
    top_features = []
    for i in indices:
        top_features.append({
            'feature': feature_names[i],
            'importance': float(importance[i])
        })
    return top_features


# ==========================================
# BÚSQUEDA DE HIPERPARÁMETROS
# ==========================================

def buscar_mejores_hiperparametros(X_train, y_train, n_iter=N_ITER_SEARCH, n_splits=N_SPLITS_CV):
    """
    Busca los mejores hiperparámetros usando RandomizedSearchCV con TimeSeriesSplit.
    
    TimeSeriesSplit es CRÍTICO para series temporales porque:
    - NUNCA usa datos del futuro para predecir el pasado
    - Cada fold usa datos anteriores para entrenar y siguientes para validar
    
    Returns:
        best_model: Mejor modelo encontrado
        best_params: Mejores hiperparámetros
        cv_score: Score de validación cruzada
    """
    # Modelo base
    xgb_base = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42,
        verbosity=0
    )
    
    # TimeSeriesSplit respeta el orden temporal
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Búsqueda randomizada
    search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=PARAM_DIST,
        n_iter=n_iter,
        scoring='neg_root_mean_squared_error',  # Negativo porque sklearn minimiza
        cv=tscv,
        verbose=0,
        n_jobs=-1,
        random_state=42
    )
    
    search.fit(X_train, y_train)
    
    return search.best_estimator_, search.best_params_, -search.best_score_


# ==========================================
# ENTRENAMIENTO PRINCIPAL
# ==========================================

def entrenar_modelos(usar_busqueda_hp=True):
    """
    Entrena modelos XGBoost con:
    1. Variables mágicas (inercia y velocidad)
    2. Búsqueda de hiperparámetros (opcional)
    3. Validación anti-overfitting
    
    Args:
        usar_busqueda_hp: Si True, usa RandomizedSearchCV. Si False, usa parámetros fijos.
    """
    
    print("=" * 70)
    print("🚀 ENTRENAMIENTO XGBoost v3 - VARIABLES MÁGICAS + ANTI-OVERFITTING")
    print("=" * 70)
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📊 MEJORAS EN ESTA VERSIÓN:")
    print(f"   🧊 Variables de INERCIA: temp_hace_1h, temp_hace_3h, ocupacion_hace_1h")
    print(f"   ⚡ Variables de VELOCIDAD: cambio_temp_1h, velocidad_consumo")
    print(f"   🔍 Detección de OVERFITTING: Comparación Train vs Test")
    if usar_busqueda_hp:
        print(f"   🎯 Búsqueda hiperparámetros: RandomizedSearchCV ({N_ITER_SEARCH} iter)")
        print(f"   ⏱️  TimeSeriesSplit: {N_SPLITS_CV} folds (respeta orden temporal)")
    
    resultados_globales = {
        'fecha_entrenamiento': datetime.now().isoformat(),
        'version': 'v3_magic_features_anti_overfit',
        'configuracion': {
            'usar_busqueda_hp': usar_busqueda_hp,
            'n_iter_search': N_ITER_SEARCH if usar_busqueda_hp else 0,
            'n_splits_cv': N_SPLITS_CV,
            'overfitting_threshold': OVERFITTING_THRESHOLD,
        },
        'variables_magicas': [
            'temp_hace_1h', 'temp_hace_3h',
            'ocupacion_hace_1h', 'ocupacion_hace_3h',
            'cambio_temp_1h', 'cambio_temp_3h',
            'cambio_ocupacion_1h',
            '{target}_velocidad_1h'
        ],
        'sedes': {}
    }
    
    total_modelos = 0
    modelos_exitosos = 0
    modelos_con_overfitting = 0
    
    for sede, archivo in ARCHIVOS_SEDES.items():
        input_path = os.path.join(INPUT_DIR, archivo)
        
        if not os.path.exists(input_path):
            print(f"\n⚠️  Saltando {sede}: No existe {input_path}")
            continue
        
        print(f"\n{'='*70}")
        print(f"📍 SEDE: {sede}")
        print(f"{'='*70}")
        
        # Cargar datos
        df = pd.read_csv(input_path)
        
        # Asegurar orden temporal
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"   📊 Datos cargados: {len(df):,} filas × {len(df.columns)} columnas")
        
        resultados_globales['sedes'][sede] = {
            'archivo': archivo,
            'n_registros': len(df),
            'modelos': {}
        }
        
        for target in TARGETS:
            if target not in df.columns:
                print(f"   ⚠️  Target '{target}' no encontrado, saltando...")
                continue
            
            total_modelos += 1
            
            print(f"\n   {'─'*50}")
            print(f"   🎯 TARGET: {target}")
            print(f"   {'─'*50}")
            
            # Preparar features con variables mágicas
            try:
                X, y, feature_names, df_clean = preparar_features(df, target)
            except Exception as e:
                print(f"      ❌ ERROR preparando features: {e}")
                continue
            
            n_features = len(feature_names)
            n_samples = len(X)
            
            print(f"      📐 Features: {n_features} | Muestras: {n_samples:,}")
            
            # Contar variables mágicas
            magic_count = sum(1 for f in feature_names if 
                            'hace_' in f or 'cambio_' in f or 'velocidad' in f)
            print(f"      ✨ Variables mágicas activas: {magic_count}")
            
            # División temporal (NUNCA shuffle en series de tiempo)
            split_idx = int(len(X) * (1 - TEST_SIZE))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            print(f"      📈 Train: {len(X_train):,} | Test: {len(X_test):,}")
            
            # ==========================================
            # ENTRENAMIENTO CON O SIN BÚSQUEDA HP
            # ==========================================
            
            if usar_busqueda_hp:
                print(f"      🔍 Buscando hiperparámetros...")
                model, best_params, cv_rmse = buscar_mejores_hiperparametros(X_train, y_train)
                print(f"      ✅ Mejor RMSE CV: {cv_rmse:.4f}")
            else:
                # Parámetros fijos (más rápido)
                best_params = {
                    'n_estimators': 500,
                    'max_depth': 7,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 5,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.5,
                }
                model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    n_jobs=-1,
                    random_state=42,
                    verbosity=0,
                    **best_params
                )
                model.fit(X_train, y_train)
            
            # Predicciones
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Métricas
            metrics_train = calcular_metricas(y_train, y_pred_train)
            metrics_test = calcular_metricas(y_test, y_pred_test)
            
            # ==========================================
            # DETECCIÓN DE OVERFITTING
            # ==========================================
            is_overfit, gap, overfit_msg = detectar_overfitting(
                metrics_train['R2'], 
                metrics_test['R2']
            )
            
            if is_overfit:
                modelos_con_overfitting += 1
            
            # Feature importance
            top_features = obtener_feature_importance(model, feature_names, top_n=7)
            
            # ==========================================
            # MOSTRAR RESULTADOS
            # ==========================================
            print(f"      📊 MÉTRICAS:")
            print(f"         Train R²: {metrics_train['R2']:.4f}")
            print(f"         Test R²:  {metrics_test['R2']:.4f} {interpretar_r2(metrics_test['R2'])}")
            print(f"         MAPE:     {metrics_test['MAPE']:.2f}%")
            print(f"         🔬 Overfitting: {overfit_msg}")
            
            print(f"      🔝 TOP FEATURES:")
            for i, feat in enumerate(top_features[:5], 1):
                is_magic = '✨' if ('hace_' in feat['feature'] or 
                                   'cambio_' in feat['feature'] or 
                                   'velocidad' in feat['feature']) else ''
                bar = "█" * int(feat['importance'] * 40)
                print(f"         {i}. {feat['feature'][:25]:<25} {bar} {is_magic}")
            
            # Guardar modelo
            model_filename = f"xgb_v3_{sede}_{target}.pkl"
            model_path = os.path.join(MODEL_DIR, model_filename)
            joblib.dump({
                'model': model,
                'feature_names': feature_names,
                'target': target,
                'sede': sede,
                'lags': LAGS,
                'rolling_windows': ROLLING_WINDOWS,
                'best_params': best_params,
                'version': 'v3_magic_features',
            }, model_path)
            print(f"      💾 Modelo guardado: {model_filename}")
            
            # Almacenar resultados
            resultados_globales['sedes'][sede]['modelos'][target] = {
                'metricas_train': metrics_train,
                'metricas_test': metrics_test,
                'overfitting': {
                    'detected': is_overfit,
                    'gap': float(gap),
                    'message': overfit_msg
                },
                'best_params': best_params,
                'top_features': top_features,
                'n_features': n_features,
                'n_magic_features': magic_count,
                'n_samples_train': len(X_train),
                'n_samples_test': len(X_test),
                'archivo_modelo': model_filename
            }
            
            modelos_exitosos += 1
    
    # ==========================================
    # RESUMEN FINAL
    # ==========================================
    print("\n" + "=" * 70)
    print("📊 RESUMEN DE ENTRENAMIENTO v3 (MAGIC FEATURES + ANTI-OVERFIT)")
    print("=" * 70)
    
    # Tabla resumen R2
    print("\n🏆 R² TEST POR SEDE Y TARGET:")
    print("-" * 90)
    header = f"{'Target':<25}"
    for sede in ARCHIVOS_SEDES.keys():
        header += f" | {sede:>10}"
    print(header)
    print("-" * 90)
    
    for target in TARGETS:
        row = f"{target:<25}"
        for sede in ARCHIVOS_SEDES.keys():
            if sede in resultados_globales['sedes']:
                if target in resultados_globales['sedes'][sede]['modelos']:
                    r2 = resultados_globales['sedes'][sede]['modelos'][target]['metricas_test']['R2']
                    overfit = resultados_globales['sedes'][sede]['modelos'][target]['overfitting']['detected']
                    marker = "⚠️" if overfit else ""
                    row += f" | {r2:>8.4f}{marker}"
                else:
                    row += f" | {'N/A':>10}"
            else:
                row += f" | {'N/A':>10}"
        print(row)
    
    print("-" * 90)
    
    # Resumen overfitting
    print(f"\n🔬 ANÁLISIS DE OVERFITTING:")
    print(f"   Total modelos: {modelos_exitosos}")
    print(f"   Con overfitting detectado: {modelos_con_overfitting}")
    print(f"   Tasa de overfitting: {modelos_con_overfitting/max(modelos_exitosos,1)*100:.1f}%")
    
    if modelos_con_overfitting > 0:
        print(f"   ⚠️ Los modelos con overfitting podrían beneficiarse de más regularización")
    else:
        print(f"   ✅ Ningún modelo muestra señales claras de overfitting")
    
    # Guardar resultados
    results_file = os.path.join(RESULTS_DIR, "resultados_entrenamiento_v3.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(resultados_globales, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ ENTRENAMIENTO COMPLETADO")
    print(f"   📈 Modelos entrenados: {modelos_exitosos}/{total_modelos}")
    print(f"   💾 Modelos guardados en: {MODEL_DIR}")
    print(f"   📊 Resultados: {results_file}")
    
    # Mejores modelos
    print("\n🏅 MEJORES MODELOS (R² más alto por sede):")
    for sede in ARCHIVOS_SEDES.keys():
        if sede in resultados_globales['sedes']:
            modelos = resultados_globales['sedes'][sede].get('modelos', {})
            if modelos:
                mejor = max(modelos.items(), key=lambda x: x[1]['metricas_test']['R2'])
                r2 = mejor[1]['metricas_test']['R2']
                print(f"   {sede}: {mejor[0]} (R²={r2:.4f})")
    
    return resultados_globales


# ==========================================
# FUNCIÓN PARA PREDICCIONES
# ==========================================

def predecir(sede, target, df_nuevos_datos):
    """
    Realiza predicciones usando el modelo v3 entrenado.
    
    IMPORTANTE: df_nuevos_datos debe tener historial suficiente para:
    - Lags: H-1, H-24, H-168
    - Variables mágicas: temperatura y ocupación históricas
    """
    model_path = os.path.join(MODEL_DIR, f"xgb_v3_{sede}_{target}.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    
    # Cargar modelo y metadatos
    data = joblib.load(model_path)
    model = data['model']
    feature_names = data['feature_names']
    lags = data['lags']
    rolling_windows = data['rolling_windows']
    
    # Preparar features
    df = df_nuevos_datos.copy()
    df = agregar_variables_magicas(df)
    df = generar_features_lag(df, target, lags)
    df = generar_features_rolling(df, target, rolling_windows)
    df = generar_velocidad_consumo(df, target)
    
    # Verificar features
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Faltan features: {missing}")
    
    X = df[feature_names].values
    predicciones = model.predict(X)
    
    return predicciones


# ==========================================
# EJECUCIÓN
# ==========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenamiento XGBoost v3')
    parser.add_argument('--fast', action='store_true', 
                       help='Modo rápido sin búsqueda de hiperparámetros')
    args = parser.parse_args()
    
    usar_busqueda = not args.fast
    
    if args.fast:
        print("🏃 Modo RÁPIDO: Sin búsqueda de hiperparámetros")
    else:
        print("🔍 Modo COMPLETO: Con RandomizedSearchCV (más lento pero mejor)")
    
    resultados = entrenar_modelos(usar_busqueda_hp=usar_busqueda)
