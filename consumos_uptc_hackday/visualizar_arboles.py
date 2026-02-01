# -*- coding: utf-8 -*-
"""
PASO 7: Visualización de Árboles de Decisión XGBoost
=====================================================
Este script carga los modelos YA ENTRENADOS y visualiza los árboles
usando plot_tree de XGBoost.

NO re-entrena el modelo, solo carga y visualiza.
"""

import joblib
import os
import sys
import xgboost as xgb
import matplotlib.pyplot as plt

# Fix encoding para Windows
sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# CONFIGURACIÓN
# ==========================================
BASE_DIR = r"c:\Users\POWER\OneDrive\Escritorio\consumos_uptc_hackday"
MODEL_DIR = os.path.join(BASE_DIR, "MODELOS_XGBOOST_V3")
OUTPUT_DIR = os.path.join(BASE_DIR, "VISUALIZACIONES_ARBOLES")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lista de sedes y targets disponibles
SEDES = ['UPTC_TUN', 'UPTC_SOG', 'UPTC_DUI', 'UPTC_CHI']
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


def listar_modelos_disponibles():
    """Lista todos los modelos disponibles en el directorio."""
    print("=" * 60)
    print("📁 MODELOS DISPONIBLES")
    print("=" * 60)
    
    modelos = []
    for archivo in sorted(os.listdir(MODEL_DIR)):
        if archivo.endswith('.pkl') and archivo.startswith('xgb_v3_'):
            # Extraer sede y target del nombre
            partes = archivo.replace('xgb_v3_', '').replace('.pkl', '').split('_', 1)
            if len(partes) >= 2:
                sede = f"{partes[0]}_{partes[1].split('_')[0]}"
                target = '_'.join(archivo.replace('xgb_v3_', '').replace('.pkl', '').split('_')[2:])
                modelos.append((sede, target, archivo))
                print(f"   • {sede} → {target}")
    
    print(f"\n   Total: {len(modelos)} modelos")
    return modelos


def cargar_modelo(sede, target):
    """
    Carga un modelo ya entrenado desde el disco.
    
    Args:
        sede: Código de la sede (ej: 'UPTC_TUN')
        target: Variable objetivo (ej: 'energia_total_kwh')
    
    Returns:
        dict con 'model', 'feature_names', y otros metadatos
    """
    model_filename = f"xgb_v3_{sede}_{target}.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Modelo no encontrado: {model_path}")
    
    print(f"📂 Cargando modelo: {model_filename}")
    data = joblib.load(model_path)
    
    print(f"   ✅ Modelo cargado exitosamente")
    print(f"   📊 Features: {len(data['feature_names'])}")
    print(f"   🔧 Versión: {data.get('version', 'unknown')}")
    
    return data


def visualizar_interpretabilidad_sin_graphviz(sede, target, top_n=15, save=True, show=True):
    """
    Visualización ALTERNATIVA que NO requiere Graphviz.
    Muestra:
    1. Feature Importance (barras horizontales)
    2. Estructura textual del primer árbol
    3. Parámetros del modelo
    
    Args:
        sede: Código de la sede
        target: Variable objetivo
        top_n: Número de features más importantes a mostrar
        save: Si True, guarda la imagen
        show: Si True, muestra la imagen
    """
    import numpy as np
    
    # Cargar modelo
    data = cargar_modelo(sede, target)
    model = data['model']
    feature_names = data['feature_names']
    best_params = data.get('best_params', {})
    
    # Obtener número total de árboles
    booster = model.get_booster()
    n_trees = len(booster.get_dump())
    
    print(f"\n📊 INTERPRETABILIDAD DEL MODELO")
    print(f"   Sede: {sede}")
    print(f"   Target: {target}")
    print(f"   Total árboles en ensemble: {n_trees}")
    
    # ==========================================
    # FIGURA 1: Feature Importance
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    
    # Obtener importancia de features
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]
    
    # Subplot 1: Barras horizontales de importancia
    ax1 = axes[0]
    y_pos = np.arange(len(indices))
    
    colors = []
    for i in indices:
        feat_name = feature_names[i]
        if 'lag_' in feat_name or 'rolling_' in feat_name:
            colors.append('#3498db')  # Azul - memoria
        elif 'hace_' in feat_name or 'cambio_' in feat_name or 'velocidad' in feat_name:
            colors.append('#e74c3c')  # Rojo - mágicas
        elif 'sin' in feat_name or 'cos' in feat_name:
            colors.append('#9b59b6')  # Morado - cíclicas
        else:
            colors.append('#2ecc71')  # Verde - otras
    
    bars = ax1.barh(y_pos, importance[indices][::-1], color=colors[::-1], edgecolor='black', alpha=0.8)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([feature_names[i] for i in indices][::-1], fontsize=10)
    ax1.set_xlabel('Importancia (Gain)', fontsize=12)
    ax1.set_title(f'🎯 TOP {top_n} Features Más Importantes\n{sede} - {target}', fontsize=14, fontweight='bold')
    
    # Agregar valores
    for i, (bar, val) in enumerate(zip(bars, importance[indices][::-1])):
        ax1.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)
    
    # Leyenda de colores
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Memoria (lags, rolling)'),
        Patch(facecolor='#e74c3c', label='Variables Mágicas'),
        Patch(facecolor='#9b59b6', label='Cíclicas (sin/cos)'),
        Patch(facecolor='#2ecc71', label='Otras'),
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # Subplot 2: Información del modelo + estructura de árbol
    ax2 = axes[1]
    ax2.axis('off')
    
    # Texto informativo
    info_text = f"""
    ═══════════════════════════════════════════════
    📋 INFORMACIÓN DEL MODELO XGBoost
    ═══════════════════════════════════════════════
    
    🏢 Sede: {sede}
    🎯 Target: {target}
    🌲 Total de Árboles: {n_trees}
    📊 Número de Features: {len(feature_names)}
    
    ═══════════════════════════════════════════════
    🔧 HIPERPARÁMETROS ÓPTIMOS
    ═══════════════════════════════════════════════
    """
    
    for param, value in best_params.items():
        info_text += f"\n    • {param}: {value}"
    
    # Agregar estructura del primer árbol (simplificada)
    tree_dump = booster.get_dump()[0]
    tree_lines = tree_dump.split('\n')[:15]  # Primeras 15 líneas
    
    info_text += f"""
    
    ═══════════════════════════════════════════════
    🌲 ESTRUCTURA DEL ÁRBOL #1 (primeras 15 líneas)
    ═══════════════════════════════════════════════
    """
    
    for line in tree_lines:
        if line.strip():
            info_text += f"\n    {line}"
    
    if len(booster.get_dump()[0].split('\n')) > 15:
        info_text += f"\n    ... (y más nodos)"
    
    ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, 
             fontsize=10, verticalalignment='top', 
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    
    plt.suptitle(f'🔍 Interpretabilidad del Modelo XGBoost v3\n{target}', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Guardar
    if save:
        filename = f"interpretabilidad_{sede}_{target}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"   💾 Guardado: {filepath}")
    
    # Mostrar
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def visualizar_arbol(sede, target, num_trees=0, figsize=(30, 15), save=True, show=True):
    """
    Visualiza un árbol específico del modelo XGBoost.
    REQUIERE GRAPHVIZ INSTALADO EN EL SISTEMA.
    
    Si no tienes Graphviz, usa visualizar_interpretabilidad_sin_graphviz() en su lugar.
    
    Args:
        sede: Código de la sede
        target: Variable objetivo
        num_trees: Índice del árbol a visualizar (0 = primer árbol)
        figsize: Tamaño de la figura
        save: Si True, guarda la imagen en OUTPUT_DIR
        show: Si True, muestra la imagen
    """
    # Cargar modelo
    data = cargar_modelo(sede, target)
    model = data['model']
    feature_names = data['feature_names']
    
    # Obtener número total de árboles
    booster = model.get_booster()
    n_trees = len(booster.get_dump())
    
    print(f"\n🌲 VISUALIZANDO ÁRBOL {num_trees + 1} de {n_trees}")
    print(f"   Sede: {sede}")
    print(f"   Target: {target}")
    
    # Crear figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot del árbol (requiere graphviz)
    xgb.plot_tree(
        model, 
        num_trees=num_trees,
        ax=ax,
        rankdir='TB'  # Top to Bottom (de arriba hacia abajo)
    )
    
    # Título
    ax.set_title(
        f"Árbol de Decisión #{num_trees + 1} - {sede} - {target}\n"
        f"(Total: {n_trees} árboles en el ensemble)",
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout()
    
    # Guardar
    if save:
        filename = f"arbol_{sede}_{target}_tree{num_trees}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"   💾 Guardado: {filepath}")
    
    # Mostrar
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def visualizar_multiples_arboles(sede, target, num_trees=[0, 1, 2], save=True):
    """
    Visualiza múltiples árboles del mismo modelo.
    """
    print(f"\n🌲🌲🌲 VISUALIZANDO {len(num_trees)} ÁRBOLES")
    
    for tree_idx in num_trees:
        try:
            visualizar_arbol(sede, target, num_trees=tree_idx, save=save, show=False)
        except Exception as e:
            print(f"   ❌ Error en árbol {tree_idx}: {e}")
    
    print(f"\n✅ Visualizaciones guardadas en: {OUTPUT_DIR}")


def mostrar_estructura_arbol(sede, target, num_trees=0):
    """
    Muestra la estructura textual del árbol (reglas de decisión).
    """
    data = cargar_modelo(sede, target)
    model = data['model']
    booster = model.get_booster()
    
    # Obtener dump del árbol
    tree_dump = booster.get_dump()[num_trees]
    
    print(f"\n📜 ESTRUCTURA DEL ÁRBOL {num_trees + 1}")
    print("=" * 60)
    print(tree_dump)
    print("=" * 60)
    
    return tree_dump


def visualizar_sede_completa(sede, top_n=10, save=True, show=True):
    """
    Visualiza la interpretabilidad de TODOS los modelos de una sede en una sola figura.
    Crea un grid 3x3 con los 9 targets.
    
    Args:
        sede: Código de la sede (ej: 'UPTC_TUN')
        top_n: Número de features más importantes a mostrar por modelo
        save: Si True, guarda la imagen
        show: Si True, muestra la imagen
    """
    import numpy as np
    from matplotlib.patches import Patch
    
    print(f"\n{'='*70}")
    print(f"VISUALIZACION COMPLETA DE SEDE: {sede}")
    print(f"{'='*70}")
    
    # Crear figura grande con grid 3x3
    fig, axes = plt.subplots(3, 3, figsize=(24, 20))
    axes = axes.flatten()
    
    # Nombres cortos para los targets (para que quepan en el título)
    target_short_names = {
        'energia_total_kwh': 'Energia Total',
        'energia_comedor_kwh': 'Comedor',
        'energia_salones_kwh': 'Salones',
        'energia_laboratorios_kwh': 'Laboratorios',
        'energia_auditorios_kwh': 'Auditorios',
        'energia_oficinas_kwh': 'Oficinas',
        'potencia_total_kw': 'Potencia Total',
        'agua_litros': 'Agua',
        'co2_kg': 'CO2',
    }
    
    for idx, target in enumerate(TARGETS):
        ax = axes[idx]
        
        try:
            # Cargar modelo (silencioso)
            model_filename = f"xgb_v3_{sede}_{target}.pkl"
            model_path = os.path.join(MODEL_DIR, model_filename)
            
            if not os.path.exists(model_path):
                ax.text(0.5, 0.5, f'Modelo no encontrado\n{target}', 
                       ha='center', va='center', fontsize=12)
                ax.set_title(target_short_names.get(target, target))
                continue
            
            data = joblib.load(model_path)
            model = data['model']
            feature_names = data['feature_names']
            
            # Obtener importancia
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1][:top_n]
            
            # Preparar colores
            colors = []
            for i in indices:
                feat_name = feature_names[i]
                if 'lag_' in feat_name or 'rolling_' in feat_name:
                    colors.append('#3498db')  # Azul - memoria
                elif 'hace_' in feat_name or 'cambio_' in feat_name or 'velocidad' in feat_name:
                    colors.append('#e74c3c')  # Rojo - mágicas
                elif 'sin' in feat_name or 'cos' in feat_name:
                    colors.append('#9b59b6')  # Morado - cíclicas
                else:
                    colors.append('#2ecc71')  # Verde - otras
            
            # Barras horizontales
            y_pos = np.arange(len(indices))
            bars = ax.barh(y_pos, importance[indices][::-1], 
                          color=colors[::-1], edgecolor='black', alpha=0.8)
            
            # Etiquetas
            ax.set_yticks(y_pos)
            feature_labels = [feature_names[i][:20] for i in indices][::-1]  # Truncar nombres largos
            ax.set_yticklabels(feature_labels, fontsize=8)
            ax.set_xlabel('Importancia', fontsize=9)
            
            # Título con nombre corto
            short_name = target_short_names.get(target, target)
            ax.set_title(f'{short_name}', fontsize=12, fontweight='bold')
            
            # Agregar valores en las barras
            for bar, val in zip(bars, importance[indices][::-1]):
                if val > 0.01:  # Solo mostrar si es significativo
                    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                           f'{val:.2f}', va='center', fontsize=7)
            
            print(f"   [OK] {target}")
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error:\n{str(e)[:30]}', 
                   ha='center', va='center', fontsize=10, color='red')
            ax.set_title(target_short_names.get(target, target))
            print(f"   [ERROR] {target}: {e}")
    
    # Leyenda global (en la parte inferior)
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Exogenas (ocupacion, temp)', edgecolor='black'),
        Patch(facecolor='#e74c3c', label='Magicas (velocidad, inercia)', edgecolor='black'),
        Patch(facecolor='#3498db', label='Memoria (lags, rolling)', edgecolor='black'),
        Patch(facecolor='#9b59b6', label='Ciclicas (sin/cos)', edgecolor='black'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
               fontsize=11, bbox_to_anchor=(0.5, 0.02))
    
    # Título principal
    fig.suptitle(f'Interpretabilidad de Modelos XGBoost - {sede}\n'
                 f'TOP {top_n} Features por Cada Variable Objetivo', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Guardar
    if save:
        filename = f"interpretabilidad_COMPLETA_{sede}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n   Guardado: {filepath}")
    
    # Mostrar
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def menu_interactivo():
    """
    Menú interactivo para explorar los modelos.
    """
    print("\n" + "=" * 60)
    print("🌲 VISUALIZACIÓN DE ÁRBOLES XGBoost v3")
    print("=" * 60)
    
    # Listar modelos disponibles
    modelos = listar_modelos_disponibles()
    
    if not modelos:
        print("❌ No se encontraron modelos en el directorio.")
        return
    
    print("\n📋 OPCIONES:")
    print("   1. Visualizar árbol interactivo")
    print("   2. Generar visualizaciones de TODOS los modelos")
    print("   3. Ver estructura textual de un árbol")
    print("   4. Salir")
    
    opcion = input("\nSelecciona opción (1-4): ").strip()
    
    if opcion == "1":
        # Seleccionar sede
        print("\n📍 SEDES DISPONIBLES:")
        for i, sede in enumerate(SEDES, 1):
            print(f"   {i}. {sede}")
        
        try:
            idx_sede = int(input("Selecciona sede (1-4): ")) - 1
            sede = SEDES[idx_sede]
        except (ValueError, IndexError):
            print("❌ Opción inválida")
            return
        
        # Seleccionar target
        print(f"\n🎯 TARGETS DISPONIBLES para {sede}:")
        for i, target in enumerate(TARGETS, 1):
            print(f"   {i}. {target}")
        
        try:
            idx_target = int(input("Selecciona target (1-9): ")) - 1
            target = TARGETS[idx_target]
        except (ValueError, IndexError):
            print("❌ Opción inválida")
            return
        
        # Número de árbol
        try:
            num_tree = int(input("Número de árbol a visualizar (0 = primero): "))
        except ValueError:
            num_tree = 0
        
        visualizar_arbol(sede, target, num_trees=num_tree, save=True, show=True)
    
    elif opcion == "2":
        print("\n⏳ Generando visualizaciones de todos los modelos...")
        for sede, target, _ in modelos:
            try:
                visualizar_arbol(sede, target, num_trees=0, save=True, show=False)
            except Exception as e:
                print(f"   ❌ Error en {sede}/{target}: {e}")
        print(f"\n✅ Todas las visualizaciones guardadas en: {OUTPUT_DIR}")
    
    elif opcion == "3":
        # Selección rápida
        print("\n📍 Ingresa: SEDE TARGET (ej: UPTC_TUN energia_total_kwh)")
        entrada = input("> ").strip().split()
        
        if len(entrada) >= 2:
            sede = entrada[0]
            target = entrada[1]
            mostrar_estructura_arbol(sede, target, num_trees=0)
        else:
            print("❌ Formato inválido")
    
    elif opcion == "4":
        print("👋 ¡Hasta luego!")
    else:
        print("❌ Opción no válida")


# ==========================================
# EJEMPLOS DE USO RÁPIDO
# ==========================================

def ejemplo_rapido():
    """
    Ejemplo rápido: visualiza la interpretabilidad del modelo de energía total de Tunja.
    NO requiere Graphviz instalado.
    """
    print("\n" + "=" * 60)
    print("🚀 EJEMPLO RÁPIDO: Interpretabilidad del Modelo XGBoost")
    print("=" * 60)
    print("💡 Usando método alternativo (NO requiere Graphviz)")
    
    # Parámetros
    SEDE = "UPTC_TUN"
    TARGET = "energia_total_kwh"
    
    # Visualizar (sin graphviz)
    visualizar_interpretabilidad_sin_graphviz(
        sede=SEDE,
        target=TARGET,
        top_n=15,
        save=True,
        show=True
    )
    
    # Mostrar estructura textual también
    print("\n📜 Mostrando estructura textual completa del árbol:")
    mostrar_estructura_arbol(SEDE, TARGET, num_trees=0)


# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualización de Árboles XGBoost v3')
    parser.add_argument('--sede', type=str, help='Código de sede (ej: UPTC_TUN)')
    parser.add_argument('--target', type=str, help='Variable objetivo (ej: energia_total_kwh)')
    parser.add_argument('--tree', type=int, default=0, help='Número del árbol a visualizar')
    parser.add_argument('--menu', action='store_true', help='Iniciar menú interactivo')
    parser.add_argument('--all', action='store_true', help='Generar visualizaciones de TODOS los modelos')
    parser.add_argument('--ejemplo', action='store_true', help='Ejecutar ejemplo rápido')
    parser.add_argument('--sede-completa', type=str, dest='sede_completa',
                       help='Visualizar TODOS los targets de una sede (ej: UPTC_TUN)')
    
    args = parser.parse_args()
    
    if args.menu:
        menu_interactivo()
    elif args.sede_completa:
        # Visualizar todos los targets de una sede
        visualizar_sede_completa(args.sede_completa, top_n=10, save=True, show=True)
    elif args.all:
        modelos = listar_modelos_disponibles()
        for sede, target, _ in modelos:
            try:
                visualizar_arbol(sede, target, num_trees=0, save=True, show=False)
            except Exception as e:
                print(f"   Error: {e}")
        print(f"\n Guardado en: {OUTPUT_DIR}")
    elif args.sede and args.target:
        visualizar_arbol(args.sede, args.target, num_trees=args.tree, save=True, show=True)
    elif args.ejemplo:
        ejemplo_rapido()
    else:
        # Por defecto: ejemplo rápido
        ejemplo_rapido()

