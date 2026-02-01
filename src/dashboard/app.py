"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🌿 ECOCAMPUS UPTC - Sede Tunja                            ║
║              Sistema Inteligente de Gestión Energética                       ║
║                      Hackathon IAMinds 2026                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

# Importar módulos propios
from claude_client import ClaudeAssistant
from data_simulator import DataSimulator
from triggers import TriggerEngine
from models import get_predictor, predict
import re
import math

# ==============================================================================
# CONFIGURACIÓN DE SEDES Y ESCENARIOS
# ==============================================================================
SEDES_CONFIG = {
    "UPTC_TUN": {"nombre": "Tunja", "estudiantes": 18000, "icon": "🏛️"},
    "UPTC_SOG": {"nombre": "Sogamoso", "estudiantes": 6000, "icon": "⛏️"},
    "UPTC_DUI": {"nombre": "Duitama", "estudiantes": 5500, "icon": "🏭"},
    "UPTC_CHI": {"nombre": "Chiquinquirá", "estudiantes": 2000, "icon": "🌾"},
}

ESCENARIOS_CONFIG = {
    "dia_normal": {"nombre": "Dia Normal", "icon": "☀️", "desc": "Dia habil tipico"},
    "fin_semana": {"nombre": "Fin de Semana", "icon": "🌙", "desc": "Sabado/Domingo"},
    "semana_parciales": {"nombre": "Semana Parciales", "icon": "📝", "desc": "Alta ocupacion"},
    "semana_finales": {"nombre": "Semana Finales", "icon": "📚", "desc": "Maxima ocupacion"},
    "vacaciones": {"nombre": "Vacaciones", "icon": "🏖️", "desc": "Consumo minimo"},
    "festivo": {"nombre": "Dia Festivo", "icon": "🎉", "desc": "Similar a fin de semana"},
    "anomalia_pico": {"nombre": "Anomalia Pico", "icon": "⚡", "desc": "Picos inesperados"},
    "anomalia_nocturna": {"nombre": "Anomalia Nocturna", "icon": "🧛", "desc": "Vampiro energetico"},
}

PERIODOS_CONFIG = {
    "dia": {"nombre": "Dia", "icon": "📅", "dias": 1, "x_label": "Hora", "format": "hora"},
    "semana": {"nombre": "Semana", "icon": "📆", "dias": 7, "x_label": "Dia", "format": "dia"},
    "mes": {"nombre": "Mes", "icon": "🗓️", "dias": 30, "x_label": "Dia", "format": "dia"},
    "semestre": {"nombre": "Semestre", "icon": "📚", "dias": 180, "x_label": "Semana", "format": "semana"},
    "anio": {"nombre": "Año", "icon": "🎯", "dias": 365, "x_label": "Mes", "format": "mes"},
}

# R² de los modelos XGBoost de Sebastian (por target)
# Estos valores se actualizan cuando Sebastian entregue los modelos reales
MODELO_R2 = {
    "energia_total_kwh": 0.95,
    "energia_comedor_kwh": 0.95,
    "energia_salones_kwh": 0.93,
    "energia_laboratorios_kwh": 0.87,
    "energia_auditorios_kwh": 0.06,  # Bajo rendimiento
    "energia_oficinas_kwh": 0.06,    # Bajo rendimiento
    "potencia_total_kw": 0.97,
    "agua_litros": 0.80,
    "co2_kg": 0.90,
}

# R² promedio ponderado para mostrar en dashboard
MODELO_R2_PROMEDIO = 0.54  # Promedio reportado por Sebastian

# Cargar variables de entorno
load_dotenv()


def format_chat_response(text: str) -> str:
    """
    Formatea la respuesta del LLM para mejor visualización en Streamlit.
    Convierte markdown a HTML donde sea necesario.
    """
    if not text:
        return text

    # Procesar línea por línea para manejar headers y listas
    lines = text.split('\n')
    formatted_lines = []
    in_list = False

    for line in lines:
        stripped = line.strip()

        # Headers ### (h4)
        if stripped.startswith('### '):
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            content = stripped[4:]
            formatted_lines.append(f'<h4 style="margin: 1rem 0 0.5rem 0; font-size: 1rem; color: #1e293b; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.3rem;">{content}</h4>')
        # Headers ## (h3)
        elif stripped.startswith('## '):
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            content = stripped[3:]
            formatted_lines.append(f'<h3 style="margin: 1.2rem 0 0.5rem 0; font-size: 1.1rem; color: #166534; font-weight: 700;">{content}</h3>')
        # Listas con guiones
        elif stripped.startswith('- ') or stripped.startswith('• '):
            if not in_list:
                formatted_lines.append('<ul style="margin: 0.5rem 0; padding-left: 1.5rem;">')
                in_list = True
            content = stripped[2:]
            formatted_lines.append(f'<li style="margin: 0.3rem 0;">{content}</li>')
        else:
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            formatted_lines.append(line)

    if in_list:
        formatted_lines.append('</ul>')

    text = '\n'.join(formatted_lines)

    # Reemplazar **texto** con <strong>texto</strong> para negritas
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)

    # Reemplazar *texto* con <em>texto</em> para itálicas
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', text)

    # Convertir saltos de línea dobles a párrafos
    text = text.replace('\n\n', '</p><p style="margin: 0.8rem 0;">')

    # Envolver en div contenedor
    text = f'<div style="line-height: 1.6;">{text}</div>'

    return text

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="EcoCampus UPTC - Gestión Energética",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# ESTILOS CSS PERSONALIZADOS
# ==============================================================================
st.markdown("""
<style>
    /* Fuente principal */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Ocultar elementos de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Header personalizado */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 1.5rem 2rem;
        border-radius: 0 0 20px 20px;
        margin: -1rem -1rem 1.5rem -1rem;
        color: white;
    }

    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
    }

    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }

    /* Tarjetas métricas */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #e8ecf1;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    }

    .metric-card .label {
        font-size: 0.85rem;
        color: #64748b;
        margin-bottom: 0.3rem;
        font-weight: 500;
    }

    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
    }

    .metric-card .delta {
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }

    .delta-positive { color: #ef4444; }
    .delta-negative { color: #22c55e; }
    .delta-neutral { color: #64748b; }

    /* Semáforo de estado */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.6rem 1.2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
    }

    .status-green {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        color: #166534;
        border: 2px solid #22c55e;
    }

    .status-yellow {
        background: linear-gradient(135deg, #fef9c3 0%, #fef08a 100%);
        color: #854d0e;
        border: 2px solid #eab308;
    }

    .status-red {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
        border: 2px solid #ef4444;
    }

    /* Tarjetas de sector */
    .sector-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    }

    .sector-card.alert {
        border-left-color: #ef4444;
        background: #fef2f2;
    }

    .sector-card.warning {
        border-left-color: #f59e0b;
        background: #fffbeb;
    }

    .sector-card.success {
        border-left-color: #22c55e;
        background: #f0fdf4;
    }

    /* Barra de progreso */
    .progress-bar {
        height: 8px;
        background: #e2e8f0;
        border-radius: 4px;
        overflow: hidden;
        margin-top: 0.5rem;
    }

    .progress-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }

    /* Alertas */
    .alert-item {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        border: 1px solid #e2e8f0;
        display: flex;
        align-items: flex-start;
        gap: 1rem;
    }

    .alert-icon {
        font-size: 1.5rem;
    }

    .alert-content {
        flex: 1;
    }

    .alert-title {
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }

    .alert-desc {
        font-size: 0.9rem;
        color: #64748b;
    }

    /* Chat */
    .chat-container {
        background: #f8fafc;
        border-radius: 16px;
        padding: 1rem;
        height: 100%;
        border: 1px solid #e2e8f0;
    }

    /* Contenedor de mensajes con scroll */
    .chat-messages-container {
        max-height: 550px;
        overflow-y: auto;
        padding-right: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Estilizar scrollbar */
    .chat-messages-container::-webkit-scrollbar {
        width: 6px;
    }

    .chat-messages-container::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 3px;
    }

    .chat-messages-container::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 3px;
    }

    .chat-messages-container::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }

    .chat-message {
        padding: 0.8rem 1rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
        max-width: 90%;
    }

    .chat-user {
        background: #3b82f6;
        color: white;
        margin-left: auto;
    }

    .chat-assistant {
        background: white;
        color: #1e293b;
        border: 1px solid #e2e8f0;
    }

    /* Forzar scroll en el contenedor de chat de Streamlit */
    [data-testid="stChatMessageContainer"] {
        max-height: 550px;
        overflow-y: auto;
    }

    /* Proyección de ahorro */
    .savings-card {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 2px solid #22c55e;
    }

    .savings-title {
        color: #166534;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }

    /* Botones sugeridos */
    .suggestion-btn {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        font-size: 0.85rem;
        color: #475569;
        cursor: pointer;
        transition: all 0.2s;
        margin: 0.2rem;
    }

    .suggestion-btn:hover {
        background: #f1f5f9;
        border-color: #3b82f6;
        color: #3b82f6;
    }

    /* Horarios críticos */
    .time-slot {
        display: flex;
        align-items: center;
        padding: 0.6rem;
        background: white;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border: 1px solid #e2e8f0;
    }

    .time-slot .dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 0.8rem;
    }

    .dot-red { background: #ef4444; }
    .dot-yellow { background: #f59e0b; }
    .dot-green { background: #22c55e; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# INICIALIZACIÓN DE COMPONENTES
# ==============================================================================
@st.cache_resource
def init_claude():
    """Inicializa el cliente de Claude"""
    try:
        assistant = ClaudeAssistant()
        return assistant
    except Exception as e:
        st.error(f"Error al conectar con Claude: {str(e)}")
        return None

@st.cache_data
def load_demo_data():
    """Carga los datos de demo desde el CSV"""
    csv_path = os.path.join(os.path.dirname(__file__), "data", "input", "demo_sensores.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

@st.cache_data
def load_periodos_data():
    """Carga los datos de periodos desde el CSV"""
    csv_path = os.path.join(os.path.dirname(__file__), "data", "input", "demo_periodos.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

def get_periodo_data(df_periodos, sede, periodo):
    """Obtiene datos de un periodo especifico para una sede"""
    if df_periodos is None:
        return None

    row = df_periodos[(df_periodos['sede'] == sede) & (df_periodos['periodo'] == periodo)]

    if row.empty:
        return None

    row = row.iloc[0]

    # Parsear curvas
    curva_real = [float(x) for x in row['curva_valores'].split(',')]
    curva_esperada = [float(x) for x in row['curva_esperada'].split(',')]

    return {
        'periodo': periodo,
        'sede': sede,
        'energia_real': row['energia_real'],
        'energia_esperada': row['energia_esperada'],
        'delta': row['delta'],
        'delta_pct': row['delta_pct'],
        'estado': row['estado'],
        'costo_delta': row['costo_delta'],
        'agua_total': row['agua_total'],
        'co2_total': row['co2_total'],
        'curva_real': curva_real,
        'curva_esperada': curva_esperada,
        'num_dias': row['num_dias']
    }

def get_data_from_csv(df_demo, sede, escenario):
    """Procesa datos del CSV para el formato que espera el dashboard.
    AHORA USA EL MODELO XGBOOST para generar predicciones reales."""
    if df_demo is None:
        simulator = DataSimulator()
        return simulator.generate_scenario('mixed')

    # Filtrar por sede y escenario
    df = df_demo[(df_demo['sede'] == sede) & (df_demo['escenario'] == escenario)].copy()

    if df.empty:
        simulator = DataSimulator()
        return simulator.generate_scenario('mixed')

    # Ordenar por hora y tomar solo primeras 24 horas
    df = df.sort_values('hora').head(24)

    # DATOS REALES del CSV (simulan sensores)
    energia_real_hourly = df['energia_total_kwh'].tolist()
    energia_total_real = sum(energia_real_hourly)

    # PREDICCIONES del MODELO XGBOOST
    predictor = get_predictor()
    energia_expected_hourly = []

    # Determinar periodo academico segun escenario
    if escenario in ['vacaciones']:
        periodo = 'vacaciones_fin'
    elif escenario in ['semana_parciales', 'semana_finales', 'dia_normal']:
        periodo = 'semestre_1'
    else:
        periodo = 'semestre_1'

    es_festivo = escenario == 'festivo'
    es_fin_semana = escenario == 'fin_semana'

    for idx, row in df.iterrows():
        hora = int(row['hora'])
        temp = row.get('temperatura_exterior', 15.0)
        ocup = row.get('ocupacion_pct', 50.0)

        # Llamar al modelo XGBoost
        pred = predictor.predict(
            sede=sede,
            target='energia_total_kwh',
            hora=hora,
            temperatura=temp,
            ocupacion_pct=ocup,
            periodo=periodo,
            es_festivo=es_festivo,
            es_fin_semana=es_fin_semana
        )
        energia_expected_hourly.append(pred)

    energia_total_expected = sum(energia_expected_hourly)

    # Calcular diferencia
    delta = energia_total_real - energia_total_expected
    delta_percent = (delta / energia_total_expected) * 100 if energia_total_expected > 0 else 0

    costo_kwh = 650
    co2_factor = 0.198

    # Construir estructura de datos
    data = {
        'total': {
            'real': energia_total_real,
            'expected': energia_total_expected,
            'delta': delta,
            'delta_percent': delta_percent,
            'cost': delta * costo_kwh,
            'co2': delta * co2_factor
        },
        'hourly': {
            'real': energia_real_hourly,
            'expected': energia_expected_hourly
        },
        'usando_modelo': True  # Flag para confirmar que usamos XGBoost
    }

    # Datos por sector - usar modelo XGBoost para cada sector
    sectores = ['laboratorios', 'oficinas', 'salones', 'comedores', 'auditorios']
    sector_cols = {
        'laboratorios': 'energia_laboratorios_kwh',
        'oficinas': 'energia_oficinas_kwh',
        'salones': 'energia_salones_kwh',
        'comedores': 'energia_comedor_kwh',
        'auditorios': 'energia_auditorios_kwh'
    }
    sector_targets = {
        'laboratorios': 'energia_laboratorios_kwh',
        'oficinas': 'energia_oficinas_kwh',
        'salones': 'energia_salones_kwh',
        'comedores': 'energia_comedor_kwh',
        'auditorios': 'energia_auditorios_kwh'
    }

    for sector in sectores:
        col = sector_cols[sector]
        real = df[col].sum()

        # Usar modelo XGBoost para prediccion del sector
        target = sector_targets[sector]
        expected_sector = 0
        for idx, row in df.iterrows():
            hora = int(row['hora'])
            temp = row.get('temperatura_exterior', 15.0)
            ocup = row.get('ocupacion_pct', 50.0)
            pred = predictor.predict(sede=sede, target=target, hora=hora, temperatura=temp, ocupacion_pct=ocup, periodo=periodo)
            expected_sector += pred

        delta_s = real - expected_sector
        delta_pct_s = (delta_s / expected_sector) * 100 if expected_sector > 0 else 0

        data[sector] = {
            'real': real,
            'expected': expected_sector,
            'delta': delta_s,
            'delta_percent': delta_pct_s
        }

    # Metadata adicional
    data['metadata'] = {
        'sede': sede,
        'escenario': escenario,
        'agua_total': df['agua_litros'].sum(),
        'co2_total': df['co2_kg'].sum(),
        'potencia_max': df['potencia_total_kw'].max(),
        'ocupacion_promedio': df['ocupacion_pct'].mean(),
        'temperatura_promedio': df['temperatura_exterior'].mean()
    }

    return data

@st.cache_data(ttl=60)
def get_simulated_data():
    """Obtiene datos simulados (se refresca cada 60 segundos)"""
    simulator = DataSimulator()
    return simulator.generate_scenario('mixed')  # Usar escenario mixto para demo

# Inicializar
claude = init_claude()
df_demo = load_demo_data()
df_periodos = load_periodos_data()
trigger_engine = TriggerEngine()

# ==============================================================================
# HEADER PRINCIPAL CON SELECTORES
# ==============================================================================
current_time = datetime.now().strftime("%H:%M")
current_date = datetime.now().strftime("%d de %B, %Y")

# Selectores de Sede
col_sel1, col_sel2 = st.columns([1, 3])

with col_sel1:
    sede_options = list(SEDES_CONFIG.keys())
    sede_labels = [f"{SEDES_CONFIG[s]['icon']} {SEDES_CONFIG[s]['nombre']}" for s in sede_options]
    sede_idx = st.selectbox(
        "Sede",
        range(len(sede_options)),
        format_func=lambda i: sede_labels[i],
        key="sede_selector"
    )
    selected_sede = sede_options[sede_idx]

with col_sel2:
    st.markdown("<p style='margin-bottom: 0.3rem; font-size: 0.85rem; color: #64748b;'>Periodo de Analisis</p>", unsafe_allow_html=True)
    periodo_cols = st.columns(5)
    periodo_options = list(PERIODOS_CONFIG.keys())

    # Inicializar periodo en session_state si no existe
    if 'selected_periodo' not in st.session_state:
        st.session_state.selected_periodo = 'dia'

    for i, periodo in enumerate(periodo_options):
        with periodo_cols[i]:
            btn_type = "primary" if st.session_state.selected_periodo == periodo else "secondary"
            if st.button(
                f"{PERIODOS_CONFIG[periodo]['icon']} {PERIODOS_CONFIG[periodo]['nombre']}",
                key=f"periodo_{periodo}",
                use_container_width=True,
                type=btn_type
            ):
                st.session_state.selected_periodo = periodo
                st.rerun()

    selected_periodo = st.session_state.selected_periodo

# Obtener datos segun periodo seleccionado
periodo_data = get_periodo_data(df_periodos, selected_sede, selected_periodo)

# Si hay datos de periodo, usarlos; si no, usar datos de escenario (dia)
if periodo_data and selected_periodo != 'dia':
    # Convertir datos de periodo al formato esperado
    data = {
        'total': {
            'real': periodo_data['energia_real'],
            'expected': periodo_data['energia_esperada'],
            'delta': periodo_data['delta'],
            'delta_percent': periodo_data['delta_pct'],
            'cost': periodo_data['costo_delta'],
            'co2': periodo_data['co2_total']
        },
        'hourly': {
            'real': periodo_data['curva_real'],
            'expected': periodo_data['curva_esperada']
        },
        'laboratorios': {'real': periodo_data['energia_real'] * 0.30, 'expected': periodo_data['energia_esperada'] * 0.30, 'delta': periodo_data['delta'] * 0.30, 'delta_percent': periodo_data['delta_pct']},
        'oficinas': {'real': periodo_data['energia_real'] * 0.25, 'expected': periodo_data['energia_esperada'] * 0.25, 'delta': periodo_data['delta'] * 0.25, 'delta_percent': periodo_data['delta_pct']},
        'salones': {'real': periodo_data['energia_real'] * 0.25, 'expected': periodo_data['energia_esperada'] * 0.25, 'delta': periodo_data['delta'] * 0.25, 'delta_percent': periodo_data['delta_pct']},
        'comedores': {'real': periodo_data['energia_real'] * 0.12, 'expected': periodo_data['energia_esperada'] * 0.12, 'delta': periodo_data['delta'] * 0.12, 'delta_percent': periodo_data['delta_pct']},
        'auditorios': {'real': periodo_data['energia_real'] * 0.08, 'expected': periodo_data['energia_esperada'] * 0.08, 'delta': periodo_data['delta'] * 0.08, 'delta_percent': periodo_data['delta_pct']},
        'metadata': {
            'periodo': selected_periodo,
            'num_dias': periodo_data['num_dias'],
            'agua_total': periodo_data['agua_total'],
            'estado': periodo_data['estado']
        }
    }
else:
    # Usar datos de escenario para vista de dia
    data = get_data_from_csv(df_demo, selected_sede, "dia_normal")

# Procesar alertas
alerts = trigger_engine.evaluate(data)

# Header con sede dinámica
sede_nombre = SEDES_CONFIG[selected_sede]['nombre']
sede_icon = SEDES_CONFIG[selected_sede]['icon']

st.markdown(f"""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1>🌿 EcoCampus UPTC - Sede {sede_nombre} {sede_icon}</h1>
            <p>Sistema Inteligente de Gestion Energetica</p>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 2rem; font-weight: 700;">{current_time}</div>
            <div style="opacity: 0.8;">{current_date}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# LAYOUT PRINCIPAL: DOS COLUMNAS
# ==============================================================================
col_dashboard, col_chat = st.columns([2, 1], gap="large")

# ==============================================================================
# COLUMNA IZQUIERDA: DASHBOARD
# ==============================================================================
with col_dashboard:

    # --- ESTADO GENERAL (SEMÁFORO) ---
    delta_total = data['total']['delta_percent']

    if delta_total < -5:
        status_class = "status-green"
        status_text = "🟢 AHORRANDO"
        status_desc = f"Consumo {abs(delta_total):.1f}% por debajo de lo esperado"
    elif delta_total <= 15:
        status_class = "status-yellow"
        status_text = "🟡 ATENCIÓN"
        status_desc = f"Consumo {delta_total:.1f}% sobre lo esperado"
    else:
        status_class = "status-red"
        status_text = "🔴 ALERTA"
        status_desc = f"Consumo {delta_total:.1f}% muy por encima de lo esperado"

    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <span class="status-indicator {status_class}">{status_text}</span>
        <p style="color: #64748b; margin-top: 0.5rem;">{status_desc}</p>
    </div>
    """, unsafe_allow_html=True)

    # --- BALANCE DEL PERIODO ---
    balance_delta = data['total']['delta']
    if balance_delta < 0:
        balance_icon = "✅"
        balance_text = "AHORRO"
        balance_color = "#22c55e"
        balance_msg = f"Se ahorraron ${abs(data['total']['cost']):,.0f} COP"
    else:
        balance_icon = "⚠️"
        balance_text = "EXCESO"
        balance_color = "#ef4444"
        balance_msg = f"Costo adicional: ${abs(data['total']['cost']):,.0f} COP"

    st.markdown(f"""
    <div style="background: {'#f0fdf4' if balance_delta < 0 else '#fef2f2'};
                border: 2px solid {balance_color};
                border-radius: 12px;
                padding: 0.8rem 1.5rem;
                margin-bottom: 1rem;
                display: flex;
                justify-content: space-between;
                align-items: center;">
        <div>
            <span style="font-size: 1.5rem; font-weight: 700; color: {balance_color};">
                {balance_icon} BALANCE: {balance_text}
            </span>
            <span style="margin-left: 1rem; color: #64748b;">
                | Periodo: {PERIODOS_CONFIG[selected_periodo]['nombre']}
            </span>
        </div>
        <div style="font-size: 1.1rem; font-weight: 600; color: {balance_color};">
            {balance_msg}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- METRICAS DEL PERIODO ---
    st.markdown(f"### 📊 Metricas - {PERIODOS_CONFIG[selected_periodo]['nombre']}")

    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Consumo Real</div>
            <div class="value">{data['total']['real']:,.0f}</div>
            <div class="delta delta-neutral">kWh</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Consumo Esperado</div>
            <div class="value">{data['total']['expected']:,.0f}</div>
            <div class="delta delta-neutral">kWh</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        delta = data['total']['delta']
        delta_class = "delta-positive" if delta > 0 else "delta-negative"
        sign = "+" if delta > 0 else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Diferencia</div>
            <div class="value" style="color: {'#ef4444' if delta > 0 else '#22c55e'};">{sign}{delta:,.0f}</div>
            <div class="delta {delta_class}">kWh {'⚠️' if delta > 0 else '✓'}</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        costo = abs(data['total']['cost'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">{'Costo Adicional' if data['total']['delta'] > 0 else 'Ahorro'}</div>
            <div class="value" style="color: {'#ef4444' if data['total']['delta'] > 0 else '#22c55e'};">${costo:,.0f}</div>
            <div class="delta delta-neutral">COP</div>
        </div>
        """, unsafe_allow_html=True)

    with m5:
        co2 = abs(data['total']['co2'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">CO₂ {'Extra' if data['total']['delta'] > 0 else 'Evitado'}</div>
            <div class="value">{co2:,.1f}</div>
            <div class="delta delta-neutral">kg</div>
        </div>
        """, unsafe_allow_html=True)

    
    # --- CONSUMO POR SECTOR ---
    col_sectors, col_hours = st.columns([1.2, 0.8])

    with col_sectors:
        st.markdown("### 📈 Consumo por Sector")

        sectors = ['laboratorios', 'oficinas', 'salones', 'comedores', 'auditorios']
        sector_icons = {
            'laboratorios': '🔬',
            'oficinas': '🏢',
            'salones': '📚',
            'comedores': '🍽️',
            'auditorios': '🎭'
        }
        sector_names = {
            'laboratorios': 'Laboratorios',
            'oficinas': 'Oficinas',
            'salones': 'Salones',
            'comedores': 'Comedores',
            'auditorios': 'Auditorios'
        }

        max_consumption = max([data[s]['real'] for s in sectors])

        for sector in sectors:
            s_data = data[sector]
            delta_pct = s_data['delta_percent']

            # Determinar estado
            if delta_pct > 25:
                card_class = "alert"
                status = "⚠️ ALERTA"
                bar_color = "#ef4444"
            elif delta_pct > 10:
                card_class = "warning"
                status = "⚠️ Revisar"
                bar_color = "#f59e0b"
            elif delta_pct < -5:
                card_class = "success"
                status = "✓ Ahorrando"
                bar_color = "#22c55e"
            else:
                card_class = ""
                status = "✓ Normal"
                bar_color = "#3b82f6"

            progress_width = (s_data['real'] / max_consumption) * 100

            st.markdown(f"""
            <div class="sector-card {card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 1.2rem;">{sector_icons[sector]}</span>
                        <strong style="margin-left: 0.5rem;">{sector_names[sector]}</strong>
                    </div>
                    <div style="text-align: right;">
                        <strong>{s_data['real']:,.0f} kWh</strong>
                        <span style="color: {'#ef4444' if delta_pct > 0 else '#22c55e'}; font-size: 0.9rem; margin-left: 0.5rem;">
                            {'+' if delta_pct > 0 else ''}{delta_pct:.0f}%
                        </span>
                    </div>
                </div>
                <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.3rem;">
                    Esperado: {s_data['expected']:,.0f} kWh → {status}
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {progress_width}%; background: {bar_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_hours:
        st.markdown("### ⏰ Horarios Críticos")

        time_slots = [
            {"time": "10:00-12:00", "label": "Pico mañana", "kwh": 890, "level": "red"},
            {"time": "14:00-16:00", "label": "Pico tarde", "kwh": 820, "level": "red"},
            {"time": "08:00-10:00", "label": "Alto", "kwh": 650, "level": "yellow"},
            {"time": "22:00-06:00", "label": "Mínimo", "kwh": 180, "level": "green"},
        ]

        for slot in time_slots:
            st.markdown(f"""
            <div class="time-slot">
                <div class="dot dot-{slot['level']}"></div>
                <div style="flex: 1;">
                    <strong>{slot['time']}</strong>
                    <div style="font-size: 0.8rem; color: #64748b;">{slot['label']}</div>
                </div>
                <div style="font-weight: 600;">{slot['kwh']} kWh</div>
            </div>
            """, unsafe_allow_html=True)

    
    # --- ALERTAS Y PROYECCIÓN ---
    col_alerts, col_savings = st.columns(2)

    with col_alerts:
        st.markdown(f"### 🚨 Alertas Activas ({len(alerts)})")

        if alerts:
            for alert in alerts[:3]:  # Mostrar máximo 3
                icon = "🔴" if alert['severity'] == 'high' else "🟡"
                st.markdown(f"""
                <div class="alert-item">
                    <div class="alert-icon">{icon}</div>
                    <div class="alert-content">
                        <div class="alert-title">{alert['sector']} - {alert['title']}</div>
                        <div class="alert-desc">{alert['description']}</div>
                        <div style="margin-top: 0.5rem; font-weight: 600; color: #ef4444;">
                            Impacto: ${alert['cost']:,.0f} COP
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No hay alertas activas. Todo está funcionando correctamente.")

    with col_savings:
        st.markdown("### 💰 Proyección de Ahorro")

        # Calcular proyección
        potential_savings_kwh = sum([max(0, data[s]['delta']) for s in sectors]) * 0.7
        potential_savings_cop = potential_savings_kwh * 650
        potential_co2 = potential_savings_kwh * 0.164

        st.markdown(f"""
        <div class="savings-card">
            <div class="savings-title">Si implementa las recomendaciones:</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div>
                    <div style="font-size: 0.85rem; color: #166534;">Reducción estimada</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #166534;">{potential_savings_kwh:,.0f} kWh/día</div>
                </div>
                <div>
                    <div style="font-size: 0.85rem; color: #166534;">CO₂ evitado</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #166534;">{potential_co2:,.1f} kg/día</div>
                </div>
            </div>
            <hr style="border-color: #22c55e; opacity: 0.3; margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 0.85rem; color: #166534;">Ahorro proyectado</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #166534;">${potential_savings_cop:,.0f}/día</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.85rem; color: #166534;">📅 Mensual</div>
                    <div style="font-size: 1.2rem; font-weight: 600; color: #166534;">${potential_savings_cop * 30:,.0f}</div>
                </div>
            </div>
            <div style="margin-top: 1rem; font-size: 0.9rem; color: #166534;">
                🌳 Equivale a: {int(potential_co2 * 30 / 21)} árboles absorbiendo CO₂ por un año
            </div>
        </div>
        """, unsafe_allow_html=True)


    # --- GRÁFICO TEMPORAL ---
    # Configurar titulo y etiquetas segun periodo
    periodo_config = PERIODOS_CONFIG.get(selected_periodo, PERIODOS_CONFIG['dia'])
    periodo_titulo = periodo_config['nombre']
    x_label = periodo_config['x_label']

    # Generar etiquetas para el eje X segun periodo
    curva_real = data['hourly']['real']
    curva_esperada = data['hourly']['expected']
    num_puntos = len(curva_real)

    if selected_periodo == 'dia':
        x_vals = list(range(24))
        x_labels = [f"{h}:00" for h in range(24)]
        titulo_grafico = f"Consumo Real vs Esperado - {periodo_titulo}"
    elif selected_periodo == 'semana':
        x_vals = list(range(num_puntos))
        dias_semana = ['Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab', 'Dom']
        x_labels = dias_semana[:num_puntos]
        titulo_grafico = f"Consumo Real vs Esperado - {periodo_titulo}"
    elif selected_periodo == 'mes':
        x_vals = list(range(num_puntos))
        x_labels = [f"Dia {i+1}" for i in range(num_puntos)]
        titulo_grafico = f"Consumo Real vs Esperado - {periodo_titulo}"
    elif selected_periodo == 'semestre':
        x_vals = list(range(num_puntos))
        x_labels = [f"Sem {i+1}" for i in range(num_puntos)]
        titulo_grafico = f"Consumo Real vs Esperado - {periodo_titulo}"
    else:  # anio
        x_vals = list(range(num_puntos))
        meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        x_labels = meses[:num_puntos]
        titulo_grafico = f"Consumo Real vs Esperado - {periodo_titulo}"

    # Mostrar titulo con R² del modelo
    r2_valor = MODELO_R2.get("energia_total_kwh", 0.54)
    r2_color = "#22c55e" if r2_valor >= 0.8 else "#f59e0b" if r2_valor >= 0.5 else "#ef4444"

    col_titulo, col_r2 = st.columns([3, 1])
    with col_titulo:
        st.markdown(f"### 📉 {titulo_grafico}")
    with col_r2:
        st.markdown(f"""
        <div style="text-align: right; padding: 0.5rem;">
            <span style="font-size: 0.85rem; color: #64748b;">Precision Modelo</span><br>
            <span style="font-size: 1.5rem; font-weight: 700; color: {r2_color};">
                R² = {r2_valor:.2f}
            </span>
        </div>
        """, unsafe_allow_html=True)

    df_chart = pd.DataFrame({
        'X': x_vals,
        'Real': curva_real[:num_puntos],
        'Esperado': curva_esperada[:num_puntos]
    })

    fig = go.Figure()

    # Area de diferencia
    fig.add_trace(go.Scatter(
        x=df_chart['X'],
        y=df_chart['Esperado'],
        fill=None,
        mode='lines',
        line=dict(color='rgba(100, 116, 139, 0.5)', width=2, dash='dash'),
        name='Esperado (Modelo)'
    ))

    fig.add_trace(go.Scatter(
        x=df_chart['X'],
        y=df_chart['Real'],
        fill='tonexty',
        mode='lines',
        line=dict(color='#3b82f6', width=3),
        fillcolor='rgba(59, 130, 246, 0.2)',
        name='Real (Sensor)'
    ))

    # Marcar anomalias (puntos donde real > 30% del esperado)
    for i, (real, expected) in enumerate(zip(df_chart['Real'], df_chart['Esperado'])):
        if real > expected * 1.3:
            fig.add_annotation(
                x=i, y=real,
                text="!",
                showarrow=False,
                font=dict(size=14, color='red')
            )

    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis_title=x_label,
        yaxis_title="Consumo (kWh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    # Configurar ticks del eje X segun periodo
    if selected_periodo == 'dia':
        fig.update_xaxes(gridcolor='#e2e8f0', tickvals=list(range(0, 24, 2)), ticktext=[f"{h}:00" for h in range(0, 24, 2)])
    elif selected_periodo == 'semana':
        fig.update_xaxes(gridcolor='#e2e8f0', tickvals=list(range(num_puntos)), ticktext=x_labels)
    elif selected_periodo == 'anio':
        fig.update_xaxes(gridcolor='#e2e8f0', tickvals=list(range(num_puntos)), ticktext=x_labels)
    else:
        fig.update_xaxes(gridcolor='#e2e8f0')

    fig.update_yaxes(gridcolor='#e2e8f0')

    st.plotly_chart(fig, use_container_width=True)

    # Nota sobre el modelo
    st.markdown(f"""
    <div style="display: flex; gap: 2rem; font-size: 0.8rem; color: #64748b; margin-bottom: 1rem;">
        <div>
            <strong>R² (Coef. Determinacion):</strong> Mide que tan bien el modelo explica la variabilidad de los datos.
            <span style="color: #22c55e;">■ &gt;0.8 Excelente</span> |
            <span style="color: #f59e0b;">■ 0.5-0.8 Aceptable</span> |
            <span style="color: #ef4444;">■ &lt;0.5 Bajo</span>
        </div>
        <div>
            <strong>Modelo:</strong> XGBoost | <strong>Features:</strong> 12 | <strong>Entrenamiento:</strong> 2018-2024
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Boton de reporte
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        st.button("📄 Generar Reporte PDF", use_container_width=True)
    with col_btn2:
        if st.button("🔄 Actualizar Datos", use_container_width=True):
            # Limpiar todos los cachés
            st.cache_data.clear()
            # Reiniciar el chat para reflejar nuevos datos
            if "messages" in st.session_state:
                del st.session_state["messages"]
            st.rerun()

# ==============================================================================
# COLUMNA DERECHA: CHATBOT
# ==============================================================================
with col_chat:
    st.markdown("""
    <div style="background: white; border-radius: 16px; padding: 0.8rem; border: 1px solid #e2e8f0; margin-bottom: 0.5rem;">
        <h3 style="margin: 0; font-size: 1.1rem;">🤖 Asistente EcoCampus</h3>
    </div>
    """, unsafe_allow_html=True)

    # Inicializar historial de chat
    if "messages" not in st.session_state:
        # Mensaje inicial del asistente
        if claude:
            initial_message = claude.get_initial_greeting(data, alerts)
        else:
            initial_message = """Buenos días, Señor Rector. 🌿

El sistema está monitoreando los 5 sectores de la sede Tunja.

⚠️ Hay alertas activas que requieren su atención.

¿En qué puedo ayudarle hoy?

_Nota: El asistente IA está en modo limitado._"""
        st.session_state.messages = [
            {"role": "assistant", "content": initial_message}
        ]

    # Input de chat ARRIBA del contenedor de mensajes
    prompt = st.chat_input("Escriba su pregunta al Rector...")

    # Preguntas sugeridas compactas
    st.markdown("<p style='margin: 0.3rem 0; font-size: 0.8rem; color: #64748b;'>Preguntas sugeridas:</p>", unsafe_allow_html=True)

    suggestions = [
        "¿Qué acciones tomar esta semana?",
        "Patrón de comedores",
        "¿Por qué alerta en laboratorios?",
        "¿Ahorro mensual proyectado?"
    ]

    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": suggestion})
                with st.spinner("Analizando..."):
                    if claude:
                        response = claude.chat(suggestion, data, alerts)
                    else:
                        response = "⚠️ El asistente IA no está disponible."
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

    # Indicador de scroll
    st.markdown("<p style='margin: 0.5rem 0 0.2rem 0; font-size: 0.75rem; color: #94a3b8; text-align: center;'>↕️ Desplácese para ver el historial</p>", unsafe_allow_html=True)

    # Contenedor del chat con scroll - altura fija
    chat_container = st.container(height=480)

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                formatted_content = format_chat_response(message["content"])
                st.markdown(formatted_content, unsafe_allow_html=True)

    # Procesar input si se envió
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Analizando datos..."):
            if claude:
                response = claude.chat(prompt, data, alerts)
            else:
                response = "⚠️ El asistente IA no está disponible."
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# ==============================================================================
# FOOTER (compacto)
# ==============================================================================
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.75rem; padding: 0.5rem 0; border-top: 1px solid #e2e8f0; margin-top: 0.5rem;">
    🌿 EcoCampus UPTC | Hackathon IAMinds 2026 | Equipo NovaIA
</div>
""", unsafe_allow_html=True)
