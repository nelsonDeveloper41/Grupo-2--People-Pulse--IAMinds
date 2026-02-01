"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🌿 ECOCAMPUS UPTC - Sede Tunja                            ║
║              Sistema Inteligente de Gestión Energética                       ║
║                      Hackathon IAMinds 2026                                  ║
║         Integración con n8n + Telegram + IA (Groq)                           ║
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
import requests
import json
from dotenv import load_dotenv
import re



# Importar módulos propios
try:
    from data_simulator import DataSimulator
    from triggers import TriggerEngine
except ImportError:
    st.warning("⚠️ Módulos no encontrados. Usando datos de prueba.")
    DataSimulator = None
    TriggerEngine = None



# Cargar variables de entorno
load_dotenv()



# ==============================================================================
# CONFIGURACIÓN DE n8n
# ==============================================================================
N8N_WEBHOOK_URL = os.getenv(
    "N8N_WEBHOOK_URL",
    "https://anitra-nonpaid-shavon.ngrok-free.dev/webhook/ee3226db-2aba-4316-a3e1-b18bb9d1c042"
)



# ==============================================================================
# FUNCIONES DE INTEGRACIÓN CON n8n
# ==============================================================================
def enviar_alerta_a_n8n(evento_data: dict, contexto_datos: dict = None) -> dict:
    """Envía alerta a n8n para procesamiento con IA (Groq) y Telegram CON DATOS COMPLETOS"""
    if not N8N_WEBHOOK_URL:
        return {"status": "error", "error": "N8N_WEBHOOK_URL no configurado"}
    
    # Usar contexto_datos o data global si no se proporciona
    if contexto_datos is None:
        # Obtener datos globales
        try:
            contexto_datos = st.session_state.get('contexto_datos', {})
        except:
            contexto_datos = {}
    
    sectors = ['laboratorios', 'oficinas', 'salones', 'comedores', 'auditorios']
    
    # Construir mensaje SIEMPRE con datos completos
    mensaje_completo = f"""📋 ALERTA DEL SISTEMA ECOCAMPUS UPTC

🔔 TIPO DE ALERTA: {evento_data.get('alerta_type', 'General')}
📍 SECTOR AFECTADO: {evento_data.get('sector', 'Unknown')}
📝 DESCRIPCIÓN: {evento_data.get('mensaje', 'Sin descripción')}
⚡ URGENCIA: {evento_data.get('urgencia', 'media').upper()}

═══════════════════════════════════════════════════════════════

📊 CONTEXTO ACTUAL DEL SISTEMA ECOCAMPUS:

CONSUMO GENERAL:
• Consumo Real Total: {contexto_datos.get('total', {}).get('real', 'N/A'):,.0f} kWh
• Consumo Esperado: {contexto_datos.get('total', {}).get('expected', 'N/A'):,.0f} kWh
• Desviación: {contexto_datos.get('total', {}).get('delta_percent', 'N/A'):+.1f}%
• Impacto Económico: ${abs(contexto_datos.get('total', {}).get('cost', 0)):,.0f} COP
• Huella de Carbono: {abs(contexto_datos.get('total', {}).get('co2', 0)):.1f} kg CO₂

DESGLOSE POR SECTOR:
"""
    
    for s in sectors:
        if s in contexto_datos:
            real = contexto_datos[s].get('real', 0)
            expected = contexto_datos[s].get('expected', 0)
            delta_pct = contexto_datos[s].get('delta_percent', 0)
            mensaje_completo += f"  • {s.upper()}: {real:,.0f} kWh (Esperado: {expected:,.0f} | {delta_pct:+.1f}%)\n"
        else:
            mensaje_completo += f"  • {s.upper()}: Datos no disponibles\n"
    
    mensaje_completo += f"""
═══════════════════════════════════════════════════════════════

⏰ TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    payload = {
        "timestamp": datetime.now().isoformat(),
        "sector": evento_data.get("sector", "Unknown"),
        "consumo": evento_data.get("consumo", 0),
        "alerta_type": evento_data.get("alerta_type", "General"),
        "mensaje": mensaje_completo,  # ✅ AHORA SÍ SIEMPRE TIENE DATOS
        "urgencia": evento_data.get("urgencia", "media")
    }
    
    try:
        response = requests.post(
            N8N_WEBHOOK_URL,
            json=payload,
            timeout=10,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        if response.status_code == 200:
            return {"status": "success", "data": response.json()}
        else:
            return {"status": "error", "error": f"HTTP {response.status_code}"}
    except requests.exceptions.Timeout:
        return {"status": "error", "error": "Timeout - n8n no respondió"}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "error": "No se pudo conectar a n8n"}
    except Exception as e:
        return {"status": "error", "error": str(e)}



def enviar_pregunta_a_n8n(pregunta: str, contexto_datos: dict) -> dict:
    """Envía pregunta al asistente n8n + Groq para análisis"""
    if not N8N_WEBHOOK_URL:
        return {"status": "error", "error": "N8N_WEBHOOK_URL no configurado"}
    
    sectors = ['laboratorios', 'oficinas', 'salones', 'comedores', 'auditorios']
    
    # Construir mensaje completo para el LLM
    mensaje_completo = f"""PREGUNTA DEL RECTOR: {pregunta}



DATOS ACTUALES DEL SISTEMA:
- Consumo Real Total: {contexto_datos['total']['real']:,.0f} kWh
- Consumo Esperado Total: {contexto_datos['total']['expected']:,.0f} kWh
- Desviación: {contexto_datos['total']['delta_percent']:.1f}%
- Costo Impacto: ${abs(contexto_datos['total']['cost']):,.0f} COP
- CO2: {abs(contexto_datos['total']['co2']):.1f} kg



CONSUMO POR SECTOR:
"""
    for s in sectors:
        mensaje_completo += f"- {s.capitalize()}: {contexto_datos[s]['real']:,.0f} kWh ({contexto_datos[s]['delta_percent']:+.0f}%)\n"
    
    alertas = contexto_datos.get('alertas', [])
    if alertas:
        mensaje_completo += f"\nAlertas Activas: {len(alertas)}\n"
    
    payload = {
        "sector": "Consulta General",
        "consumo": contexto_datos['total']['real'],
        "alerta_type": "Pregunta del Rector",
        "mensaje": mensaje_completo,
        "urgencia": "alta"
    }
    
    try:
        response = requests.post(
            N8N_WEBHOOK_URL,
            json=payload,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            try:
                resp_data = response.json()
                
                if isinstance(resp_data, dict):
                    texto_respuesta = (
                        resp_data.get("message") or 
                        resp_data.get("text") or 
                        resp_data.get("output") or
                        json.dumps(resp_data, ensure_ascii=False)
                    )
                    
                    if isinstance(texto_respuesta, str) and texto_respuesta.strip().startswith('{'):
                        try:
                            parsed = json.loads(texto_respuesta)
                            texto_respuesta = parsed.get("text", texto_respuesta)
                        except:
                            pass
                    
                    return {"status": "success", "respuesta": str(texto_respuesta)}
                else:
                    return {"status": "success", "respuesta": str(resp_data)}
                    
            except json.JSONDecodeError:
                return {"status": "success", "respuesta": response.text}
                
        elif response.status_code == 500:
            error_msg = "Error interno en n8n. Verifica:\n"
            error_msg += "1. Que el flujo esté activo\n"
            error_msg += "2. Que las credenciales de Groq sean válidas\n"
            error_msg += "3. Que el prompt no pida formato JSON\n"
            error_msg += "4. Logs en n8n para más detalles"
            return {"status": "error", "error": error_msg}
        else:
            return {"status": "error", "error": f"HTTP {response.status_code}: {response.text[:200]}"}
            
    except requests.exceptions.Timeout:
        return {"status": "error", "error": "⏱️ n8n tardó más de 30 segundos. El flujo puede estar procesando."}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "error": "❌ No se pudo conectar a n8n. Verifica que ngrok esté activo."}
    except Exception as e:
        return {"status": "error", "error": f"Error inesperado: {str(e)}"}



# ==============================================================================
# FUNCIONES DE FORMATO
# ==============================================================================
def format_chat_response(text: str) -> str:
    """Formatea respuesta markdown a HTML"""
    if not text:
        return text



    lines = text.split('\n')
    formatted_lines = []
    in_list = False



    for line in lines:
        stripped = line.strip()



        if stripped.startswith('### '):
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            content = stripped[4:]
            formatted_lines.append(f'<h4 style="margin: 1rem 0 0.5rem 0; font-size: 1rem; color: #1e293b; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.3rem;">{content}</h4>')
        elif stripped.startswith('## '):
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            content = stripped[3:]
            formatted_lines.append(f'<h3 style="margin: 1.2rem 0 0.5rem 0; font-size: 1.1rem; color: #166534; font-weight: 700;">{content}</h3>')
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
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', text)
    text = text.replace('\n\n', '</p><p style="margin: 0.8rem 0;">')
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
# ESTILOS CSS
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 1.5rem 2rem; border-radius: 0 0 20px 20px;
        margin: -1rem -1rem 1.5rem -1rem; color: white;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; font-weight: 600; }
    .main-header p { margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }
    
    .metric-card {
        background: white; border-radius: 16px; padding: 1.2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08); border: 1px solid #e8ecf1;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(0,0,0,0.12); }
    .metric-card .label { font-size: 0.85rem; color: #64748b; margin-bottom: 0.3rem; font-weight: 500; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #1e293b; }
    .metric-card .delta { font-size: 0.9rem; margin-top: 0.3rem; }
    .delta-positive { color: #ef4444; } .delta-negative { color: #22c55e; } .delta-neutral { color: #64748b; }
    
    .status-indicator {
        display: inline-flex; align-items: center; gap: 0.5rem;
        padding: 0.6rem 1.2rem; border-radius: 50px; font-weight: 600; font-size: 1rem;
    }
    .status-green { background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); color: #166534; border: 2px solid #22c55e; }
    .status-yellow { background: linear-gradient(135deg, #fef9c3 0%, #fef08a 100%); color: #854d0e; border: 2px solid #eab308; }
    .status-red { background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); color: #991b1b; border: 2px solid #ef4444; }
    
    .sector-card {
        background: white; border-radius: 12px; padding: 1rem; margin-bottom: 0.8rem;
        border-left: 4px solid #3b82f6; box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    }
    .sector-card.alert { border-left-color: #ef4444; background: #fef2f2; }
    .sector-card.warning { border-left-color: #f59e0b; background: #fffbeb; }
    .sector-card.success { border-left-color: #22c55e; background: #f0fdf4; }
    
    .progress-bar { height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden; margin-top: 0.5rem; }
    .progress-fill { height: 100%; border-radius: 4px; transition: width 0.3s ease; }
    
    .alert-item {
        background: white; border-radius: 12px; padding: 1rem; margin-bottom: 0.8rem;
        border: 1px solid #e2e8f0; display: flex; align-items: flex-start; gap: 1rem;
    }
    .alert-icon { font-size: 1.5rem; }
    .alert-content { flex: 1; }
    .alert-title { font-weight: 600; color: #1e293b; margin-bottom: 0.2rem; }
    .alert-desc { font-size: 0.9rem; color: #64748b; }
    
    .savings-card {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-radius: 16px; padding: 1.5rem; border: 2px solid #22c55e;
    }
    .savings-title { color: #166534; font-weight: 600; font-size: 1.1rem; margin-bottom: 1rem; }
    
    .time-slot {
        display: flex; align-items: center; padding: 0.6rem;
        background: white; border-radius: 8px; margin-bottom: 0.5rem; border: 1px solid #e2e8f0;
    }
    .time-slot .dot { width: 12px; height: 12px; border-radius: 50%; margin-right: 0.8rem; }
    .dot-red { background: #ef4444; } .dot-yellow { background: #f59e0b; } .dot-green { background: #22c55e; }
</style>
""", unsafe_allow_html=True)



# ==============================================================================
# INICIALIZACIÓN
# ==============================================================================
@st.cache_data(ttl=60)
def get_simulated_data():
    """Obtiene datos simulados"""
    try:
        if DataSimulator:
            simulator = DataSimulator()
            return simulator.generate_scenario('mixed')
    except Exception as e:
        st.warning(f"⚠️ Error al generar datos reales: {str(e)}")
    
    # Datos de prueba por defecto
    return {
        'total': {'real': 4850, 'expected': 4200, 'delta': 650, 'delta_percent': 15.5, 'cost': 422500, 'co2': 106.7},
        'laboratorios': {'real': 1200, 'expected': 950, 'delta': 250, 'delta_percent': 26.3},
        'oficinas': {'real': 980, 'expected': 890, 'delta': 90, 'delta_percent': 10.1},
        'salones': {'real': 850, 'expected': 800, 'delta': 50, 'delta_percent': 6.3},
        'comedores': {'real': 650, 'expected': 550, 'delta': 100, 'delta_percent': 18.2},
        'auditorios': {'real': 170, 'expected': 10, 'delta': 160, 'delta_percent': 1600},
        'hourly': {'real': list(np.random.uniform(150, 250, 24)), 'expected': list(np.random.uniform(160, 220, 24))}
    }



data = get_simulated_data()

# Guardar en session_state para que enviar_alerta_a_n8n pueda acceder
st.session_state['contexto_datos'] = data


# Obtener alertas
alerts = []
try:
    if TriggerEngine:
        trigger_engine = TriggerEngine()
        alerts = trigger_engine.evaluate(data)
except:
    # Alertas de prueba
    alerts = [
        {'sector': 'Laboratorios', 'title': 'Consumo Crítico', 'description': 'Supera umbral de 1200 kWh', 'severity': 'high', 'cost': 162500},
        {'sector': 'Comedores', 'title': 'Consumo Elevado', 'description': 'Consumo 18% por encima de lo esperado', 'severity': 'medium', 'cost': 65000}
    ]



data['alertas'] = alerts



# ==============================================================================
# HEADER
# ==============================================================================
current_time = datetime.now().strftime("%H:%M")
current_date = datetime.now().strftime("%d de %B, %Y")



st.markdown(f"""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1>🌿 EcoCampus UPTC - Sede Tunja</h1>
            <p>Sistema Inteligente de Gestión Energética</p>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 2rem; font-weight: 700;">{current_time}</div>
            <div style="opacity: 0.8;">{current_date}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)



# ==============================================================================
# LAYOUT PRINCIPAL
# ==============================================================================
col_dashboard, col_chat = st.columns([2, 1], gap="large")



# ==============================================================================
# DASHBOARD
# ==============================================================================
with col_dashboard:
    delta_total = data['total']['delta_percent']
    
    if delta_total < -5:
        status_class, status_text = "status-green", "🟢 AHORRANDO"
        status_desc = f"Consumo {abs(delta_total):.1f}% por debajo de lo esperado"
    elif delta_total <= 15:
        status_class, status_text = "status-yellow", "🟡 ATENCIÓN"
        status_desc = f"Consumo {delta_total:.1f}% sobre lo esperado"
    else:
        status_class, status_text = "status-red", "🔴 ALERTA"
        status_desc = f"Consumo {delta_total:.1f}% muy por encima de lo esperado"



    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <span class="status-indicator {status_class}">{status_text}</span>
        <p style="color: #64748b; margin-top: 0.5rem;">{status_desc}</p>
    </div>
    """, unsafe_allow_html=True)



    st.markdown("### 📊 Métricas del Día")
    m1, m2, m3, m4, m5 = st.columns(5)



    with m1:
        st.markdown(f"""<div class="metric-card"><div class="label">Consumo Real</div>
        <div class="value">{data['total']['real']:,.0f}</div><div class="delta delta-neutral">kWh</div></div>""", unsafe_allow_html=True)



    with m2:
        st.markdown(f"""<div class="metric-card"><div class="label">Consumo Esperado</div>
        <div class="value">{data['total']['expected']:,.0f}</div><div class="delta delta-neutral">kWh</div></div>""", unsafe_allow_html=True)



    with m3:
        delta = data['total']['delta']
        sign = "+" if delta > 0 else ""
        delta_class = "delta-positive" if delta > 0 else "delta-negative"
        st.markdown(f"""<div class="metric-card"><div class="label">Diferencia</div>
        <div class="value" style="color: {'#ef4444' if delta > 0 else '#22c55e'};">{sign}{delta:,.0f}</div>
        <div class="delta {delta_class}">kWh {'⚠️' if delta > 0 else '✓'}</div></div>""", unsafe_allow_html=True)



    with m4:
        costo = abs(data['total']['cost'])
        st.markdown(f"""<div class="metric-card"><div class="label">{'Costo Adicional' if data['total']['delta'] > 0 else 'Ahorro'}</div>
        <div class="value" style="color: {'#ef4444' if data['total']['delta'] > 0 else '#22c55e'};">${costo:,.0f}</div>
        <div class="delta delta-neutral">COP</div></div>""", unsafe_allow_html=True)



    with m5:
        co2 = abs(data['total']['co2'])
        st.markdown(f"""<div class="metric-card"><div class="label">CO₂ {'Extra' if data['total']['delta'] > 0 else 'Evitado'}</div>
        <div class="value">{co2:,.1f}</div><div class="delta delta-neutral">kg</div></div>""", unsafe_allow_html=True)



    col_sectors, col_hours = st.columns([1.2, 0.8])



    with col_sectors:
        st.markdown("### 📈 Consumo por Sector")
        sectors = ['laboratorios', 'oficinas', 'salones', 'comedores', 'auditorios']
        sector_icons = {'laboratorios': '🔬', 'oficinas': '🏢', 'salones': '📚', 'comedores': '🍽️', 'auditorios': '🎭'}
        sector_names = {'laboratorios': 'Laboratorios', 'oficinas': 'Oficinas', 'salones': 'Salones', 'comedores': 'Comedores', 'auditorios': 'Auditorios'}
        max_consumption = max([data[s]['real'] for s in sectors]) if any([data[s]['real'] for s in sectors]) else 1



        for sector in sectors:
            s_data = data[sector]
            delta_pct = s_data['delta_percent']



            if delta_pct > 25:
                card_class, status, bar_color = "alert", "⚠️ ALERTA", "#ef4444"
            elif delta_pct > 10:
                card_class, status, bar_color = "warning", "⚠️ Revisar", "#f59e0b"
            elif delta_pct < -5:
                card_class, status, bar_color = "success", "✓ Ahorrando", "#22c55e"
            else:
                card_class, status, bar_color = "", "✓ Normal", "#3b82f6"



            progress_width = (s_data['real'] / max_consumption) * 100 if max_consumption > 0 else 0



            st.markdown(f"""
            <div class="sector-card {card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div><span style="font-size: 1.2rem;">{sector_icons[sector]}</span>
                    <strong style="margin-left: 0.5rem;">{sector_names[sector]}</strong></div>
                    <div style="text-align: right;"><strong>{s_data['real']:,.0f} kWh</strong>
                    <span style="color: {'#ef4444' if delta_pct > 0 else '#22c55e'}; font-size: 0.9rem; margin-left: 0.5rem;">
                    {'+' if delta_pct > 0 else ''}{delta_pct:.0f}%</span></div>
                </div>
                <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.3rem;">
                    Esperado: {s_data['expected']:,.0f} kWh → {status}
                </div>
                <div class="progress-bar"><div class="progress-fill" style="width: {progress_width}%; background: {bar_color};"></div></div>
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
            st.markdown(f"""<div class="time-slot"><div class="dot dot-{slot['level']}"></div>
            <div style="flex: 1;"><strong>{slot['time']}</strong>
            <div style="font-size: 0.8rem; color: #64748b;">{slot['label']}</div></div>
            <div style="font-weight: 600;">{slot['kwh']} kWh</div></div>""", unsafe_allow_html=True)



    col_alerts, col_savings = st.columns(2)



    with col_alerts:
        st.markdown(f"### 🚨 Alertas Activas ({len(alerts)})")
        if alerts:
            for i, alert in enumerate(alerts[:3]):
                icon = "🔴" if alert.get('severity') == 'high' else "🟡"
                col_alert_content, col_alert_btn = st.columns([5, 1])
                
                with col_alert_content:
                    st.markdown(f"""<div class="alert-item"><div class="alert-icon">{icon}</div>
                    <div class="alert-content"><div class="alert-title">{alert.get('sector', 'Unknown')} - {alert.get('title', 'Alerta')}</div>
                    <div class="alert-desc">{alert.get('description', 'Sin descripción')}</div>
                    <div style="margin-top: 0.5rem; font-weight: 600; color: #ef4444;">Impacto: ${alert.get('cost', 0):,.0f} COP</div>
                    </div></div>""", unsafe_allow_html=True)
                
                with col_alert_btn:
                    if st.button("📤", key=f"alert_{i}", help="Enviar a n8n + Telegram"):
                        with st.spinner("Enviando..."):
                            sector_key = alert.get('sector', 'laboratorios').lower()
                            if sector_key in data:
                                consumo_value = data[sector_key]['real']
                            else:
                                consumo_value = data['total']['real']
                            
                            result = enviar_alerta_a_n8n({
                                "sector": alert.get('sector', 'Unknown'),
                                "consumo": consumo_value,
                                "alerta_type": alert.get('title', 'Alerta'),
                                "mensaje": alert.get('description', 'Sin descripción'),
                                "urgencia": alert.get('severity', 'medium')
                            }, data)
                            if result['status'] == 'success':
                                st.success("✅ Alerta enviada")
                            else:
                                st.error(f"❌ {result.get('error')}")
        else:
            st.success("✅ No hay alertas activas")



    with col_savings:
        st.markdown("### 💰 Proyección de Ahorro")
        potential_savings_kwh = sum([max(0, data[s]['delta']) for s in sectors]) * 0.7
        potential_savings_cop = potential_savings_kwh * 650
        potential_co2 = potential_savings_kwh * 0.164



        st.markdown(f"""<div class="savings-card"><div class="savings-title">Si implementa las recomendaciones:</div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
            <div><div style="font-size: 0.85rem; color: #166534;">Reducción estimada</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #166534;">{potential_savings_kwh:,.0f} kWh/día</div></div>
            <div><div style="font-size: 0.85rem; color: #166534;">CO₂ evitado</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #166534;">{potential_co2:,.1f} kg/día</div></div>
        </div><hr style="border-color: #22c55e; opacity: 0.3; margin: 1rem 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div><div style="font-size: 0.85rem; color: #166534;">Ahorro proyectado</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #166534;">${potential_savings_cop:,.0f}/día</div></div>
            <div style="text-align: right;"><div style="font-size: 0.85rem; color: #166534;">📅 Mensual</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: #166534;">${potential_savings_cop * 30:,.0f}</div></div>
        </div><div style="margin-top: 1rem; font-size: 0.9rem; color: #166534;">
        🌳 Equivale a: {int(potential_co2 * 30 / 21) if potential_co2 > 0 else 0} árboles absorbiendo CO₂/año</div></div>""", unsafe_allow_html=True)



    st.markdown("### 📉 Consumo Real vs Esperado (24 horas)")
    try:
        hours = list(range(24))
        df_chart = pd.DataFrame({'Hora': hours, 'Real': data['hourly']['real'], 'Esperado': data['hourly']['expected']})
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_chart['Hora'], y=df_chart['Esperado'], fill=None, mode='lines',
            line=dict(color='rgba(100, 116, 139, 0.5)', width=2, dash='dash'), name='Esperado'))
        fig.add_trace(go.Scatter(x=df_chart['Hora'], y=df_chart['Real'], fill='tonexty', mode='lines',
            line=dict(color='#3b82f6', width=3), fillcolor='rgba(59, 130, 246, 0.2)', name='Real'))
        
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0), xaxis_title="Hora del día",
            yaxis_title="Consumo (kWh)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        fig.update_xaxes(gridcolor='#e2e8f0', tickvals=list(range(0, 24, 2)))
        fig.update_yaxes(gridcolor='#e2e8f0')
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Error en gráfico: {str(e)}")



    col_btn1, col_btn2, _ = st.columns([1, 1, 2])
    with col_btn1:
        st.button("📄 Generar Reporte PDF", use_container_width=True)
    with col_btn2:
        if st.button("🔄 Actualizar Datos", use_container_width=True):
            st.cache_data.clear()
            if "messages" in st.session_state:
                del st.session_state["messages"]
            st.rerun()



# ==============================================================================
# CHATBOT CON n8n + GROQ
# ==============================================================================
with col_chat:
    st.markdown("""<div style="background: white; border-radius: 16px; padding: 0.8rem; border: 1px solid #e2e8f0; margin-bottom: 0.5rem;">
    <h3 style="margin: 0; font-size: 1.1rem;">🤖 Asistente EcoCampus</h3>
    <p style="margin: 0.2rem 0 0 0; font-size: 0.75rem; color: #64748b;">Powered by n8n + Groq</p></div>""", unsafe_allow_html=True)



    if "messages" not in st.session_state:
        initial_message = f"""Buenos días, Señor Rector. 🌿



El sistema está monitoreando los 5 sectores de la sede Tunja.



{"⚠️ Hay " + str(len(alerts)) + " alertas activas." if alerts else "✅ Todo funcionando correctamente."}



¿En qué puedo ayudarle hoy?"""
        st.session_state.messages = [{"role": "assistant", "content": initial_message}]



    prompt = st.chat_input("Escriba su pregunta...")



    st.markdown("<p style='margin: 0.3rem 0; font-size: 0.8rem; color: #64748b;'>Preguntas sugeridas:</p>", unsafe_allow_html=True)
    suggestions = ["¿Qué acciones tomar esta semana?", "Patrón de comedores", "¿Por qué alerta en laboratorios?", "¿Ahorro mensual proyectado?"]
    
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": suggestion})
                with st.spinner("🤖 n8n + Groq analizando..."):
                    result = enviar_pregunta_a_n8n(suggestion, data)
                    response = result['respuesta'] if result['status'] == 'success' else f"❌ Error: {result['error']}"
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()



    st.markdown("<p style='margin: 0.5rem 0 0.2rem 0; font-size: 0.75rem; color: #94a3b8; text-align: center;'>↕️ Desplácese para ver el historial</p>", unsafe_allow_html=True)



    chat_container = st.container(height=480)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                formatted_content = format_chat_response(message["content"])
                st.markdown(formatted_content, unsafe_allow_html=True)



    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("🤖 n8n + Groq analizando datos..."):
            result = enviar_pregunta_a_n8n(prompt, data)
            response = result['respuesta'] if result['status'] == 'success' else f"❌ Error: {result['error']}"
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()



# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("""<div style="text-align: center; color: #94a3b8; font-size: 0.75rem; padding: 0.5rem 0; border-top: 1px solid #e2e8f0; margin-top: 0.5rem;">
🌿 EcoCampus UPTC | Hackathon IAMinds 2026 | Equipo NovaIA | ⚡ n8n + Groq + Telegram</div>""", unsafe_allow_html=True)