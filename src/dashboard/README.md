# 🌿 EcoCampus UPTC - Dashboard

Sistema Inteligente de Gestión Energética para la Universidad Pedagógica y Tecnológica de Colombia.

## 🚀 Inicio Rápido

### Opción 1: Windows (Doble clic)
```
Ejecutar: run.bat
```

### Opción 2: Línea de comandos
```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar dashboard
streamlit run app.py
```

El dashboard estará disponible en: **http://localhost:8501**

## 📁 Estructura de Archivos

```
src/dashboard/
├── app.py              # Aplicación principal de Streamlit
├── gemini_client.py    # Cliente para Gemini API (chatbot)
├── data_simulator.py   # Simulador de datos de sensores
├── triggers.py         # Motor de alertas
├── requirements.txt    # Dependencias Python
├── .env               # API Key de Gemini (no compartir)
├── run.bat            # Script de ejecución Windows
├── .streamlit/
│   └── config.toml    # Configuración de Streamlit
└── README.md          # Este archivo
```

## ⚙️ Configuración

### API Key de Gemini
El archivo `.env` debe contener:
```
GOOGLE_API_KEY="tu-api-key-aqui"
```

Obtén tu API key gratis en: https://aistudio.google.com/apikey

## 🎯 Funcionalidades

### Dashboard (Zona Izquierda)
- **Estado General**: Semáforo de estado (🟢🟡🔴)
- **Métricas del Día**: Consumo real vs esperado, costo, CO₂
- **Consumo por Sector**: Los 5 sectores con indicadores
- **Horarios Críticos**: Franjas de mayor consumo
- **Alertas Activas**: Anomalías detectadas
- **Proyección de Ahorro**: Reducción estimada
- **Gráfico Temporal**: Real vs Esperado 24h

### Chatbot (Zona Derecha)
- Asistente inteligente con Gemini
- Respuestas en lenguaje natural
- Preguntas sugeridas
- Análisis personalizado por sector

## 📊 Sectores Monitoreados

| Sector | Icono | Descripción |
|--------|-------|-------------|
| Laboratorios | 🔬 | Mayor consumo, equipos pesados |
| Oficinas | 🏢 | Consumo estable, horario fijo |
| Salones | 📚 | Varía con ocupación estudiantil |
| Comedores | 🍽️ | Refrigeración 24/7, picos en comidas |
| Auditorios | 🎭 | Uso esporádico, picos en eventos |

## 💡 Métricas de Conversión

- **Costo por kWh**: $650 COP
- **Factor CO₂**: 0.164 kg por kWh
- **1 beca alimentación**: $650,000 COP/mes
- **1 árbol**: absorbe ~21 kg CO₂/año

## 🏆 Hackathon IAMinds 2026

**Equipo NovaIA**
- Universidad Pedagógica y Tecnológica de Colombia (UPTC)
- Enero 2026
