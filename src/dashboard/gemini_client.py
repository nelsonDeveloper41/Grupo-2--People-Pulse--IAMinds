"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Cliente de Gemini para EcoCampus UPTC                     ║
║              Asistente Inteligente de Gestión Energética                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# PROMPT DEL SISTEMA - CONDICIONA EL COMPORTAMIENTO DEL ASISTENTE
# ==============================================================================
SYSTEM_PROMPT = """
Eres el Asistente EcoCampus, un sistema inteligente de gestión energética para la Universidad Pedagógica y Tecnológica de Colombia (UPTC), sede Tunja.

## TU ROL
Eres el consejero energético del Rector. Tu trabajo es traducir datos técnicos de consumo eléctrico a decisiones ejecutivas claras.

## TU AUDIENCIA
El Rector de la UPTC: un ejecutivo no técnico que:
- Toma decisiones basadas en presupuesto
- Necesita información sintetizada
- Quiere saber: ¿Estamos desperdiciando dinero? ¿Qué hago?
- Responde ante entes de control sobre sostenibilidad

## LOS 5 SECTORES QUE MONITOREAS
1. 🔬 Laboratorios - Mayor consumo, equipos pesados
2. 🏢 Oficinas - Consumo estable, horario fijo
3. 📚 Salones - Varía con ocupación estudiantil
4. 🍽️ Comedores - Refrigeración 24/7, picos en comidas
5. 🎭 Auditorios - Uso esporádico, picos en eventos

## FORMATO DE TUS RESPUESTAS
Siempre usa este formato cuando reportes anomalías o recomendaciones:

**[SECTOR] - Sede Tunja**
"[Descripción del hallazgo con datos específicos]"
**Recomendación:** [Acción concreta]
**Ahorro potencial:** [X] kWh/mes ($[Y] COP)

## FACTORES DE CONVERSIÓN
- Costo por kWh: $650 COP
- Factor CO2: 0.164 kg por kWh
- 1 beca de alimentación = $650,000 COP/mes
- 1 árbol absorbe ~21 kg CO2/año

## REGLAS DE COMUNICACIÓN
1. Siempre saluda al Rector con respeto ("Señor Rector", "Rector")
2. Usa lenguaje ejecutivo, no técnico
3. Siempre cuantifica el impacto en PESOS y CO2
4. Prioriza las recomendaciones (Alta, Media, Baja)
5. Celebra los logros cuando hay ahorro
6. Sé conciso pero completo
7. Usa emojis con moderación para indicar estado (🟢🟡🔴)
8. Termina con una acción clara

## EJEMPLOS DE RESPUESTAS

### Cuando hay alerta:
"Señor Rector, detectamos consumo anómalo en **Laboratorios** entre las 2-5 AM, cuando no debería haber actividad. Esto representa $156,000 COP de desperdicio diario.

**Recomendación:** Verificar con vigilancia si hay equipos encendidos sin autorización.
**Ahorro potencial:** 240 kWh/día ($156,000 COP)"

### Cuando preguntan por un sector:
"**Patrón de COMEDORES - Sede Tunja**

El consumo de refrigeración entre 2-5am es 35% superior al baseline.

**Recomendación:** Verificar termostatos y estado de empaques de refrigeradores.
**Ahorro potencial:** 120 kWh/mes ($78,000 COP)"

### Cuando hay buen desempeño:
"¡Excelentes noticias, Rector! 🟢

Los **Salones** están operando 8% por debajo del consumo esperado. Esto representa un ahorro de $45,000 COP hoy.

El protocolo de apagado nocturno está funcionando correctamente. Sugiero replicar esta práctica en Oficinas."

## CONTEXTO ACTUAL
Recibirás datos en tiempo real del sistema. Úsalos para dar respuestas precisas y contextualizadas.
"""


class GeminiAssistant:
    """Cliente para interactuar con Gemini API como asistente de gestión energética"""

    def __init__(self):
        """Inicializa el cliente de Gemini"""
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("No se encontró GOOGLE_API_KEY en el archivo .env")

        # Limpiar la API key (quitar comillas si las tiene)
        api_key = api_key.strip().strip('"').strip("'")

        genai.configure(api_key=api_key)

        # Configuración del modelo
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 4096,  # Aumentado para respuestas completas
        }

        # Configuración de seguridad más permisiva
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        # Lista de modelos a intentar (en orden de preferencia) - Actualizado Enero 2026
        model_candidates = [
            "models/gemini-2.0-flash",
            "models/gemini-2.5-flash",
            "models/gemini-flash-latest",
            "models/gemini-2.0-flash-001",
            "models/gemini-pro-latest",
            "models/gemini-2.5-pro",
        ]

        self.model = None
        self.model_name = None

        # Intentar cada modelo hasta que uno funcione
        for model_name in model_candidates:
            try:
                self.model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings,
                )
                # Probar que el modelo funciona
                test_response = self.model.generate_content("Hola")
                if test_response:
                    self.model_name = model_name
                    print(f"✅ Modelo Gemini conectado: {model_name}")
                    break
            except Exception as e:
                print(f"⚠️ Modelo {model_name} no disponible: {str(e)[:50]}")
                continue

        if self.model is None:
            raise ValueError("No se pudo conectar a ningún modelo de Gemini")

        # Historial de conversación
        self.chat_session = self.model.start_chat(history=[])
        self.system_context = SYSTEM_PROMPT

    def _format_data_context(self, data: dict, alerts: list) -> str:
        """Formatea los datos del sistema para el contexto del LLM"""

        context = f"""
## DATOS ACTUALES DEL SISTEMA (Tiempo Real)

### Estado General de Sede Tunja
- Consumo Real Total: {data['total']['real']:,.0f} kWh
- Consumo Esperado: {data['total']['expected']:,.0f} kWh
- Diferencia: {data['total']['delta']:+,.0f} kWh ({data['total']['delta_percent']:+.1f}%)
- Impacto Económico: ${data['total']['cost']:,.0f} COP
- Impacto Ambiental: {data['total']['co2']:.1f} kg CO2

### Consumo por Sector
"""
        sectors = ['laboratorios', 'oficinas', 'salones', 'comedores', 'auditorios']
        sector_names = {
            'laboratorios': '🔬 Laboratorios',
            'oficinas': '🏢 Oficinas',
            'salones': '📚 Salones',
            'comedores': '🍽️ Comedores',
            'auditorios': '🎭 Auditorios'
        }

        for sector in sectors:
            if sector in data:
                s = data[sector]
                context += f"""
**{sector_names[sector]}**
- Real: {s['real']:,.0f} kWh | Esperado: {s['expected']:,.0f} kWh
- Diferencia: {s['delta']:+,.0f} kWh ({s['delta_percent']:+.1f}%)
"""

        if alerts:
            context += "\n### Alertas Activas\n"
            for alert in alerts:
                context += f"- **{alert['sector']}**: {alert['title']} - {alert['description']} (Impacto: ${alert['cost']:,.0f} COP)\n"
        else:
            context += "\n### Alertas Activas\nNo hay alertas activas.\n"

        return context

    def get_initial_greeting(self, data: dict, alerts: list) -> str:
        """Genera el saludo inicial basado en el estado actual"""

        context = self._format_data_context(data, alerts)

        prompt = f"""
{self.system_context}

---
{context}
---

Genera un saludo inicial breve para el Rector que incluya:
1. Un saludo cordial con la hora del día (Buenos días/tardes/noches según la hora)
2. Resumen del estado general (1-2 oraciones)
3. Si hay alertas, menciona la más importante
4. Si hay buen desempeño en algún sector, menciónalo brevemente
5. Termina invitando a hacer preguntas

Sé conciso (máximo 4-5 oraciones).
"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            # Mensaje de fallback
            hora = "días" if 5 <= __import__('datetime').datetime.now().hour < 12 else "tardes" if 12 <= __import__('datetime').datetime.now().hour < 19 else "noches"
            alert_msg = f"⚠️ Hay {len(alerts)} alerta(s) que requieren su atención." if alerts else "✅ Todos los sectores operan dentro de parámetros normales."

            return f"""Buenos {hora}, Señor Rector. 🌿

El sistema está monitoreando los 5 sectores de la sede Tunja.
{alert_msg}

¿En qué puedo ayudarle hoy?"""

    def chat(self, user_message: str, data: dict, alerts: list) -> str:
        """Procesa un mensaje del usuario y genera una respuesta"""

        context = self._format_data_context(data, alerts)

        full_prompt = f"""
{self.system_context}

---
{context}
---

PREGUNTA DEL RECTOR: {user_message}

Responde de manera clara, concisa y ejecutiva siguiendo el formato indicado en las instrucciones.
Siempre incluye datos específicos del contexto proporcionado.
"""

        try:
            response = self.chat_session.send_message(full_prompt)
            return response.text
        except Exception as e:
            # Intentar sin historial de chat
            try:
                response = self.model.generate_content(full_prompt)
                return response.text
            except Exception as e2:
                return f"""Disculpe, Rector. Hubo un problema al procesar su consulta.

Por favor, intente de nuevo o reformule su pregunta.

_Error técnico: {str(e2)}_"""

    def analyze_sector(self, sector: str, data: dict) -> str:
        """Genera un análisis detallado de un sector específico"""

        s = data.get(sector, {})
        if not s:
            return f"No se encontraron datos para el sector {sector}."

        prompt = f"""
{self.system_context}

---
Genera un análisis detallado del sector {sector.upper()} con estos datos:
- Consumo Real: {s.get('real', 0):,.0f} kWh
- Consumo Esperado: {s.get('expected', 0):,.0f} kWh
- Diferencia: {s.get('delta', 0):+,.0f} kWh ({s.get('delta_percent', 0):+.1f}%)

Incluye:
1. Evaluación del estado actual
2. Posibles causas de la desviación
3. Recomendación específica
4. Ahorro potencial cuantificado

Usa el formato estándar de reporte.
"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error al analizar el sector: {str(e)}"


# ==============================================================================
# TEST
# ==============================================================================
if __name__ == "__main__":
    # Datos de prueba
    test_data = {
        'total': {'real': 3450, 'expected': 3080, 'delta': 370, 'delta_percent': 12, 'cost': 240500, 'co2': 60.7},
        'laboratorios': {'real': 1340, 'expected': 1000, 'delta': 340, 'delta_percent': 34},
        'oficinas': {'real': 620, 'expected': 600, 'delta': 20, 'delta_percent': 3},
        'salones': {'real': 580, 'expected': 600, 'delta': -20, 'delta_percent': -3},
        'comedores': {'real': 710, 'expected': 680, 'delta': 30, 'delta_percent': 4},
        'auditorios': {'real': 200, 'expected': 200, 'delta': 0, 'delta_percent': 0},
        'hourly': {'real': [100]*24, 'expected': [90]*24}
    }

    test_alerts = [
        {'sector': 'Laboratorios', 'title': 'Pico anómalo', 'description': '+340 kWh desde 8AM', 'cost': 221000, 'severity': 'high'}
    ]

    print("Inicializando asistente...")
    assistant = GeminiAssistant()

    print("\n=== SALUDO INICIAL ===")
    greeting = assistant.get_initial_greeting(test_data, test_alerts)
    print(greeting)

    print("\n=== PRUEBA DE CHAT ===")
    response = assistant.chat("Muéstrame el patrón de consumo de comedores", test_data, test_alerts)
    print(response)
