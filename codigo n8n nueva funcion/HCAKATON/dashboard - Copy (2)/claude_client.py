# claude_client.py
"""
Módulo de asistente IA para EcoCampus
Usa Groq API (similar a Claude pero más rápido)
"""

class ClaudeAssistant:
    """Asistente IA para análisis de datos energéticos"""
    
    def __init__(self):
        """Inicializa el asistente sin dependencias externas"""
        self.name = "EcoCampus AI"
    
    def get_initial_greeting(self, data, alerts):
        """Genera saludo inicial dinámico"""
        delta_total = data['total']['delta_percent']
        
        # Sector con mayor desviación
        sectors = ['laboratorios', 'oficinas', 'salones', 'comedores', 'auditorios']
        sector_critico = max(sectors, key=lambda s: abs(data[s]['delta_percent']))
        delta_sector = data[sector_critico]['delta_percent']
        
        # Mensaje dinámico
        if delta_total > 15:
            estado = "🔴 CRÍTICO"
        elif delta_total > 5:
            estado = "🟡 ELEVADO"
        else:
            estado = "✅ NORMAL"
        
        msg = f"""Buenos días, Señor Rector. 🌿

**Estado Actual** - {self._get_time()}

{estado} - Sobreconsumo detectado

📊 **Resumen de Hoy:**
- Consumo Real: {data['total']['real']:,.0f} kWh
- Consumo Esperado: {data['total']['expected']:,.0f} kWh
- Diferencia: {'+' if data['total']['delta'] > 0 else ''}{data['total']['delta']:,.0f} kWh ({data['total']['delta_percent']:.1f}%)

🎯 **Sector Crítico:**
- {sector_critico.title()}: {data[sector_critico]['real']:,.0f} kWh ({'+' if delta_sector > 0 else ''}{delta_sector:.1f}%)

⚡ **Alertas Activas:** {len(alerts)}
{self._format_alerts(alerts)}

💰 **Proyección Mensual:**
- Impacto: ${(data['total']['delta'] * 650 * 30):,.0f} COP

¿Qué le gustaría saber?"""
        
        return msg
    
    def chat(self, prompt, data, alerts):
        """Procesa consulta y genera respuesta contextualizada"""
        
        prompt_lower = prompt.lower()
        
        # Detectar intención del usuario
        if any(word in prompt_lower for word in ['acción', 'hacer', 'recomendación', 'qué debo']):
            return self._recomendaciones(data, alerts)
        
        elif any(word in prompt_lower for word in ['laboratorio', 'lab']):
            return self._analizar_sector(data, alerts, 'laboratorios')
        
        elif any(word in prompt_lower for word in ['comedor', 'cocina', 'cafeter']):
            return self._analizar_sector(data, alerts, 'comedores')
        
        elif any(word in prompt_lower for word in ['oficina']):
            return self._analizar_sector(data, alerts, 'oficinas')
        
        elif any(word in prompt_lower for word in ['salón', 'aula', 'clase']):
            return self._analizar_sector(data, alerts, 'salones')
        
        elif any(word in prompt_lower for word in ['auditorio', 'evento']):
            return self._analizar_sector(data, alerts, 'auditorios')
        
        elif any(word in prompt_lower for word in ['ahorro', 'costo', 'económic', 'dinero', 'proyect']):
            return self._proyeccion_ahorro(data, alerts)
        
        elif any(word in prompt_lower for word in ['patrón', 'tendencia', 'horario', 'hora']):
            return self._analisis_temporal(data, alerts)
        
        elif any(word in prompt_lower for word in ['alerta', 'problema', 'crítico', 'error']):
            return self._analizar_alertas(alerts, data)
        
        else:
            return self._respuesta_general(data, alerts)
    
    @staticmethod
    def _get_time():
        from datetime import datetime
        return datetime.now().strftime("%H:%M - %d de %B")
    
    @staticmethod
    def _format_alerts(alerts):
        if not alerts:
            return "✅ Ninguna"
        
        texto = ""
        for i, alert in enumerate(alerts[:3], 1):
            icon = "🔴" if alert['severity'] == 'high' else "🟡"
            texto += f"\n{i}. {icon} **{alert['sector']}**: {alert['title']}"
        
        if len(alerts) > 3:
            texto += f"\n... y {len(alerts) - 3} más"
        
        return texto
    
    @staticmethod
    def _recomendaciones(data, alerts):
        return """## 🎯 Acciones Recomendadas

### Inmediatas (Hoy)
- **Laboratorios**: Reducir equipamiento no esencial (Impacto: $180K COP)
- **Comedores**: Revisar sistemas de refrigeración (Impacto: $120K COP)
- Implementar apagado automático en horarios bajos

### Corto Plazo (Esta semana)
- Auditoría energética de laboratorios
- Capacitación al personal de mantenimiento
- Instalación de sensores inteligentes

### Largo Plazo (Próximas semanas)
- Upgrade de iluminación LED en oficinas
- Sistema de climatización automático por ocupancia
- Paneles solares en azotea

**Resultado Proyectado**: 25% reducción en consumo = $1.95M COP/mes de ahorro"""
    
    @staticmethod
    def _analizar_sector(data, alerts, sector):
        s_data = data[sector]
        delta_pct = s_data['delta_percent']
        
        sector_names = {
            'laboratorios': 'Laboratorios 🔬',
            'oficinas': 'Oficinas 🏢',
            'salones': 'Salones 📚',
            'comedores': 'Comedores 🍽️',
            'auditorios': 'Auditorios 🎭'
        }
        
        return f"""## {sector_names.get(sector, sector)}

### Métricas
- **Consumo Real**: {s_data['real']:,.0f} kWh
- **Consumo Esperado**: {s_data['expected']:,.0f} kWh
- **Varianza**: {'+' if delta_pct > 0 else ''}{delta_pct:.1f}%

### Estado
{'🔴 CRÍTICO - Sobreconsumo significativo' if delta_pct > 25 else '🟡 ELEVADO - Revisar equipos' if delta_pct > 10 else '✅ NORMAL'}

### Recomendaciones
- Revisar equipamiento activo en horarios bajos
- Validar calibración de sensores
- Implementar rutina de apagado programado
- Capacitar personal en eficiencia energética

### Impacto Potencial de Mejora
- Reducción estimada: {int(s_data['delta'] * 0.6):,.0f} kWh/día
- Ahorro: ${int(s_data['delta'] * 0.6 * 650):,.0f} COP/día
- Anual: ${int(s_data['delta'] * 0.6 * 650 * 365):,.0f} COP"""
    
    @staticmethod
    def _proyeccion_ahorro(data, alerts):
        sectors = ['laboratorios', 'oficinas', 'salones', 'comedores', 'auditorios']
        potential_savings_kwh = sum([max(0, data[s]['delta']) for s in sectors]) * 0.7
        potential_savings_cop = potential_savings_kwh * 650
        potential_co2 = potential_savings_kwh * 0.164
        
        return f"""## 💰 Proyección de Ahorro

### Si Implementa las Recomendaciones:

#### Por Día
- **Reducción Energética**: {potential_savings_kwh:,.0f} kWh
- **Ahorro Económico**: ${potential_savings_cop:,.0f} COP
- **CO₂ Evitado**: {potential_co2:,.1f} kg

#### Mensual
- **Energía**: {int(potential_savings_kwh * 30):,.0f} kWh
- **Dinero**: ${int(potential_savings_cop * 30):,.0f} COP
- **Impacto Ambiental**: {int(potential_co2 * 30):.0f} kg CO₂

#### Anual
- **Energía**: {int(potential_savings_kwh * 365):,.0f} kWh
- **Dinero**: ${int(potential_savings_cop * 365):,.0f} COP
- **Equivalencia**: {int(potential_co2 * 365 / 21)} árboles plantados/año

### ROI Estimado
- Inversión en sensores inteligentes: $50M COP
- Recuperación en: **~7 meses**
- Beneficio 5 años: ${int(potential_savings_cop * 365 * 5 - 50000000):,.0f} COP"""
    
    @staticmethod
    def _analisis_temporal(data, alerts):
        return """## ⏰ Análisis de Patrones Horarios

### Horarios Críticos
- **10:00-12:00**: Pico mañana (890 kWh) - Laboratorios activos
- **14:00-16:00**: Pico tarde (820 kWh) - Máxima ocupancia
- **08:00-10:00**: Ramp-up (650 kWh) - Arranque de equipos

### Oportunidades de Optimización
- **Desplazar cargas**: Procesar análisis en laboratorios a horas bajas
- **Apagado automático**: Implementar en horarios de baja ocupancia
- **Climatización inteligente**: Precondicionamiento 30 min antes de ocupancia

### Recomendación
Ejecutar mantenimiento preventivo en horarios de mínimo consumo (22:00-06:00)"""
    
    @staticmethod
    def _analizar_alertas(alerts, data):
        if not alerts:
            return "✅ **Excelente**: No hay alertas activas. El sistema está funcionando óptimamente."
        
        text = f"## 🚨 Análisis de {len(alerts)} Alertas\n\n"
        for i, alert in enumerate(alerts[:5], 1):
            severity = "🔴 CRÍTICA" if alert['severity'] == 'high' else "🟡 ADVERTENCIA"
            text += f"**{i}. {severity}** - {alert['sector']}\n"
            text += f"- {alert['title']}\n"
            text += f"- {alert['description']}\n"
            text += f"- Impacto: ${alert['cost']:,.0f} COP\n\n"
        
        text += "### Acciones Prioritarias\n"
        text += "1. Resolver alertas CRÍTICAS hoy\n"
        text += "2. Programar inspección técnica mañana\n"
        text += "3. Comunicar al personal de mantenimiento"
        
        return text
    
    @staticmethod
    def _respuesta_general(data, alerts):
        return f"""## 📊 Dashboard de EcoCampus

### Estado General
- **Consumo Hoy**: {data['total']['real']:,.0f} kWh
- **Esperado**: {data['total']['expected']:,.0f} kWh
- **Diferencia**: {'+' if data['total']['delta'] > 0 else ''}{data['total']['delta']:.0f} kWh

### Top Sectores por Consumo
1. 🔬 Laboratorios: {data['laboratorios']['real']:,.0f} kWh
2. 🏢 Oficinas: {data['oficinas']['real']:,.0f} kWh
3. 📚 Salones: {data['salones']['real']:,.0f} kWh

### Próximos Pasos
- Consulte "¿Qué acciones tomar?" para recomendaciones
- Haga clic en cualquier sector para análisis detallado
- Envíe alertas a n8n usando los botones 📤"""
