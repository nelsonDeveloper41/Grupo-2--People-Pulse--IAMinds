"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Motor de Triggers para EcoCampus UPTC                     ║
║              Detecta anomalías y genera alertas automáticas                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from datetime import datetime
from typing import List, Dict


class TriggerEngine:
    """
    Motor de reglas que evalúa el estado del sistema y genera alertas.
    Implementa los triggers definidos en el documento rector.
    """

    # Factor de conversión
    COST_PER_KWH = 650  # COP

    # Configuración de triggers por sector
    TRIGGER_CONFIG = {
        'laboratorios': {
            'threshold_alert': 25,      # % sobre esperado para ALERTA
            'threshold_warning': 15,    # % sobre esperado para ADVERTENCIA
            'night_hours': [22, 23, 0, 1, 2, 3, 4, 5],
            'triggers': [
                {
                    'id': 'T1',
                    'name': 'Vampiro Nocturno',
                    'condition': 'night_excess',
                    'threshold': 50,  # % sobre esperado en noche
                    'description': 'Consumo nocturno excesivo sin ocupación',
                    'cause': 'Equipos encendidos sin supervisión'
                },
                {
                    'id': 'T3',
                    'name': 'Pico de Laboratorio',
                    'condition': 'lunch_no_drop',
                    'threshold': 20,
                    'description': 'Laboratorios no reducen consumo en almuerzo',
                    'cause': 'Maquinaria en stand-by innecesario'
                }
            ]
        },
        'oficinas': {
            'threshold_alert': 30,
            'threshold_warning': 15,
            'night_hours': [19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7],
            'triggers': [
                {
                    'id': 'T2',
                    'name': 'Fuga de Fin de Semana',
                    'condition': 'weekend_excess',
                    'threshold': 50,
                    'description': 'Consumo elevado en fin de semana',
                    'cause': 'Aires acondicionados o luces olvidadas'
                }
            ]
        },
        'salones': {
            'threshold_alert': 30,
            'threshold_warning': 15,
            'night_hours': [21, 22, 23, 0, 1, 2, 3, 4, 5, 6],
            'triggers': [
                {
                    'id': 'T4',
                    'name': 'Fantasma de Vacaciones',
                    'condition': 'vacation_excess',
                    'threshold': 30,
                    'description': 'Consumo alto en período de vacaciones',
                    'cause': 'Espacios que deberían estar cerrados'
                }
            ]
        },
        'comedores': {
            'threshold_alert': 35,
            'threshold_warning': 20,
            'night_hours': [2, 3, 4, 5],  # Solo madrugada (refrigeración es normal)
            'triggers': [
                {
                    'id': 'T5',
                    'name': 'Refrigeración Excesiva',
                    'condition': 'night_excess',
                    'threshold': 35,
                    'description': 'Consumo de refrigeración muy alto en madrugada',
                    'cause': 'Empaques dañados o termostatos descalibrados'
                }
            ]
        },
        'auditorios': {
            'threshold_alert': 100,  # Muy sensible por ser bajo consumo base
            'threshold_warning': 50,
            'night_hours': list(range(24)),  # Siempre debería ser bajo
            'triggers': [
                {
                    'id': 'T6',
                    'name': 'Evento Fantasma',
                    'condition': 'unexpected_consumption',
                    'threshold': 200,  # kWh absolutos
                    'description': 'Consumo alto sin evento programado',
                    'cause': 'Sistemas de audio/iluminación encendidos sin uso'
                }
            ]
        }
    }

    # Nombres amigables de sectores
    SECTOR_NAMES = {
        'laboratorios': '🔬 Laboratorios',
        'oficinas': '🏢 Oficinas',
        'salones': '📚 Salones',
        'comedores': '🍽️ Comedores',
        'auditorios': '🎭 Auditorios'
    }

    def __init__(self):
        """Inicializa el motor de triggers"""
        self.current_hour = datetime.now().hour
        self.is_weekend = datetime.now().weekday() >= 5
        self.is_vacation = False  # Se podría conectar a un calendario

    def evaluate(self, data: Dict) -> List[Dict]:
        """
        Evalúa todos los triggers contra los datos actuales.

        Args:
            data: Diccionario con datos de todos los sectores

        Returns:
            Lista de alertas generadas
        """
        alerts = []

        for sector, config in self.TRIGGER_CONFIG.items():
            if sector not in data:
                continue

            sector_data = data[sector]
            delta_pct = sector_data.get('delta_percent', 0)

            # Evaluar umbrales generales
            if delta_pct > config['threshold_alert']:
                alert = self._create_alert(
                    sector=self.SECTOR_NAMES.get(sector, sector),
                    title='Consumo excesivo',
                    description=f'+{delta_pct:.0f}% sobre lo esperado',
                    severity='high',
                    delta=sector_data.get('delta', 0)
                )
                alerts.append(alert)

            elif delta_pct > config['threshold_warning']:
                alert = self._create_alert(
                    sector=self.SECTOR_NAMES.get(sector, sector),
                    title='Consumo elevado',
                    description=f'+{delta_pct:.0f}% sobre lo esperado',
                    severity='medium',
                    delta=sector_data.get('delta', 0)
                )
                alerts.append(alert)

            # Evaluar triggers específicos
            for trigger in config.get('triggers', []):
                if self._evaluate_trigger(trigger, sector_data, config):
                    alert = self._create_alert(
                        sector=self.SECTOR_NAMES.get(sector, sector),
                        title=trigger['name'],
                        description=trigger['description'],
                        severity='high' if trigger.get('threshold', 0) > 30 else 'medium',
                        delta=sector_data.get('delta', 0),
                        cause=trigger.get('cause', '')
                    )
                    alerts.append(alert)

        # Ordenar por severidad y costo
        alerts.sort(key=lambda x: (
            0 if x['severity'] == 'high' else 1,
            -x['cost']
        ))

        return alerts

    def _evaluate_trigger(self, trigger: Dict, sector_data: Dict, config: Dict) -> bool:
        """Evalúa si un trigger específico se activa"""
        condition = trigger.get('condition', '')
        threshold = trigger.get('threshold', 0)
        delta_pct = sector_data.get('delta_percent', 0)

        if condition == 'night_excess':
            # Verificar si estamos en horario nocturno y hay exceso
            if self.current_hour in config.get('night_hours', []):
                return delta_pct > threshold

        elif condition == 'weekend_excess':
            # Verificar exceso en fin de semana
            if self.is_weekend:
                return delta_pct > threshold

        elif condition == 'vacation_excess':
            # Verificar exceso en vacaciones
            if self.is_vacation:
                return delta_pct > threshold

        elif condition == 'lunch_no_drop':
            # Verificar si no hay reducción en almuerzo (12-14h)
            if self.current_hour in [12, 13, 14]:
                return delta_pct > threshold

        elif condition == 'unexpected_consumption':
            # Consumo absoluto inesperado
            return sector_data.get('real', 0) > threshold

        return False

    def _create_alert(self, sector: str, title: str, description: str,
                      severity: str, delta: float, cause: str = '') -> Dict:
        """Crea un objeto de alerta estructurado"""
        cost = abs(delta * self.COST_PER_KWH)

        return {
            'sector': sector,
            'title': title,
            'description': description,
            'severity': severity,  # 'high', 'medium', 'low'
            'delta_kwh': delta,
            'cost': cost,
            'cause': cause,
            'timestamp': datetime.now().isoformat(),
            'hour': self.current_hour
        }

    def get_opportunities(self, data: Dict) -> List[Dict]:
        """
        Identifica oportunidades de ahorro (sectores eficientes).

        Args:
            data: Diccionario con datos de todos los sectores

        Returns:
            Lista de oportunidades identificadas
        """
        opportunities = []

        for sector in self.TRIGGER_CONFIG.keys():
            if sector not in data:
                continue

            sector_data = data[sector]
            delta_pct = sector_data.get('delta_percent', 0)

            # Sector está ahorrando
            if delta_pct < -5:
                savings = abs(sector_data.get('delta', 0))
                cost_saved = savings * self.COST_PER_KWH

                opportunity = {
                    'sector': self.SECTOR_NAMES.get(sector, sector),
                    'title': 'Eficiencia sostenida',
                    'description': f'{abs(delta_pct):.0f}% por debajo de lo esperado',
                    'type': 'success',
                    'savings_kwh': savings,
                    'savings_cop': cost_saved,
                    'recommendation': 'Replicar prácticas en otros sectores'
                }
                opportunities.append(opportunity)

        return opportunities


# ==============================================================================
# TEST
# ==============================================================================
if __name__ == "__main__":
    # Datos de prueba
    test_data = {
        'laboratorios': {'real': 1340, 'expected': 1000, 'delta': 340, 'delta_percent': 34},
        'oficinas': {'real': 620, 'expected': 600, 'delta': 20, 'delta_percent': 3},
        'salones': {'real': 580, 'expected': 600, 'delta': -20, 'delta_percent': -3},
        'comedores': {'real': 710, 'expected': 680, 'delta': 30, 'delta_percent': 4},
        'auditorios': {'real': 200, 'expected': 200, 'delta': 0, 'delta_percent': 0},
    }

    engine = TriggerEngine()

    print("=== ALERTAS ===")
    alerts = engine.evaluate(test_data)
    for alert in alerts:
        print(f"[{alert['severity'].upper()}] {alert['sector']}: {alert['title']}")
        print(f"  {alert['description']}")
        print(f"  Costo: ${alert['cost']:,.0f} COP")
        print()

    print("=== OPORTUNIDADES ===")
    opportunities = engine.get_opportunities(test_data)
    for opp in opportunities:
        print(f"[✓] {opp['sector']}: {opp['title']}")
        print(f"  {opp['description']}")
        print(f"  Ahorro: ${opp['savings_cop']:,.0f} COP")
