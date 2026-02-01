"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Simulador de Datos para EcoCampus UPTC                    ║
║              Genera datos realistas de sensores para la demo                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


class DataSimulator:
    """
    Simula datos de sensores de consumo energético para la sede Tunja.
    Genera patrones realistas con anomalías intencionales para demostrar
    las capacidades del sistema.
    """

    # Factores de conversión
    COST_PER_KWH = 650  # COP
    CO2_PER_KWH = 0.164  # kg

    # Perfiles de consumo por sector (kWh promedio por hora)
    SECTOR_PROFILES = {
        'laboratorios': {
            'base': 30,       # Consumo base (equipos siempre encendidos)
            'peak_factor': 3.5,  # Multiplicador en horas pico
            'peak_hours': [9, 10, 11, 14, 15, 16, 17],  # Horas de mayor uso
            'night_factor': 0.3,  # Factor nocturno (debería ser bajo)
        },
        'oficinas': {
            'base': 20,
            'peak_factor': 2.5,
            'peak_hours': [9, 10, 11, 12, 14, 15, 16, 17],
            'night_factor': 0.1,
        },
        'salones': {
            'base': 25,
            'peak_factor': 3.0,
            'peak_hours': [8, 9, 10, 11, 14, 15, 16, 17, 18],
            'night_factor': 0.05,
        },
        'comedores': {
            'base': 25,  # Refrigeración constante
            'peak_factor': 2.0,
            'peak_hours': [7, 8, 12, 13, 18, 19],  # Horas de comidas
            'night_factor': 0.8,  # Alto por refrigeración
        },
        'auditorios': {
            'base': 5,
            'peak_factor': 8.0,  # Muy alto cuando hay eventos
            'peak_hours': [10, 11, 15, 16, 17],  # Eventos típicos
            'night_factor': 0.02,
        }
    }

    def __init__(self, seed=None):
        """
        Inicializa el simulador.

        Args:
            seed: Semilla para reproducibilidad (None = aleatorio)
        """
        if seed is None:
            # Usar minuto actual como semilla para variación cada minuto
            seed = datetime.now().minute
        np.random.seed(seed)
        random.seed(seed)

    def _generate_hourly_profile(self, sector: str, inject_anomaly: bool = False) -> tuple:
        """
        Genera el perfil de consumo horario para un sector.

        Returns:
            tuple: (consumo_real, consumo_esperado) arrays de 24 horas
        """
        profile = self.SECTOR_PROFILES[sector]
        hours = np.arange(24)

        # Generar consumo esperado (modelo ideal)
        expected = np.zeros(24)

        for hour in range(24):
            if hour in profile['peak_hours']:
                expected[hour] = profile['base'] * profile['peak_factor']
            elif hour >= 22 or hour <= 5:
                expected[hour] = profile['base'] * profile['night_factor']
            else:
                expected[hour] = profile['base'] * 1.2  # Horario normal

        # Generar consumo real (con variación natural)
        noise = np.random.normal(0, profile['base'] * 0.1, 24)
        real = expected + noise

        # Inyectar anomalías si corresponde
        if inject_anomaly:
            anomaly_type = random.choice(['night_spike', 'sustained_high', 'peak_excess'])

            if anomaly_type == 'night_spike':
                # Pico nocturno (equipos olvidados)
                anomaly_hours = random.sample([2, 3, 4], 2)
                for h in anomaly_hours:
                    real[h] = expected[h] * 4  # 300% más de lo esperado

            elif anomaly_type == 'sustained_high':
                # Consumo elevado sostenido
                start_hour = random.randint(8, 14)
                for h in range(start_hour, start_hour + 4):
                    if h < 24:
                        real[h] = expected[h] * 1.5

            elif anomaly_type == 'peak_excess':
                # Exceso en horas pico
                for h in profile['peak_hours']:
                    real[h] = expected[h] * 1.3

        # Asegurar valores positivos
        real = np.maximum(real, 0)
        expected = np.maximum(expected, 0)

        return real, expected

    def _calculate_metrics(self, real: float, expected: float) -> dict:
        """Calcula métricas derivadas"""
        delta = real - expected
        delta_percent = (delta / expected * 100) if expected > 0 else 0
        cost = delta * self.COST_PER_KWH
        co2 = delta * self.CO2_PER_KWH

        return {
            'real': round(real, 1),
            'expected': round(expected, 1),
            'delta': round(delta, 1),
            'delta_percent': round(delta_percent, 1),
            'cost': round(cost, 0),
            'co2': round(co2, 2)
        }

    def generate_current_state(self) -> dict:
        """
        Genera el estado actual completo del sistema.

        Returns:
            dict: Datos de todos los sectores con métricas
        """
        current_hour = datetime.now().hour

        # Determinar qué sectores tienen anomalías
        # Para la demo, inyectamos anomalías en 1-2 sectores
        sectors_with_anomalies = random.sample(
            list(self.SECTOR_PROFILES.keys()),
            k=random.randint(1, 2)
        )

        data = {}
        total_real = 0
        total_expected = 0

        # Generar datos para cada sector
        for sector in self.SECTOR_PROFILES.keys():
            inject_anomaly = sector in sectors_with_anomalies

            hourly_real, hourly_expected = self._generate_hourly_profile(
                sector, inject_anomaly
            )

            # Sumar hasta la hora actual
            sector_real = sum(hourly_real[:current_hour + 1])
            sector_expected = sum(hourly_expected[:current_hour + 1])

            data[sector] = self._calculate_metrics(sector_real, sector_expected)
            data[sector]['hourly_real'] = hourly_real.tolist()
            data[sector]['hourly_expected'] = hourly_expected.tolist()
            data[sector]['has_anomaly'] = inject_anomaly

            total_real += sector_real
            total_expected += sector_expected

        # Calcular totales
        data['total'] = self._calculate_metrics(total_real, total_expected)

        # Agregar datos horarios agregados
        hourly_real_total = np.zeros(24)
        hourly_expected_total = np.zeros(24)

        for sector in self.SECTOR_PROFILES.keys():
            hourly_real_total += np.array(data[sector]['hourly_real'])
            hourly_expected_total += np.array(data[sector]['hourly_expected'])

        data['hourly'] = {
            'real': hourly_real_total.tolist(),
            'expected': hourly_expected_total.tolist()
        }

        # Metadata
        data['timestamp'] = datetime.now().isoformat()
        data['sede'] = 'Tunja'
        data['current_hour'] = current_hour

        return data

    def generate_scenario(self, scenario_type: str) -> dict:
        """
        Genera escenarios específicos para demostración.

        Args:
            scenario_type: 'normal', 'crisis', 'efficient', 'mixed'

        Returns:
            dict: Datos del escenario
        """
        if scenario_type == 'normal':
            # Sin anomalías significativas
            np.random.seed(42)
            return self._generate_normal_scenario()

        elif scenario_type == 'crisis':
            # Múltiples anomalías graves
            return self._generate_crisis_scenario()

        elif scenario_type == 'efficient':
            # Todo por debajo de lo esperado
            return self._generate_efficient_scenario()

        elif scenario_type == 'mixed':
            # Algunos sectores bien, otros mal
            return self._generate_mixed_scenario()

        else:
            return self.generate_current_state()

    def _generate_normal_scenario(self) -> dict:
        """Genera escenario normal sin anomalías"""
        data = {}
        total_real = 0
        total_expected = 0

        for sector in self.SECTOR_PROFILES.keys():
            hourly_real, hourly_expected = self._generate_hourly_profile(sector, False)

            # Ajustar para que esté dentro de ±5%
            adjustment = np.random.uniform(0.97, 1.03, 24)
            hourly_real = hourly_expected * adjustment

            sector_real = sum(hourly_real[:18])  # Hasta las 6 PM
            sector_expected = sum(hourly_expected[:18])

            data[sector] = self._calculate_metrics(sector_real, sector_expected)
            data[sector]['has_anomaly'] = False

            total_real += sector_real
            total_expected += sector_expected

        data['total'] = self._calculate_metrics(total_real, total_expected)
        data['hourly'] = {'real': [], 'expected': []}
        data['timestamp'] = datetime.now().isoformat()
        data['sede'] = 'Tunja'

        return data

    def _generate_crisis_scenario(self) -> dict:
        """Genera escenario de crisis con múltiples alertas"""
        data = {}
        total_real = 0
        total_expected = 0

        crisis_factors = {
            'laboratorios': 1.45,  # 45% sobre lo esperado
            'oficinas': 1.20,
            'salones': 1.10,
            'comedores': 1.35,  # Refrigeración fallando
            'auditorios': 2.0,  # Evento no apagado
        }

        for sector, factor in crisis_factors.items():
            hourly_real, hourly_expected = self._generate_hourly_profile(sector, False)
            hourly_real = hourly_expected * factor

            sector_real = sum(hourly_real[:18])
            sector_expected = sum(hourly_expected[:18])

            data[sector] = self._calculate_metrics(sector_real, sector_expected)
            data[sector]['has_anomaly'] = factor > 1.2

            total_real += sector_real
            total_expected += sector_expected

        data['total'] = self._calculate_metrics(total_real, total_expected)
        data['hourly'] = {'real': [], 'expected': []}
        data['timestamp'] = datetime.now().isoformat()
        data['sede'] = 'Tunja'

        return data

    def _generate_efficient_scenario(self) -> dict:
        """Genera escenario eficiente con ahorro en todos los sectores"""
        data = {}
        total_real = 0
        total_expected = 0

        for sector in self.SECTOR_PROFILES.keys():
            hourly_real, hourly_expected = self._generate_hourly_profile(sector, False)

            # Reducir consumo real 5-15%
            reduction = np.random.uniform(0.85, 0.95, 24)
            hourly_real = hourly_expected * reduction

            sector_real = sum(hourly_real[:18])
            sector_expected = sum(hourly_expected[:18])

            data[sector] = self._calculate_metrics(sector_real, sector_expected)
            data[sector]['has_anomaly'] = False

            total_real += sector_real
            total_expected += sector_expected

        data['total'] = self._calculate_metrics(total_real, total_expected)
        data['hourly'] = {'real': [], 'expected': []}
        data['timestamp'] = datetime.now().isoformat()
        data['sede'] = 'Tunja'

        return data

    def _generate_mixed_scenario(self) -> dict:
        """Genera escenario mixto: algunos bien, algunos mal"""
        data = {}
        total_real = 0
        total_expected = 0

        mixed_factors = {
            'laboratorios': 1.34,  # Mal - Anomalía
            'oficinas': 1.03,     # Normal
            'salones': 0.92,      # Bien - Ahorrando
            'comedores': 1.15,    # Revisar
            'auditorios': 1.00,   # Normal
        }

        for sector, factor in mixed_factors.items():
            hourly_real, hourly_expected = self._generate_hourly_profile(sector, False)

            # Aplicar factor con algo de ruido
            noise = np.random.uniform(0.98, 1.02, 24)
            hourly_real = hourly_expected * factor * noise

            sector_real = sum(hourly_real[:18])
            sector_expected = sum(hourly_expected[:18])

            data[sector] = self._calculate_metrics(sector_real, sector_expected)
            data[sector]['has_anomaly'] = factor > 1.25

            total_real += sector_real
            total_expected += sector_expected

        # Agregar datos horarios para el gráfico
        hourly_real_total = []
        hourly_expected_total = []

        for hour in range(24):
            hr = 0
            he = 0
            for sector in self.SECTOR_PROFILES.keys():
                profile = self.SECTOR_PROFILES[sector]
                if hour in profile['peak_hours']:
                    he += profile['base'] * profile['peak_factor']
                elif hour >= 22 or hour <= 5:
                    he += profile['base'] * profile['night_factor']
                else:
                    he += profile['base'] * 1.2

                hr = he * mixed_factors[sector] * np.random.uniform(0.95, 1.05)

            hourly_real_total.append(round(hr, 1))
            hourly_expected_total.append(round(he, 1))

        data['total'] = self._calculate_metrics(total_real, total_expected)
        data['hourly'] = {
            'real': hourly_real_total,
            'expected': hourly_expected_total
        }
        data['timestamp'] = datetime.now().isoformat()
        data['sede'] = 'Tunja'
        data['current_hour'] = datetime.now().hour

        return data


# ==============================================================================
# TEST
# ==============================================================================
if __name__ == "__main__":
    simulator = DataSimulator(seed=42)

    print("=== ESCENARIO MIXTO ===")
    data = simulator.generate_scenario('mixed')

    print(f"\nTotal Sede Tunja:")
    print(f"  Real: {data['total']['real']:,.0f} kWh")
    print(f"  Esperado: {data['total']['expected']:,.0f} kWh")
    print(f"  Delta: {data['total']['delta']:+,.0f} kWh ({data['total']['delta_percent']:+.1f}%)")
    print(f"  Costo: ${data['total']['cost']:,.0f} COP")

    print("\nPor Sector:")
    for sector in ['laboratorios', 'oficinas', 'salones', 'comedores', 'auditorios']:
        s = data[sector]
        print(f"  {sector.capitalize()}: {s['real']:,.0f} kWh ({s['delta_percent']:+.1f}%)")
