# data_simulator.py
"""
Simulador de datos energéticos para EcoCampus
"""
import random
import numpy as np

class DataSimulator:
    """Genera datos realistas de consumo energético"""
    
    def __init__(self):
        self.base_consumption = {
            'laboratorios': 950,
            'oficinas': 580,
            'salones': 420,
            'comedores': 350,
            'auditorios': 280
        }
    
    def generate_scenario(self, scenario_type='mixed'):
        """Genera un escenario de datos"""
        
        # Generar consumo por sector
        sectors_data = {}
        total_real = 0
        total_expected = 0
        
        for sector, base in self.base_consumption.items():
            # Variación realista (±20%)
            variation = random.uniform(0.8, 1.2)
            real = int(base * variation)
            
            # Esperado (con pequeña variación)
            expected = int(base * random.uniform(0.95, 1.05))
            
            sectors_data[sector] = {
                'real': real,
                'expected': expected,
                'delta': real - expected,
                'delta_percent': ((real - expected) / expected * 100) if expected > 0 else 0
            }
            
            total_real += real
            total_expected += expected
        
        # Calcular totales
        total_delta = total_real - total_expected
        total_delta_percent = (total_delta / total_expected * 100) if total_expected > 0 else 0
        
        # Costo (650 COP por kWh en Colombia)
        total_cost = total_delta * 650
        
        # CO2 (164g por kWh promedio en Colombia)
        total_co2 = total_real * 0.164
        
        # Datos horarios
        hourly_real = [int(random.uniform(150, 400)) for _ in range(24)]
        hourly_expected = [int(random.uniform(140, 380)) for _ in range(24)]
        
        return {
            'total': {
                'real': total_real,
                'expected': total_expected,
                'delta': total_delta,
                'delta_percent': total_delta_percent,
                'cost': total_cost,
                'co2': total_co2
            },
            'laboratorios': sectors_data['laboratorios'],
            'oficinas': sectors_data['oficinas'],
            'salones': sectors_data['salones'],
            'comedores': sectors_data['comedores'],
            'auditorios': sectors_data['auditorios'],
            'hourly': {
                'real': hourly_real,
                'expected': hourly_expected
            }
        }
