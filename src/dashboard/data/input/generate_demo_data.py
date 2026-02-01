"""
Generador de Datos Sinteticos para Demo - EcoCampus UPTC
=========================================================
VERSION 2: Con anomalias MAS EVIDENTES para comparar con modelo XGBoost

Los datos "Real" simulan sensores con:
- Picos inesperados (anomalia_pico)
- Consumo nocturno alto (anomalia_nocturna / vampiro)
- Variacion realista por hora/dia
"""

import csv
import math
import random
from datetime import datetime, timedelta

SEDES = ["UPTC_TUN", "UPTC_SOG", "UPTC_DUI", "UPTC_CHI"]

SEDE_PROFILES = {
    "UPTC_TUN": {
        "nombre": "Tunja",
        "energia_base": 1.5,
        "energia_max": 8.0,
        "agua_media": 1500,
        "temp_base": 13.0,
        "distribucion": {"comedor": 0.12, "salones": 0.25, "laboratorios": 0.30, "auditorios": 0.08, "oficinas": 0.25}
    },
    "UPTC_SOG": {
        "nombre": "Sogamoso",
        "energia_base": 4.0,
        "energia_max": 18.0,
        "agua_media": 130,
        "temp_base": 14.0,
        "distribucion": {"comedor": 0.10, "salones": 0.20, "laboratorios": 0.35, "auditorios": 0.10, "oficinas": 0.25}
    },
    "UPTC_DUI": {
        "nombre": "Duitama",
        "energia_base": 3.5,
        "energia_max": 16.0,
        "agua_media": 160,
        "temp_base": 15.0,
        "distribucion": {"comedor": 0.10, "salones": 0.25, "laboratorios": 0.25, "auditorios": 0.10, "oficinas": 0.30}
    },
    "UPTC_CHI": {
        "nombre": "Chiquinquira",
        "energia_base": 1.0,
        "energia_max": 6.0,
        "agua_media": 50,
        "temp_base": 14.0,
        "distribucion": {"comedor": 0.08, "salones": 0.40, "laboratorios": 0.20, "auditorios": 0.10, "oficinas": 0.22}
    }
}

CO2_FACTOR = 0.198

def hora_ciclica(hora):
    rad = 2 * math.pi * hora / 24
    return math.sin(rad), math.cos(rad)

def temperatura_hora(hora, temp_base):
    rad = 2 * math.pi * (hora - 4) / 24
    variacion = 6 * math.sin(rad)
    return round(temp_base + variacion + random.uniform(-1, 1), 1)

def ocupacion_hora(hora, escenario):
    # Patron base por hora
    if 0 <= hora <= 5:
        base = 3
    elif 6 <= hora <= 7:
        base = 20
    elif 8 <= hora <= 11:
        base = 75
    elif 12 <= hora <= 13:
        base = 55
    elif 14 <= hora <= 17:
        base = 80
    elif 18 <= hora <= 20:
        base = 45
    else:
        base = 12

    # Modificadores por escenario
    multiplicadores = {
        "dia_normal": 1.0,
        "fin_semana": 0.12,
        "semana_parciales": 1.35,
        "semana_finales": 1.45,
        "vacaciones": 0.05,
        "festivo": 0.08,
        "anomalia_pico": 1.1,
        "anomalia_nocturna": 0.25 if hora >= 8 else 0.9
    }

    mult = multiplicadores.get(escenario, 1.0)
    ocupacion = base * mult + random.uniform(-8, 8)
    return max(0, min(100, round(ocupacion, 1)))

def generar_energia(sede, hora, ocupacion, escenario):
    """Genera consumo energetico CON ANOMALIAS EVIDENTES."""
    perfil = SEDE_PROFILES[sede]
    factor_ocupacion = ocupacion / 100

    energia_base = perfil["energia_base"]
    energia_max = perfil["energia_max"]

    # Energia base + proporcional
    energia_total = energia_base + (energia_max - energia_base) * factor_ocupacion

    # Ruido normal
    energia_total *= random.uniform(0.92, 1.08)

    # ===== ANOMALIAS EVIDENTES =====

    # ANOMALIA PICO: Picos muy altos entre 10-14h
    if escenario == "anomalia_pico":
        if 10 <= hora <= 14:
            # Multiplicar por 2.5-3.5x = PICO MUY EVIDENTE
            energia_total *= random.uniform(2.5, 3.5)
        elif 8 <= hora <= 9 or 15 <= hora <= 16:
            energia_total *= random.uniform(1.5, 2.0)

    # ANOMALIA NOCTURNA (Vampiro): Consumo alto de noche
    elif escenario == "anomalia_nocturna":
        if hora >= 22 or hora <= 5:
            # Consumo nocturno = 70-90% del maximo (deberia ser ~10%)
            energia_total = energia_max * random.uniform(0.7, 0.9)
        elif 6 <= hora <= 8:
            energia_total *= random.uniform(0.6, 0.8)  # Bajo de dia

    # FIN DE SEMANA: Muy bajo
    elif escenario == "fin_semana":
        energia_total = energia_base * random.uniform(0.8, 1.2)

    # VACACIONES: Minimo
    elif escenario == "vacaciones":
        energia_total = energia_base * random.uniform(0.5, 0.8)

    # FESTIVO: Similar a fin de semana
    elif escenario == "festivo":
        energia_total = energia_base * random.uniform(0.7, 1.0)

    # SEMANA PARCIALES: Alto
    elif escenario == "semana_parciales":
        energia_total *= random.uniform(1.1, 1.3)

    # SEMANA FINALES: Muy alto
    elif escenario == "semana_finales":
        energia_total *= random.uniform(1.2, 1.4)

    energia_total = max(0, round(energia_total, 3))

    # Distribuir por sectores
    dist = perfil["distribucion"]
    sectores = {}
    total_asignado = 0

    for sector in ["comedor", "salones", "laboratorios", "auditorios"]:
        valor = energia_total * dist[sector] * random.uniform(0.85, 1.15)
        valor = max(0, round(valor, 4))
        sectores[sector] = valor
        total_asignado += valor

    sectores["oficinas"] = max(0, round(energia_total - total_asignado, 4))

    return energia_total, sectores

def generar_agua(sede, ocupacion):
    perfil = SEDE_PROFILES[sede]
    factor = ocupacion / 100
    agua = perfil["agua_media"] * (0.2 + 0.8 * factor)
    agua *= random.uniform(0.8, 1.2)
    return max(0, round(agua, 2))

def generar_potencia(energia_total):
    return round(energia_total * random.uniform(1.2, 1.5), 3)

def generar_fila(escenario, sede, timestamp, periodo_academico, es_festivo_flag=False):
    hora = timestamp.hour
    dia_semana = timestamp.weekday()

    hora_sin, hora_cos = hora_ciclica(hora)

    es_fin_semana = 1 if dia_semana >= 5 else 0
    es_festivo = 1 if es_festivo_flag else 0
    semana_parciales = 1 if escenario == "semana_parciales" else 0
    semana_finales = 1 if escenario == "semana_finales" else 0

    periodo_s1 = 1 if periodo_academico == "semestre1" else 0
    periodo_s2 = 1 if periodo_academico == "semestre2" else 0
    periodo_vac = 1 if periodo_academico == "vacaciones" else 0

    perfil = SEDE_PROFILES[sede]
    temperatura = temperatura_hora(hora, perfil["temp_base"])
    ocupacion = ocupacion_hora(hora, escenario)

    energia_total, sectores = generar_energia(sede, hora, ocupacion, escenario)
    agua = generar_agua(sede, ocupacion)
    potencia = generar_potencia(energia_total)
    co2 = round(energia_total * CO2_FACTOR, 4)
    co2_input = round(co2 * random.uniform(0.9, 1.1), 4)

    return {
        "escenario": escenario,
        "sede": sede,
        "sede_nombre": perfil["nombre"],
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "hora": hora,
        "hora_sin": round(hora_sin, 6),
        "hora_cos": round(hora_cos, 6),
        "dia_semana": dia_semana,
        "es_fin_semana": es_fin_semana,
        "es_festivo": es_festivo,
        "semana_parciales": semana_parciales,
        "semana_finales": semana_finales,
        "periodo_semestre1": periodo_s1,
        "periodo_semestre2": periodo_s2,
        "periodo_vacaciones": periodo_vac,
        "temperatura_exterior": temperatura,
        "ocupacion_pct": ocupacion,
        "co2_input": co2_input,
        "energia_total_kwh": energia_total,
        "energia_comedor_kwh": sectores["comedor"],
        "energia_salones_kwh": sectores["salones"],
        "energia_laboratorios_kwh": sectores["laboratorios"],
        "energia_auditorios_kwh": sectores["auditorios"],
        "energia_oficinas_kwh": sectores["oficinas"],
        "potencia_total_kw": potencia,
        "agua_litros": agua,
        "co2_kg": co2
    }

def generar_escenario(escenario, fecha_base, periodo_academico, horas=24, es_festivo=False):
    filas = []
    for hora in range(horas):
        timestamp = fecha_base + timedelta(hours=hora)
        for sede in SEDES:
            fila = generar_fila(escenario, sede, timestamp, periodo_academico, es_festivo)
            filas.append(fila)
    return filas

def main():
    random.seed(42)

    todas_las_filas = []
    fecha_base = datetime(2025, 3, 10, 0, 0, 0)

    # Escenarios con anomalias mas evidentes
    escenarios = [
        ("dia_normal", fecha_base, "semestre1", False),
        ("fin_semana", fecha_base + timedelta(days=5), "semestre1", False),
        ("semana_parciales", fecha_base + timedelta(days=30), "semestre1", False),
        ("semana_finales", fecha_base + timedelta(days=90), "semestre1", False),
        ("vacaciones", datetime(2025, 7, 1), "vacaciones", False),
        ("festivo", datetime(2025, 5, 1), "semestre1", True),
        ("anomalia_pico", fecha_base + timedelta(days=15), "semestre1", False),
        ("anomalia_nocturna", fecha_base + timedelta(days=20), "semestre1", False),
    ]

    for escenario, fecha, periodo, festivo in escenarios:
        print(f"Generando: {escenario}...")
        todas_las_filas.extend(generar_escenario(escenario, fecha, periodo, es_festivo=festivo))

    # Guardar CSV
    campos = list(todas_las_filas[0].keys())
    output_path = "demo_sensores.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(todas_las_filas)

    print(f"\n[OK] Generado: {output_path}")
    print(f"    - Total filas: {len(todas_las_filas)}")
    print(f"    - Escenarios: {len(escenarios)}")

    # Mostrar resumen de anomalias
    print("\nResumen de energia por escenario (UPTC_SOG, hora 12):")
    for fila in todas_las_filas:
        if fila['sede'] == 'UPTC_SOG' and fila['hora'] == 12:
            print(f"  {fila['escenario']:20} | {fila['energia_total_kwh']:8.2f} kWh")

if __name__ == "__main__":
    main()
