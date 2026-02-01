"""
Generador de Datos por Periodos - EcoCampus UPTC
=================================================
VERSION 3: Curvas Real y Esperado SIMILARES con POCOS picos aislados
"""

import csv
import math
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Agregar path para importar models
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models import get_predictor, SEDES, SEDE_NOMBRES

# ============================================================================
# CONFIGURACION
# ============================================================================

SEDE_PROFILES = {
    "UPTC_TUN": {
        "nombre": "Tunja",
        "energia_base": 1.5,
        "energia_max": 8.0,
        "agua_media": 1500,
        "temp_base": 13.0,
    },
    "UPTC_SOG": {
        "nombre": "Sogamoso",
        "energia_base": 4.0,
        "energia_max": 18.0,
        "agua_media": 130,
        "temp_base": 14.0,
    },
    "UPTC_DUI": {
        "nombre": "Duitama",
        "energia_base": 3.5,
        "energia_max": 16.0,
        "agua_media": 160,
        "temp_base": 15.0,
    },
    "UPTC_CHI": {
        "nombre": "Chiquinquira",
        "energia_base": 1.0,
        "energia_max": 6.0,
        "agua_media": 50,
        "temp_base": 14.0,
    }
}

CO2_FACTOR = 0.198
COSTO_KWH = 650

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def ocupacion_hora(hora, dia_semana):
    """Retorna ocupacion esperada por hora."""
    if 0 <= hora <= 5:
        base = 5
    elif 6 <= hora <= 7:
        base = 25
    elif 8 <= hora <= 11:
        base = 75
    elif 12 <= hora <= 13:
        base = 55
    elif 14 <= hora <= 17:
        base = 80
    elif 18 <= hora <= 20:
        base = 45
    else:
        base = 15

    if dia_semana >= 5:
        base *= 0.15

    return min(100, max(0, base + random.uniform(-5, 5)))

def temperatura_hora(hora, temp_base):
    """Temperatura con variacion diaria."""
    rad = 2 * math.pi * (hora - 4) / 24
    variacion = 6 * math.sin(rad)
    return temp_base + variacion + random.uniform(-1, 1)

# ============================================================================
# GENERACION DE CURVAS
# ============================================================================

def generar_curva_dia_con_picos(sede, fecha, predictor, horas_pico=None):
    """
    Genera curva de 24 horas.
    - La mayoria de horas: Real ≈ Esperado (variacion +/- 5%)
    - horas_pico: lista de horas donde hay anomalia (pico 2-3x)
    """
    if horas_pico is None:
        horas_pico = []

    dia_semana = fecha.weekday()
    perfil = SEDE_PROFILES[sede]

    curva_real = []
    curva_esperada = []

    for hora in range(24):
        ocupacion = ocupacion_hora(hora, dia_semana)
        temperatura = temperatura_hora(hora, perfil["temp_base"])

        # ESPERADO: Prediccion XGBoost
        energia_esperada = predictor.predict(
            sede=sede,
            target='energia_total_kwh',
            hora=hora,
            dia_semana=dia_semana,
            mes=fecha.month,
            es_fin_semana=(dia_semana >= 5),
            temperatura=temperatura,
            ocupacion_pct=ocupacion,
            periodo='semestre_1' if fecha.month in [2,3,4,5,6] else 'semestre_2'
        )
        curva_esperada.append(round(energia_esperada, 3))

        # REAL: Similar al esperado CON PEQUEÑA VARIACION
        if hora in horas_pico:
            # PICO: 2.5x a 3.5x el valor esperado
            energia_real = energia_esperada * random.uniform(2.5, 3.5)
        else:
            # NORMAL: Muy similar al esperado (+/- 8%)
            energia_real = energia_esperada * random.uniform(0.92, 1.08)

        curva_real.append(round(energia_real, 3))

    return curva_real, curva_esperada

def generar_curva_semana(sede, fecha_inicio, predictor, dias_pico=None):
    """
    Genera datos para 7 dias.
    - dias_pico: lista de indices de dias con anomalia
    """
    if dias_pico is None:
        dias_pico = []

    totales_real = []
    totales_esperado = []

    for i in range(7):
        fecha = fecha_inicio + timedelta(days=i)

        # Solo algunas horas de pico si es dia de anomalia
        if i in dias_pico:
            horas_pico = [10, 11, 12, 13, 14]  # Horas pico del dia
        else:
            horas_pico = []

        curva_real, curva_esperada = generar_curva_dia_con_picos(
            sede, fecha, predictor, horas_pico
        )

        totales_real.append(sum(curva_real))
        totales_esperado.append(sum(curva_esperada))

    return totales_real, totales_esperado

def generar_curva_mes(sede, fecha_inicio, predictor, dias_pico=None):
    """
    Genera datos para 30 dias.
    - dias_pico: lista de indices de dias con anomalia
    """
    if dias_pico is None:
        dias_pico = []

    totales_real = []
    totales_esperado = []

    for i in range(30):
        fecha = fecha_inicio + timedelta(days=i)

        if i in dias_pico:
            horas_pico = [10, 11, 12, 13, 14]
        else:
            horas_pico = []

        curva_real, curva_esperada = generar_curva_dia_con_picos(
            sede, fecha, predictor, horas_pico
        )

        totales_real.append(sum(curva_real))
        totales_esperado.append(sum(curva_esperada))

    return totales_real, totales_esperado

def generar_curva_semestre(sede, fecha_inicio, predictor, semanas_pico=None):
    """
    Genera datos para ~26 semanas.
    - semanas_pico: lista de indices de semanas con anomalia
    """
    if semanas_pico is None:
        semanas_pico = []

    totales_real = []
    totales_esperado = []

    for semana in range(26):
        fecha_semana = fecha_inicio + timedelta(weeks=semana)

        # Si es semana de pico, 1 dia tiene anomalia
        if semana in semanas_pico:
            dias_pico = [2]  # Miercoles
        else:
            dias_pico = []

        # Generar 7 dias
        total_real_semana = 0
        total_esperado_semana = 0

        for i in range(7):
            fecha = fecha_semana + timedelta(days=i)

            if i in dias_pico:
                horas_pico = [10, 11, 12, 13, 14]
            else:
                horas_pico = []

            curva_real, curva_esperada = generar_curva_dia_con_picos(
                sede, fecha, predictor, horas_pico
            )

            total_real_semana += sum(curva_real)
            total_esperado_semana += sum(curva_esperada)

        totales_real.append(total_real_semana)
        totales_esperado.append(total_esperado_semana)

    return totales_real, totales_esperado

def generar_curva_anio(sede, fecha_inicio, predictor, meses_pico=None):
    """
    Genera datos para 12 meses.
    - meses_pico: lista de indices de meses con anomalia
    """
    if meses_pico is None:
        meses_pico = []

    totales_real = []
    totales_esperado = []

    for mes in range(12):
        fecha_mes = fecha_inicio + timedelta(days=mes * 30)

        # Si es mes de pico, 2 dias tienen anomalia
        if mes in meses_pico:
            dias_pico = [5, 15]
        else:
            dias_pico = []

        # Generar 30 dias
        total_real_mes = 0
        total_esperado_mes = 0

        for i in range(30):
            fecha = fecha_mes + timedelta(days=i)

            if i in dias_pico:
                horas_pico = [10, 11, 12, 13, 14]
            else:
                horas_pico = []

            curva_real, curva_esperada = generar_curva_dia_con_picos(
                sede, fecha, predictor, horas_pico
            )

            total_real_mes += sum(curva_real)
            total_esperado_mes += sum(curva_esperada)

        totales_real.append(total_real_mes)
        totales_esperado.append(total_esperado_mes)

    return totales_real, totales_esperado

# ============================================================================
# MAIN
# ============================================================================

def main():
    random.seed(42)

    print("Cargando modelos XGBoost...")
    predictor = get_predictor()

    if not predictor.model_loaded:
        print("[ERROR] No se pudieron cargar los modelos XGBoost")
        return

    print(f"[OK] Modelos cargados")

    fecha_base = datetime(2025, 2, 1)
    todas_las_filas = []

    # Configuracion de anomalias (POCAS y especificas)
    CONFIG_ANOMALIAS = {
        "dia": {"horas_pico": [10, 11, 14]},  # Solo 3 horas con pico
        "semana": {"dias_pico": [2]},          # Solo miercoles
        "mes": {"dias_pico": [5, 18]},         # Solo 2 dias en el mes
        "semestre": {"semanas_pico": [3, 12, 20]},  # Solo 3 semanas
        "anio": {"meses_pico": [2, 6, 9]},     # Solo 3 meses
    }

    for sede in SEDES:
        print(f"\nGenerando datos para {SEDE_NOMBRES.get(sede, sede)}...")

        # ===== DIA =====
        horas_pico = CONFIG_ANOMALIAS["dia"]["horas_pico"]
        curva_real, curva_esperada = generar_curva_dia_con_picos(
            sede, fecha_base, predictor, horas_pico
        )
        energia_real = sum(curva_real)
        energia_esperada = sum(curva_esperada)
        delta_pct = (energia_real - energia_esperada) / energia_esperada * 100

        todas_las_filas.append({
            "periodo": "dia",
            "sede": sede,
            "sede_nombre": SEDE_NOMBRES.get(sede, sede),
            "fecha_inicio": fecha_base.strftime("%Y-%m-%d"),
            "num_dias": 1,
            "energia_real": round(energia_real, 2),
            "energia_esperada": round(energia_esperada, 2),
            "delta": round(energia_real - energia_esperada, 2),
            "delta_pct": round(delta_pct, 2),
            "estado": "exceso" if delta_pct > 10 else "normal",
            "costo_delta": round((energia_real - energia_esperada) * COSTO_KWH, 0),
            "agua_total": round(SEDE_PROFILES[sede]["agua_media"] * random.uniform(0.9, 1.1), 2),
            "co2_total": round(energia_real * CO2_FACTOR, 2),
            "curva_valores": ",".join([str(round(v, 2)) for v in curva_real]),
            "curva_esperada": ",".join([str(round(v, 2)) for v in curva_esperada]),
        })
        print(f"  dia        | Delta: {delta_pct:+.1f}% | Picos en horas: {horas_pico}")

        # ===== SEMANA =====
        dias_pico = CONFIG_ANOMALIAS["semana"]["dias_pico"]
        curva_real, curva_esperada = generar_curva_semana(
            sede, fecha_base, predictor, dias_pico
        )
        energia_real = sum(curva_real)
        energia_esperada = sum(curva_esperada)
        delta_pct = (energia_real - energia_esperada) / energia_esperada * 100

        todas_las_filas.append({
            "periodo": "semana",
            "sede": sede,
            "sede_nombre": SEDE_NOMBRES.get(sede, sede),
            "fecha_inicio": fecha_base.strftime("%Y-%m-%d"),
            "num_dias": 7,
            "energia_real": round(energia_real, 2),
            "energia_esperada": round(energia_esperada, 2),
            "delta": round(energia_real - energia_esperada, 2),
            "delta_pct": round(delta_pct, 2),
            "estado": "exceso" if delta_pct > 10 else "normal",
            "costo_delta": round((energia_real - energia_esperada) * COSTO_KWH, 0),
            "agua_total": round(SEDE_PROFILES[sede]["agua_media"] * 7 * random.uniform(0.9, 1.1), 2),
            "co2_total": round(energia_real * CO2_FACTOR, 2),
            "curva_valores": ",".join([str(round(v, 2)) for v in curva_real]),
            "curva_esperada": ",".join([str(round(v, 2)) for v in curva_esperada]),
        })
        print(f"  semana     | Delta: {delta_pct:+.1f}% | Pico en dia: {dias_pico}")

        # ===== MES =====
        dias_pico = CONFIG_ANOMALIAS["mes"]["dias_pico"]
        curva_real, curva_esperada = generar_curva_mes(
            sede, fecha_base, predictor, dias_pico
        )
        energia_real = sum(curva_real)
        energia_esperada = sum(curva_esperada)
        delta_pct = (energia_real - energia_esperada) / energia_esperada * 100

        todas_las_filas.append({
            "periodo": "mes",
            "sede": sede,
            "sede_nombre": SEDE_NOMBRES.get(sede, sede),
            "fecha_inicio": fecha_base.strftime("%Y-%m-%d"),
            "num_dias": 30,
            "energia_real": round(energia_real, 2),
            "energia_esperada": round(energia_esperada, 2),
            "delta": round(energia_real - energia_esperada, 2),
            "delta_pct": round(delta_pct, 2),
            "estado": "exceso" if delta_pct > 10 else "normal",
            "costo_delta": round((energia_real - energia_esperada) * COSTO_KWH, 0),
            "agua_total": round(SEDE_PROFILES[sede]["agua_media"] * 30 * random.uniform(0.9, 1.1), 2),
            "co2_total": round(energia_real * CO2_FACTOR, 2),
            "curva_valores": ",".join([str(round(v, 2)) for v in curva_real]),
            "curva_esperada": ",".join([str(round(v, 2)) for v in curva_esperada]),
        })
        print(f"  mes        | Delta: {delta_pct:+.1f}% | Picos en dias: {dias_pico}")

        # ===== SEMESTRE =====
        semanas_pico = CONFIG_ANOMALIAS["semestre"]["semanas_pico"]
        curva_real, curva_esperada = generar_curva_semestre(
            sede, fecha_base, predictor, semanas_pico
        )
        energia_real = sum(curva_real)
        energia_esperada = sum(curva_esperada)
        delta_pct = (energia_real - energia_esperada) / energia_esperada * 100

        todas_las_filas.append({
            "periodo": "semestre",
            "sede": sede,
            "sede_nombre": SEDE_NOMBRES.get(sede, sede),
            "fecha_inicio": fecha_base.strftime("%Y-%m-%d"),
            "num_dias": 180,
            "energia_real": round(energia_real, 2),
            "energia_esperada": round(energia_esperada, 2),
            "delta": round(energia_real - energia_esperada, 2),
            "delta_pct": round(delta_pct, 2),
            "estado": "exceso" if delta_pct > 10 else "normal",
            "costo_delta": round((energia_real - energia_esperada) * COSTO_KWH, 0),
            "agua_total": round(SEDE_PROFILES[sede]["agua_media"] * 180 * random.uniform(0.9, 1.1), 2),
            "co2_total": round(energia_real * CO2_FACTOR, 2),
            "curva_valores": ",".join([str(round(v, 2)) for v in curva_real]),
            "curva_esperada": ",".join([str(round(v, 2)) for v in curva_esperada]),
        })
        print(f"  semestre   | Delta: {delta_pct:+.1f}% | Picos en semanas: {semanas_pico}")

        # ===== AÑO =====
        meses_pico = CONFIG_ANOMALIAS["anio"]["meses_pico"]
        curva_real, curva_esperada = generar_curva_anio(
            sede, fecha_base, predictor, meses_pico
        )
        energia_real = sum(curva_real)
        energia_esperada = sum(curva_esperada)
        delta_pct = (energia_real - energia_esperada) / energia_esperada * 100

        todas_las_filas.append({
            "periodo": "anio",
            "sede": sede,
            "sede_nombre": SEDE_NOMBRES.get(sede, sede),
            "fecha_inicio": fecha_base.strftime("%Y-%m-%d"),
            "num_dias": 365,
            "energia_real": round(energia_real, 2),
            "energia_esperada": round(energia_esperada, 2),
            "delta": round(energia_real - energia_esperada, 2),
            "delta_pct": round(delta_pct, 2),
            "estado": "exceso" if delta_pct > 10 else "normal",
            "costo_delta": round((energia_real - energia_esperada) * COSTO_KWH, 0),
            "agua_total": round(SEDE_PROFILES[sede]["agua_media"] * 365 * random.uniform(0.9, 1.1), 2),
            "co2_total": round(energia_real * CO2_FACTOR, 2),
            "curva_valores": ",".join([str(round(v, 2)) for v in curva_real]),
            "curva_esperada": ",".join([str(round(v, 2)) for v in curva_esperada]),
        })
        print(f"  anio       | Delta: {delta_pct:+.1f}% | Picos en meses: {meses_pico}")

    # Guardar CSV
    campos = list(todas_las_filas[0].keys())
    output_path = Path(__file__).parent / "demo_periodos.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(todas_las_filas)

    print(f"\n{'='*60}")
    print(f"[OK] Generado: {output_path}")
    print(f"    - Total filas: {len(todas_las_filas)}")
    print(f"\nAhora las curvas son SIMILARES con POCOS picos aislados!")

if __name__ == "__main__":
    main()
