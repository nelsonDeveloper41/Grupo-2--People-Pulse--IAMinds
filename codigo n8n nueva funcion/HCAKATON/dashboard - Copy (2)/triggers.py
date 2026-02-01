# triggers.py
"""
Motor de triggers para alertas automáticas
"""

class TriggerEngine:
    """Genera alertas basadas en umbrales"""
    
    def evaluate(self, data):
        """Evalúa datos y genera alertas"""
        alerts = []
        sectors = ['laboratorios', 'oficinas', 'salones', 'comedores', 'auditorios']
        
        for sector in sectors:
            s_data = data[sector]
            delta_pct = s_data['delta_percent']
            
            # Trigger 1: Sobreconsumo crítico (>30%)
            if delta_pct > 30:
                alerts.append({
                    'sector': sector.title(),
                    'title': 'Sobreconsumo Crítico',
                    'description': f'Consumo {delta_pct:.1f}% por encima de lo esperado',
                    'severity': 'high',
                    'cost': abs(s_data['delta'] * 650)
                })
            
            # Trigger 2: Sobreconsumo moderado (15-30%)
            elif delta_pct > 15:
                alerts.append({
                    'sector': sector.title(),
                    'title': 'Sobreconsumo Moderado',
                    'description': f'Revisar equipamiento activo ({delta_pct:.1f}%)',
                    'severity': 'medium',
                    'cost': abs(s_data['delta'] * 650)
                })
            
            # Trigger 3: Anomalía en patrón (consumo muy bajo)
            elif delta_pct < -20:
                alerts.append({
                    'sector': sector.title(),
                    'title': 'Consumo Anormalmente Bajo',
                    'description': 'Verificar sensores o falta de ocupancia',
                    'severity': 'low',
                    'cost': abs(s_data['delta'] * 650)
                })
        
        # Trigger 4: Sobreconsumo total
        if data['total']['delta_percent'] > 20:
            alerts.append({
                'sector': 'GLOBAL',
                'title': 'Sobreconsumo General',
                'description': f'Consumo total {data["total"]["delta_percent"]:.1f}% arriba de límite',
                'severity': 'high',
                'cost': abs(data['total']['delta'] * 650)
            })
        
        # Ordenar por severidad
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        alerts.sort(key=lambda x: severity_order.get(x['severity'], 999))
        
        return alerts
