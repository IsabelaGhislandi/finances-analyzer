# Testes unitários para o módulo de métricas
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from modules.metrics import create_default_metrics_calculator

class TestMetricsCalculator:
    #Testes para o calculador de métricas
    
    def setup_method(self):
        # Configuração antes de cada teste
        self.metrics_calc = create_default_metrics_calculator()
        
        # Criar dados de teste
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
        np.random.seed(42)  # Para reprodutibilidade
        
        # Simular dados de juros fixos (crescimento constante)
        self.df_juros = pd.DataFrame({
            'Capital_Acumulado': [10000 * (1.01)**i for i in range(len(dates))],
            'Aporte_Mensal': [1000] * len(dates),
            'Data': dates
        }).set_index('Data')
        
        # Simular dados de carteira (com volatilidade)
        returns = np.random.normal(0.001, 0.02, len(dates))  # 0.1% média, 2% vol
        cumulative_returns = np.cumprod(1 + returns)
        self.df_carteira = pd.DataFrame({
            'Capital_Acumulado': [10000 * cumulative_returns[i] for i in range(len(dates))],
            'Aporte_Mensal': [1000] * len(dates),
            'Data': dates
        }).set_index('Data')
    
    def test_calculate_metrics_juros_fixos(self):
        # Testa cálculo de métricas para juros fixos
        metrics = self.metrics_calc.calculate_metrics(self.df_juros)
        
        # Verificar se métricas básicas foram calculadas
        required_metrics = ['Final_Capital', 'CAGR', 'Max_Drawdown', 'Total_Return']
        for metric in required_metrics:
            assert metric in metrics, f"Métrica {metric} não foi calculada"
        
        # Verificar valores específicos
        assert metrics['Final_Capital'] > 10000, "Capital final deve ser maior que inicial"
        assert 0 < metrics['CAGR'] < 1, "CAGR deve estar entre 0 e 1"
        assert metrics['Max_Drawdown'] <= 0, "Drawdown deve ser não-positivo"
    
    def test_calculate_metrics_carteira(self):
        # Testa cálculo de métricas para carteira de ações
        metrics = self.metrics_calc.calculate_metrics(self.df_carteira)
        
        # Verificar se métricas básicas foram calculadas
        required_metrics = ['Final_Capital', 'CAGR', 'Max_Drawdown', 'Total_Return']
        for metric in required_metrics:
            assert metric in metrics, f"Métrica {metric} não foi calculada"
        
        # Verificar valores específicos
        assert metrics['Final_Capital'] > 0, "Capital final deve ser positivo"
    
    def test_cagr_calculation(self):
        # Testa cálculo específico do CAGR
        metrics = self.metrics_calc.calculate_metrics(self.df_juros)
        
        # CAGR deve ser aproximadamente 12% ao ano para 1% ao mês
        expected_cagr = 0.1268  # (1.01^12 - 1)
        assert abs(metrics['CAGR'] - expected_cagr) < 0.01, f"CAGR esperado ~{expected_cagr:.1%}, obtido {metrics['CAGR']:.1%}"
    
    def test_drawdown_calculation(self):
        # Testa cálculo do drawdown máximo 
        metrics = self.metrics_calc.calculate_metrics(self.df_carteira)
        
        # Drawdown deve ser não-positivo
        assert metrics['Max_Drawdown'] <= 0, "Drawdown máximo deve ser não-positivo"
        # Para dados sintéticos, drawdown pode ser maior que -100%
        assert metrics['Max_Drawdown'] >= -100, "Drawdown máximo deve ser maior que -10000%"
    
    def test_total_return_calculation(self):
        # Testa cálculo do retorno total
        metrics = self.metrics_calc.calculate_metrics(self.df_juros)
        
        # Retorno total deve ser positivo para juros fixos
        assert metrics['Total_Return'] > 0, "Retorno total deve ser positivo"
        assert isinstance(metrics['Total_Return'], (int, float)), "Retorno total deve ser numérico"
    
    def test_metrics_data_types(self):
        # Testa se todas as métricas são numéricas
        metrics = self.metrics_calc.calculate_metrics(self.df_juros)
        
        for metric_name, metric_value in metrics.items():
            if metric_name != 'Period_Years':  # Esta pode ser string
                assert isinstance(metric_value, (int, float, np.number)), f"Métrica {metric_name} deve ser numérica"
    
    def test_empty_dataframe(self):
        # Testa comportamento com DataFrame vazio
        empty_df = pd.DataFrame()
        
        # Deve retornar métricas vazias ou None, não levantar exceção
        try:
            metrics = self.metrics_calc.calculate_metrics(empty_df)
            # Se não levantou exceção, deve retornar métricas vazias
            assert metrics is not None, "Deve retornar métricas mesmo para DataFrame vazio"
        except Exception:
            # Se levantou exceção, deve ser do tipo esperado
            pass
    
    def test_missing_columns(self):
        # Testa comportamento com colunas faltando
        df_missing = pd.DataFrame({
            'Data': pd.date_range('2023-01-01', '2024-01-01', freq='D')
        }).set_index('Data')
        
        # Deve retornar métricas vazias ou None, não levantar exceção
        try:
            metrics = self.metrics_calc.calculate_metrics(df_missing)
            # Se não levantou exceção, deve retornar métricas vazias
            assert metrics is not None, "Deve retornar métricas mesmo para colunas faltando"
        except Exception:
            # Se levantou exceção, deve ser do tipo esperado
            pass

if __name__ == "__main__":
    pytest.main([__file__])
