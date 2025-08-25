# Testes unitários para o módulo de previsões
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TestForecastBasics:
    # Testes básicos para funcionalidades de previsão
    
    def setup_method(self):
        """Configuração antes de cada teste"""
        # Criar dados de teste sintéticos
        np.random.seed(42)  # Para reprodutibilidade
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
        
        # Simular série temporal com tendência e sazonalidade
        trend = np.linspace(100, 120, len(dates))  # Tendência crescente
        seasonal = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)  # Sazonalidade mensal
        noise = np.random.normal(0, 2, len(dates))  # Ruído
        
        self.test_series = pd.Series(
            trend + seasonal + noise,
            index=dates,
            name='Close'
        )
    
    def test_data_creation(self):
    # Testa se os dados de teste foram criados corretamente
        assert len(self.test_series) > 0, "Série temporal deve ter dados"
        assert isinstance(self.test_series.index, pd.DatetimeIndex), "Índice deve ser temporal"
        assert not self.test_series.isna().any(), "Não deve haver valores NaN"
        assert self.test_series.min() > 0, "Preços devem ser positivos"
    
    def test_trend_analysis(self):
        # Calcular tendência linear
        x = np.arange(len(self.test_series))
        y = self.test_series.values
        
        # Regressão linear simples
        slope = np.polyfit(x, y, 1)[0]
        
        # Tendência deve ser crescente (slope > 0)
        assert slope > 0, "Dados devem ter tendência crescente"
        assert slope < 1, "Tendência deve ser razoável"
    
    def test_volatility_calculation(self):
        # Testa cálculo de volatilidade
        # Calcular retornos logarítmicos
        returns = np.log(self.test_series / self.test_series.shift(1)).dropna()
        
        # Volatilidade anualizada
        volatility = returns.std() * np.sqrt(252)  # 252 dias úteis
        
        assert volatility > 0, "Volatilidade deve ser positiva"
        assert volatility < 1, "Volatilidade deve ser menor que 100%"
    
    def test_seasonality_detection(self):
        # Testa detecção de sazonalidade
        # Calcular autocorrelação para detectar sazonalidade
        autocorr = self.test_series.autocorr(lag=30)  # Lag de 30 dias
        
        # Autocorrelação deve ser significativa para dados sazonais
        assert abs(autocorr) > 0.1, "Dados devem mostrar alguma sazonalidade"
    
    def test_forecast_preparation(self):
        # Testa preparação de dados para previsão
        # Dividir dados em treino e teste
        split_point = int(len(self.test_series) * 0.8)
        train_data = self.test_series[:split_point]
        test_data = self.test_series[split_point:]
        
        assert len(train_data) > 0, "Dados de treino não devem estar vazios"
        assert len(test_data) > 0, "Dados de teste não devem estar vazios"
        assert len(train_data) + len(test_data) == len(self.test_series), "Divisão deve preservar todos os dados"
    
    def test_rolling_statistics(self):
        # Testa estatísticas móveis
        # Média móvel de 30 dias
        rolling_mean = self.test_series.rolling(window=30).mean()
        
        # Verificar se a média móvel foi calculada
        assert not rolling_mean.isna().all(), "Média móvel deve ter valores válidos"
        assert len(rolling_mean) == len(self.test_series), "Média móvel deve ter mesmo tamanho"
        
        # Média móvel deve ser mais suave que os dados originais
        original_std = self.test_series.std()
        rolling_std = rolling_mean.std()
        assert rolling_std < original_std, "Média móvel deve ser mais suave"
    
    def test_data_quality(self):
        # Testa qualidade dos dados
        # Verificar se não há outliers extremos
        q1 = self.test_series.quantile(0.25)
        q3 = self.test_series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = self.test_series[(self.test_series < lower_bound) | (self.test_series > upper_bound)]
        
        # Não deve haver muitos outliers (menos de 5%)
        outlier_percentage = len(outliers) / len(self.test_series)
        assert outlier_percentage < 0.05, "Não deve haver muitos outliers"
    
    def test_time_series_properties(self):
        # Testa propriedades básicas da série temporal
        # Verificar se é estacionária (teste básico)
        first_half = self.test_series[:len(self.test_series)//2]
        second_half = self.test_series[len(self.test_series)//2:]
        
        # Variância das duas metades deve ser similar
        var_ratio = first_half.var() / second_half.var()
        assert 0.5 < var_ratio < 2.0, "Variância deve ser relativamente estável"
    
    def test_forecast_horizon_validation(self):
        # Testa validação de horizonte de previsão
        # Horizonte deve ser positivo e razoável
        max_horizon = len(self.test_series) // 4  # Máximo 25% dos dados
        
        # Testar diferentes horizontes
        for horizon in [1, 7, 30, max_horizon]:
            assert horizon > 0, "Horizonte deve ser positivo"
            assert horizon <= max_horizon, "Horizonte deve ser razoável"
    
    def test_data_transformation(self):
        # Testa transformações de dados
        # Logaritmo dos preços
        log_prices = np.log(self.test_series)
        
        # Retornos
        returns = self.test_series.pct_change().dropna()
        
        # Verificar transformações
        assert not log_prices.isna().any(), "Log não deve gerar NaN"
        assert not returns.isna().all(), "Retornos não devem ser todos NaN"
        assert len(returns) == len(self.test_series) - 1, "Retornos devem ter tamanho correto"

if __name__ == "__main__":
    pytest.main([__file__])
