
# Testes unitários para os simuladores de investimento
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from modules.interests import InvestmentSimulator

class TestSimulators:
    #Testes para os simuladores de investimento
    
    def setup_method(self):
        #Configuração antes de cada teste"""
        # Parâmetros de teste
        self.initial_capital = 10000
        self.monthly_contribution = 1000
        self.start_date = '2023-01-01'
        self.end_date = '2024-01-01'
    
    def test_compound_interest_simulator(self):
        #Testa simulador de juros compostos"""
        simulator = InvestmentSimulator(
            initial_capital=self.initial_capital,
            monthly_contribution=self.monthly_contribution,
            monthly_interest_rate=1.0  # 1% ao mês
        )
        
        # Verificar se o simulador foi criado
        assert simulator is not None, "Simulador de juros compostos não foi criado"
        
        # Executar simulação
        result = simulator.simulate_fixed_interest_scenario(
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Verificar resultado
        assert isinstance(result, pd.DataFrame), "Resultado deve ser um DataFrame"
        assert not result.empty, "Resultado não deve estar vazio"
        assert len(result) > 0, "Resultado deve ter pelo menos uma linha"
        
        # Verificar colunas
        expected_columns = ['Accumulated_Capital', 'Monthly_Contribution']
        for col in expected_columns:
            assert col in result.columns, f"Coluna {col} não encontrada"
        
        # Verificar se o capital cresce
        initial_cap = result.iloc[0]['Accumulated_Capital']
        final_cap = result.iloc[-1]['Accumulated_Capital']
        assert final_cap > initial_cap, "Capital final deve ser maior que inicial"
    
    def test_stock_portfolio_simulation(self):
        #Testa simulação de carteira de ações
        # Criar dados sintéticos para simulação
        dates = pd.date_range(self.start_date, self.end_date, freq='D')
        np.random.seed(42)
        
        # Simular retornos de ações
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = [100]  # Preço inicial
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        stock_data = {
            'PETR4.SA': pd.DataFrame({
                'Close': prices,
                'Data': dates
            }).set_index('Data')
        }
        
        # Criar simulador
        simulator = InvestmentSimulator(
            initial_capital=self.initial_capital,
            monthly_contribution=self.monthly_contribution,
            monthly_interest_rate=0.01
        )
        
        # Simular carteira - passar monthly_return_rate explicitamente
        result = simulator.simulate_stock_portfolio_scenario(
            stock_data=stock_data,
            weights=[1.0],  # 100% em um ativo
            start_date=self.start_date,
            end_date=self.end_date,
            monthly_return_rate=0.01  # 1% ao mês
        )
        
        # Verificar resultado
        assert isinstance(result, pd.DataFrame), "Resultado deve ser um DataFrame"
        assert not result.empty, "Resultado não deve estar vazio"
        assert len(result) > 0, "Resultado deve ter pelo menos uma linha"
        
        # Verificar colunas
        expected_columns = ['Accumulated_Capital', 'Monthly_Contribution']
        for col in expected_columns:
            assert col in result.columns, f"Coluna {col} não encontrada"
    
    def test_simulation_parameters(self):
        #Testa parâmetros de simulação
        simulator = InvestmentSimulator(
            initial_capital=self.initial_capital,
            monthly_contribution=500,  
            monthly_interest_rate=2.0  
        )
        
        # Testar com diferentes períodos
        short_result = simulator.simulate_fixed_interest_scenario(
            start_date='2023-01-01',
            end_date='2023-06-01'
        )
        
        long_result = simulator.simulate_fixed_interest_scenario(
            start_date='2023-01-01',
            end_date='2025-01-01'
        )
        
        # Verificar que períodos diferentes geram resultados diferentes
        assert len(short_result) < len(long_result), "Período maior deve gerar mais linhas"
        
        # Verificar que aportes maiores geram capital final maior
        short_final = short_result.iloc[-1]['Accumulated_Capital']
        long_final = long_result.iloc[-1]['Accumulated_Capital']
        assert long_final > short_final, "Aporte maior deve gerar capital final maior"
    
    def test_monthly_rate_impact(self):
        # Testa impacto da taxa mensal no resultado
        # Simulador com taxa baixa
        low_rate_sim = InvestmentSimulator(
            initial_capital=self.initial_capital,
            monthly_contribution=self.monthly_contribution,
            monthly_interest_rate=0.5  # 0.5% ao mês
        )
        
        # Simulador com taxa alta
        high_rate_sim = InvestmentSimulator(
            initial_capital=self.initial_capital,
            monthly_contribution=self.monthly_contribution,
            monthly_interest_rate=2.0  # 2% ao mês
        )
        
        # Executar simulações
        low_result = low_rate_sim.simulate_fixed_interest_scenario(
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        high_result = high_rate_sim.simulate_fixed_interest_scenario(
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Taxa maior deve gerar capital final maior
        low_final = low_result.iloc[-1]['Accumulated_Capital']
        high_final = high_result.iloc[-1]['Accumulated_Capital']
        assert high_final > low_final, "Taxa maior deve gerar capital final maior"
    
    def test_contribution_impact(self):
        # Testa impacto do aporte mensal no resultado
        simulator = InvestmentSimulator(
            initial_capital=self.initial_capital,
            monthly_contribution=500,  # Aporte baixo
            monthly_interest_rate=1.0
        )
        
        # Simular com aportes diferentes
        low_contribution = simulator.simulate_fixed_interest_scenario(
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Mudar aporte
        simulator.monthly_contribution = 2000  # Aporte alto
        high_contribution = simulator.simulate_fixed_interest_scenario(
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Aporte maior deve gerar capital final maior
        low_final = low_contribution.iloc[-1]['Accumulated_Capital']
        high_final = high_contribution.iloc[-1]['Accumulated_Capital']
        assert high_final > low_final, "Aporte maior deve gerar capital final maior"
    
    def test_invalid_dates(self):
        # Testa comportamento com datas inválidas
        simulator = InvestmentSimulator(
            initial_capital=self.initial_capital,
            monthly_contribution=self.monthly_contribution,
            monthly_interest_rate=1.0
        )
        
        # Data final anterior à inicial - deve gerar erro ou retornar DataFrame vazio
        try:
            result = simulator.simulate_fixed_interest_scenario(
                start_date='2024-01-01',
                end_date='2023-01-01'
            )
            # Se não levantou exceção, deve retornar DataFrame vazio
            assert result.empty, "Datas inválidas devem retornar DataFrame vazio"
        except Exception as e:
            # Se levantou exceção, deve ser do tipo esperado
            assert isinstance(e, (ValueError, KeyError)), f"Exceção deve ser ValueError ou KeyError, não {type(e)}"
    
    def test_compare_scenarios(self):
        """Testa comparação de cenários"""
        simulator = InvestmentSimulator(
            initial_capital=self.initial_capital,
            monthly_contribution=self.monthly_contribution,
            monthly_interest_rate=1.0
        )
        
        # Simular ambos os cenários
        fixed_interest = simulator.simulate_fixed_interest_scenario(
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Criar dados sintéticos para carteira
        dates = pd.date_range(self.start_date, self.end_date, freq='D')
        stock_data = {
            'PETR4.SA': pd.DataFrame({
                'Close': [100 + i*0.1 for i in range(len(dates))],
                'Data': dates
            }).set_index('Data')
        }
        
        stock_portfolio = simulator.simulate_stock_portfolio_scenario(
            stock_data=stock_data,
            weights=[1.0],
            start_date=self.start_date,
            end_date=self.end_date,
            monthly_return_rate=0.01  # Passar explicitamente
        )
        
        # Comparar cenários
        comparison = simulator.compare_scenarios(fixed_interest, stock_portfolio)
        
        assert not comparison.empty, "Comparação não deve estar vazia"
        assert 'Fixed_Interest_Capital' in comparison.columns, "Deve ter coluna de juros fixos"
        assert 'Stock_Portfolio_Capital' in comparison.columns, "Deve ter coluna de carteira"
        assert 'Capital_Difference' in comparison.columns, "Deve ter coluna de diferença"
    
    def test_simulator_creation(self):
        """Testa criação básica do simulador"""
        simulator = InvestmentSimulator(
            initial_capital=5000,
            monthly_contribution=500,
            monthly_interest_rate=0.5
        )
        
        # Verificar atributos básicos
        assert simulator.initial_capital == 5000, "Capital inicial deve ser 5000"
        assert simulator.monthly_contribution == 500, "Aporte mensal deve ser 500"
        assert simulator.monthly_interest_rate == 0.5, "Taxa de juros deve ser 0.5%"
        assert simulator.factory is not None, "Factory deve ser criada"

if __name__ == "__main__":
    pytest.main([__file__])
