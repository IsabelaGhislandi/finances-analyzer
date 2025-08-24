# Módulo para cálculos de juros compostos e simulação de aportes
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from modules.data import DataCollectorFactory

logger = logging.getLogger(__name__)

class SimulatorInterface(ABC):
    @abstractmethod
    def simulate(self, aporte_mensal: float, data_inicio: str, data_fim: str) -> pd.DataFrame:
        #Simula evolução do investimento
        pass
    
    @abstractmethod
    def get_scenario(self) -> str:
        #Retorna o nome do cenário
        pass

class MetricsCalculatorInterface(ABC):
    @abstractmethod
    def calcular_metricas(self, df: pd.DataFrame) -> Dict[str, float]:
        #Calcula métricas de performance"""
        pass

class CompoundInterestSimulator(SimulatorInterface):
    #Simulador de juros compostos com aportes mensais
    def __init__(self, initial_capital: float, monthly_rate: float):
        self.initial_capital = initial_capital
        self.monthly_rate = monthly_rate / 100
        logger.info(f"Simulador de juros compostos: Capital R$ {initial_capital:,.2f}, Taxa {monthly_rate:.2f}%")
    
    def simulate(self, monthly_contribution: float, start_date: str, end_date: str) -> pd.DataFrame:
        #Simula juros compostos com aportes mensais
        start, end = pd.to_datetime(start_date), pd.to_datetime(end_date)
        monthly_dates = pd.date_range(start=start, end=end, freq='MS')
        business_dates = [self._next_business_day(date) for date in monthly_dates]
        
        current_capital = self.initial_capital
        results = []
        
        for i, date in enumerate(business_dates):
            monthly_interest = current_capital * self.monthly_rate
            current_capital += monthly_contribution + monthly_interest
            
            results.append({
                'Date': date,
                'Accumulated_Capital': current_capital,  
                'Monthly_Contribution': monthly_contribution,  
                'Monthly_Interest': monthly_interest,  
                'Total_Contributions': monthly_contribution * (i + 1),  
                'Accumulated_Interest': sum(r['Monthly_Interest'] for r in results) + monthly_interest  
            })
        
        df = pd.DataFrame(results)
        df['Retorno_Mensal'] = df['Accumulated_Capital'].pct_change()
        df['Retorno_Acumulado'] = (df['Accumulated_Capital'] / self.initial_capital - 1) * 100
        df['Scenario'] = self.get_scenario()
        
        logger.info(f"Compound interest simulation: {len(df)} months, Final capital R$ {current_capital:,.2f}")
        return df
    
    def get_scenario(self) -> str:
        return 'Compound_Interest'
    
    def _next_business_day(self, date: datetime) -> datetime:
        while date.weekday() >= 5:
            date += timedelta(days=1)
        return date

class StockPortfolioSimulator(SimulatorInterface):
    #Simulador de carteira de ações com aportes mensais
    def __init__(self, tickers: List[str], weights: List[float], initial_capital: float, 
                 monthly_return_rate: float = 0.01):
        self.tickers = tickers
        self.weights = self._normalize_weights(weights)
        self.initial_capital = initial_capital
        self.monthly_return_rate = monthly_return_rate  
        
        logger.info(f"Stock portfolio simulator: {len(tickers)} assets, "
                   f"Capital R$ {initial_capital:,.2f}, "
                   f"Retorno mensal {monthly_return_rate*100:.2f}%")
    
    def simulate(self, monthly_contribution: float, start_date: str, end_date: str, 
             stock_data: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        try:
            # Preparar datas
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            monthly_dates = pd.date_range(start=start, end=end, freq='MS')
            
            current_capital = self.initial_capital
            results = []
            
            # Verificar se tem dados reais válidos
            use_real_data = stock_data and self._has_valid_data(stock_data)
            
            if use_real_data:
                logger.info("Usando dados reais das ações")
                monthly_returns = self._get_monthly_returns(stock_data)
            else:
                logger.info("Usando retorno fixo mensal")
            
            for i, date in enumerate(monthly_dates):
                # Calcular retorno mensal
                if use_real_data and i > 0:
                    monthly_return = self._get_portfolio_return(monthly_returns, i-1)
                else:
                    monthly_return = self.monthly_return_rate
                
                # Aplicar retorno e aporte
                current_capital = current_capital * (1 + monthly_return)
                current_capital += monthly_contribution
                
                results.append({
                    'Date': date,
                    'Accumulated_Capital': current_capital,  
                    'Monthly_Contribution': monthly_contribution, 
                    'Monthly_Return': monthly_return,       
                    'Total_Contributions': monthly_contribution * (i + 1) 
                })
            
            # Criar DataFrame
            df = pd.DataFrame(results)
            df['Accumulated_Return'] = (df['Accumulated_Capital'] / self.initial_capital - 1) * 100
            df['Scenario'] = self.get_scenario()
            
            simulation_type = "dados reais" if use_real_data else "retorno fixo"
            logger.info(f"Simulação com {simulation_type} concluída: Capital final R$ {current_capital:,.2f}")
            
            return df
            
        except Exception as e:
            logger.error(f"Erro na simulação: {e}")
            return pd.DataFrame()
    
    def _has_valid_data(self, stock_data: Dict[str, pd.DataFrame]) -> bool:
        #Verifica se os dados são válidos
        try:
            return (stock_data and 
                   all('Close' in data.columns for data in stock_data.values()) and
                   all(len(data) > 1 for data in stock_data.values()))
        except:
            return False
    
    def _get_monthly_returns(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        #Calcula retornos mensais das ações
        returns = {}
        for ticker, data in stock_data.items():
            monthly_prices = data['Close'].resample('ME').last()
            returns[ticker] = monthly_prices.pct_change().dropna()
        return returns
    
    def _get_portfolio_return(self, monthly_returns: Dict[str, pd.Series], month_index: int) -> float:
        #Calcula retorno mensal ponderado do portfólio
        return sum(weight * monthly_returns[ticker].iloc[month_index] 
                  for ticker, weight in zip(self.tickers, self.weights)
                  if ticker in monthly_returns and month_index < len(monthly_returns[ticker]))
    
    def get_scenario(self) -> str:
        return 'Stock_Portfolio'
    
    def _normalize_weights(self, weights: List[float]) -> np.ndarray:
        weights_array = np.array(weights)
        if np.any(weights_array < 0):
            raise ValueError("Weights cannot be negative")
        return weights_array / weights_array.sum()

class BasicMetricsCalculator(MetricsCalculatorInterface):
    #Calculadora básica de métricas"""
    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return {}
        
        final_capital = df['Accumulated_Capital'].iloc[-1]
        period_years = len(df) / 12
        
        metrics = {
            'Final_Capital': final_capital,
            'Period_Years': period_years,
            'Total_Return': (final_capital / df['Accumulated_Capital'].iloc[0] - 1) * 100
        }
        
        if period_years > 0:
            metrics['CAGR'] = (final_capital / df['Accumulated_Capital'].iloc[0]) ** (1 / period_years) - 1
        
        return metrics

class InvestmentSimulatorFactory:
    #Factory para criar simuladores de investimento
    
    @staticmethod
    def create_simulator(simulator_type: str, **kwargs) -> SimulatorInterface:
        #Cria um simulador baseado no tipo especificado
        if simulator_type == 'compound_interest':
            return CompoundInterestSimulator(
                initial_capital=kwargs.get('initial_capital', 10000.0),
                monthly_rate=kwargs.get('monthly_rate', 1.0)
            )
        
        elif simulator_type == 'stock_portfolio':
            return StockPortfolioSimulator (
                tickers=kwargs.get('tickers', []),
                weights=kwargs.get('weights', []),
                initial_capital=kwargs.get('initial_capital', 10000.0),
                monthly_return_rate=kwargs.get('monthly_return_rate', 0.01) 
        )
        
        else:
            raise ValueError(f"Unsupported simulator type: {simulator_type}")
    
    @staticmethod
    def create_metrics_calculator(calculator_type: str = 'basic') -> MetricsCalculatorInterface:
        #Cria uma calculadora de métricas
        if calculator_type == 'basic':
            return BasicMetricsCalculator()
        else:
            raise ValueError(f"Tipo de calculadora não suportado: {calculator_type}")

class InvestmentSimulator:
    # Classe principal 
    
    def __init__(self, initial_capital: float, monthly_contribution: float, monthly_interest_rate: float):
        self.initial_capital = initial_capital
        self.monthly_contribution = monthly_contribution
        self.monthly_interest_rate = monthly_interest_rate       
        self.factory = InvestmentSimulatorFactory()
        
        logger.info(f"Main simulator: Capital R$ {initial_capital:,.2f}, "
                   f"Contribution R$ {monthly_contribution:,.2f}, Rate {monthly_interest_rate:.2f}%")
    
    def simulate_fixed_interest_scenario(self, start_date: str, end_date: str) -> pd.DataFrame:
        simulator = self.factory.create_simulator(
            'compound_interest',
            initial_capital=self.initial_capital,
            monthly_rate=self.monthly_interest_rate
        )
        return simulator.simulate(self.monthly_contribution, start_date, end_date)
    
    def simulate_stock_portfolio_scenario(self, stock_data: Dict[str, pd.DataFrame], 
                                    weights: List[float], 
                                    start_date: str, end_date: str,
                                    monthly_return_rate: float = None) -> pd.DataFrame:
        simulator = self.factory.create_simulator(
        'stock_portfolio',
        tickers=list(stock_data.keys()) if stock_data else [], 
        weights=weights,
        initial_capital=self.initial_capital,
        monthly_return_rate=monthly_return_rate
        )
        return simulator.simulate(self.monthly_contribution, start_date, end_date, stock_data)
    
    def compare_scenarios(self, fixed_interest_df: pd.DataFrame, 
                     stock_portfolio_df: pd.DataFrame) -> pd.DataFrame:
        #Compara os dois cenários
        if fixed_interest_df.empty or stock_portfolio_df.empty:
            return pd.DataFrame()
        # Identificar colunas de capital (pode variar entre simuladores)
        capital_col_juros = self._get_capital_column(fixed_interest_df)
        capital_col_carteira = self._get_capital_column(stock_portfolio_df)
    
        if not capital_col_juros or not capital_col_carteira:
            return pd.DataFrame()
    
        # Identificar coluna de data (pode variar)
        date_col = 'Data' if 'Data' in fixed_interest_df else 'Date'
    
        comparison_df = pd.DataFrame({
            'Date': fixed_interest_df[date_col],
            'Fixed_Interest_Capital': fixed_interest_df[capital_col_juros],
            'Stock_Portfolio_Capital': stock_portfolio_df[capital_col_carteira],
            'Capital_Difference': stock_portfolio_df[capital_col_carteira] - fixed_interest_df[capital_col_juros]
        })
        
        return comparison_df

    def _get_capital_column(self, df: pd.DataFrame) -> str:
    #Identifica qual coluna contém o capital
        possible_columns = ['Capital_Acumulado', 'Accumulated_Capital', 'Capital']
        for col in possible_columns:
            if col in df.columns:
                return col
        return None
    
    def calculate_metrics(self, df: pd.DataFrame, metrics_type: str = 'basic') -> Dict[str, float]:
        calculator = self.factory.create_metrics_calculator(metrics_type)
        return calculator.calculate_metrics(df)

def create_simulator(simulator_type: str, **kwargs) -> SimulatorInterface:

    return InvestmentSimulatorFactory.create_simulator(simulator_type, **kwargs)

def create_metrics_calculator(calculator_type: str = 'basic') -> MetricsCalculatorInterface:

    return InvestmentSimulatorFactory.create_metrics_calculator(calculator_type)