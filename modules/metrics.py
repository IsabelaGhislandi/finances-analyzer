# modules/metrics.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class MetricsCalculatorInterface(ABC):
    @abstractmethod
    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        pass

class PerformanceMetrics(MetricsCalculatorInterface):
    
    def __init__(self):
        logger.info("PerformanceMetrics inicializado")
    
    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return {}
        
        #Identificar coluna de capital
        capital_col = self._get_capital_column(df)
        if capital_col is None:
            return {}

        final_capital = df[capital_col].iloc[-1]
        initial_capital = df[capital_col].iloc[0]
        period_years = len(df) / 12
        
        metrics = {
            'Final_Capital': final_capital,
            'Period_Years': period_years,
            'Total_Return': (final_capital / initial_capital - 1) * 100
        }
        
        if period_years > 0:
            metrics['CAGR'] = (final_capital / initial_capital) ** (1 / period_years) - 1

        if 'Retorno_Mensal' in df.columns:
            monthly_returns = df['Retorno_Mensal'].dropna()
            
            if len(monthly_returns) > 1:
                # Volatilidade Anualizada
                annual_volatility = monthly_returns.std() * np.sqrt(12) * 100
                metrics['Annual_Volatility'] = annual_volatility
                
                # Sharpe Ratio (se tiver CAGR e volatilidade)
                if 'CAGR' in metrics and annual_volatility > 0:
                    sharpe_ratio = metrics['CAGR'] / (annual_volatility / 100)
                    metrics['Sharpe_Ratio'] = sharpe_ratio
                
                # Sortino Ratio (só retornos negativos)
                negative_returns = monthly_returns[monthly_returns < 0]
                if len(negative_returns) > 1:
                    downside_deviation = negative_returns.std() * np.sqrt(12) * 100
                    if downside_deviation > 0 and 'CAGR' in metrics:
                        sortino_ratio = metrics['CAGR'] / (downside_deviation / 100)
                        metrics['Sortino_Ratio'] = sortino_ratio
        
        # MÉTRICAS DE RISCO
        if capital_col:
            # Máximo Drawdown
            cumulative_returns = df[capital_col]
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max * 100
            max_drawdown = drawdown.min()
            metrics['Max_Drawdown'] = max_drawdown
            
            # Calmar Ratio (CAGR / Max Drawdown)
            if 'CAGR' in metrics and abs(max_drawdown) > 0:
                calmar_ratio = metrics['CAGR'] / abs(max_drawdown)
                metrics['Calmar_Ratio'] = calmar_ratio
        
        logger.info(f"Métricas calculadas: {len(metrics)} indicadores")
        return metrics
    
    def _get_capital_column(self, df: pd.DataFrame) -> str:
        #Identifica qual coluna contém o capital"""
        possible_columns = ['Capital_Acumulado', 'Accumulated_Capital', 'Capital']
        for col in possible_columns:
            if col in df.columns:
                return col
        return None
    
    def get_metrics_summary(self, metrics: Dict[str, float]) -> str:
        #Gera resumo executivo das métricas
        if not metrics:
            return "Nenhuma métrica disponível"
        
        summary_lines = []
        summary_lines.append("📊 RESUMO DAS MÉTRICAS:")
        summary_lines.append("-" * 50)
        
        if 'Final_Capital' in metrics:
            summary_lines.append(f"💰 Capital Final: R$ {metrics['Final_Capital']:,.2f}")
        
        if 'Total_Return' in metrics:
            summary_lines.append(f"📈 Retorno Total: {metrics['Total_Return']:.2f}%")
        
        if 'CAGR' in metrics:
            summary_lines.append(f"📊 CAGR: {metrics['CAGR']*100:.2f}%")
        
        # Métricas de risco
        if 'Annual_Volatility' in metrics:
            summary_lines.append(f" Volatilidade Anual: {metrics['Annual_Volatility']:.2f}%")
        
        if 'Max_Drawdown' in metrics:
            summary_lines.append(f" Máximo Drawdown: {metrics['Max_Drawdown']:.2f}%")
        
        # Métricas de performance
        if 'Sharpe_Ratio' in metrics:
            summary_lines.append(f"📊 Sharpe Ratio: {metrics['Sharpe_Ratio']:.3f}")
        
        if 'Sortino_Ratio' in metrics:
            summary_lines.append(f"📊 Sortino Ratio: {metrics['Sortino_Ratio']:.3f}")
        
        if 'Calmar_Ratio' in metrics:
            summary_lines.append(f"📊 Calmar Ratio: {metrics['Calmar_Ratio']:.3f}")
        
        return "\n".join(summary_lines)
    
    def compare_metrics(self, metrics1: Dict[str, float], metrics2: Dict[str, float], 
                       name1: str = "Cenário 1", name2: str = "Cenário 2") -> str:
        #Compara métricas entre dois cenários
        if not metrics1 or not metrics2:
            return "Não é possível comparar métricas vazias"
        
        comparison_lines = []
        comparison_lines.append(f" COMPARAÇÃO: {name1} vs {name2}")
        comparison_lines.append("=" * 60)
        
        # Métricas para comparar
        key_metrics = ['CAGR', 'Annual_Volatility', 'Max_Drawdown', 'Sharpe_Ratio']
        
        for metric in key_metrics:
            if metric in metrics1 and metric in metrics2:
                value1 = metrics1[metric]
                value2 = metrics2[metric]
                
                if metric in ['CAGR', 'Sharpe_Ratio']:
                    # Para métricas onde maior é melhor
                    if value1 > value2:
                        winner = f"🏆 {name1}"
                        diff = f"+{(value1 - value2):.3f}"
                    else:
                        winner = f"🏆 {name2}"
                        diff = f"+{(value2 - value1):.3f}"
                else:
                    # Para métricas onde menor é melhor (volatilidade, drawdown)
                    if value1 < value2:
                        winner = f"🏆 {name1}"
                        diff = f"-{(value2 - value1):.3f}"
                    else:
                        winner = f"🏆 {name2}"
                        diff = f"-{(value1 - value2):.3f}"
                
                comparison_lines.append(f"{metric:>15}: {name1:>10} {value1:>8.3f} | {name2:>10} {value2:>8.3f} | {winner} ({diff})")
        
        return "\n".join(comparison_lines)

class MetricsCalculatorFactory:
  
    @staticmethod
    def create_calculator(calculator_type: str = 'performance') -> MetricsCalculatorInterface:
        #Cria uma calculadora de métricas baseada no tipo
        if calculator_type == 'performance':
            return PerformanceMetrics()
        else:
            raise ValueError(f"Tipo de calculadora não suportado: {calculator_type}")
    
    @staticmethod
    def create_default_calculator() -> PerformanceMetrics:
        #Cria a calculadora padrão (PerformanceMetrics)
        return PerformanceMetrics()

# Funções de conveniência
def create_metrics_calculator(calculator_type: str = 'performance') -> MetricsCalculatorInterface:
    return MetricsCalculatorFactory.create_calculator(calculator_type)

def create_default_metrics_calculator() -> PerformanceMetrics:
    return MetricsCalculatorFactory.create_default_calculator()