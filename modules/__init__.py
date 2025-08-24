# modules/__init__.py
from .data import DataCollectorFactory
from .metrics import MetricsCalculatorFactory
from .portfolio import PortfolioFactory
from .report import ReportFactory
from .interests import InvestmentSimulatorFactory

class InvestmentAnalysisFactory:
    """Factory central que orquestra todo o sistema"""
    def __init__(self):
        self.data_factory = DataCollectorFactory()
        self.metrics_factory = MetricsCalculatorFactory()
        self.portfolio_factory = PortfolioFactory()
        self.report_factory = ReportFactory()
        self.interests_factory = InvestmentSimulatorFactory()
    
    def create_complete_system(self, config: dict, tickers=None, weights=None):
        
        system = {
        'data_collector': self.data_factory.create_collector('yfinance'),
        'metrics_calculator': self.metrics_factory.create_calculator('performance'),
        'portfolio_analyzer': self.portfolio_factory.create_portfolio_manager(tickers or [], weights or []),
        'report_generator': self.report_factory.create_generator('simple'),
        'investment_simulator': self.interests_factory
        }
        
        return system

# Exportar para uso fácil
__all__ = [
    'InvestmentAnalysisFactory',
    'DataCollectorFactory',
    'MetricsCalculatorFactory',
    'PortfolioFactory',
    'ReportFactory',
    'InvestmentSimulatorFactory'
]