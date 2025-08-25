from .data import DataCollectorFactory
from .metrics import MetricsCalculatorFactory
from .portfolio import PortfolioFactory
from .report import ReportFactory
from .interests import InvestmentSimulatorFactory
from .forecast import ForecastFactory

class InvestmentAnalysisFactory:
    # Factory central que orquestra todo o sistema
    def __init__(self):
        self.data_factory = DataCollectorFactory()
        self.metrics_factory = MetricsCalculatorFactory()
        self.portfolio_factory = PortfolioFactory()
        self.report_factory = ReportFactory()
        self.interests_factory = InvestmentSimulatorFactory()
        self.forecast_factory = ForecastFactory()
    
    def create_complete_system(self, config: dict, tickers=None, weights=None):
        
        system = {
        'data_collector': self.data_factory.create_collector('yfinance'),
        'metrics_calculator': self.metrics_factory.create_calculator('performance'),
        'report_generator': self.report_factory.create_generator('simple'),
        'investment_simulator': self.interests_factory,
        'forecast_manager': self.forecast_factory.create_forecaster('arima'),
        'backtest_manager': self.forecast_factory.create_backtest_manager(60)
        }
        
        # Só criar portfolio_analyzer se houver pesos
        if weights and len(weights) == len(tickers):
            system['portfolio_analyzer'] = self.portfolio_factory.create_portfolio_manager(tickers or [], weights or [])
        
        return system

# Exportar para uso fácil
__all__ = [
    'InvestmentAnalysisFactory',
    'DataCollectorFactory',
    'MetricsCalculatorFactory',
    'PortfolioFactory',
    'ReportFactory',
    'InvestmentSimulatorFactory',
    'ForecastFactory'
]