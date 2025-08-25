import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class PortfolioInterface(ABC):
    #Interface para gerenciadores de portfólio
    
    @abstractmethod
    def analyze_portfolio(self) -> Dict[str, any]:
        #Analisa o portfólio atual
        pass
    
    @abstractmethod
    def calculate_risk_metrics(self) -> Dict[str, float]:
        #Calcula métricas de risco"""
        pass

class PortfolioManager(PortfolioInterface):
    #Gerenciador completo de portfólio (análise + rebalanceamento)
    
    def __init__(self, tickers: List[str], weights: List[float]):
        self.tickers = tickers
        self.weights = np.array(weights)
        self._validate_portfolio()
        
        logger.info(f"PortfolioManager criado: {len(tickers)} ativos")
    
    def _validate_portfolio(self):
        #Valida se o portfólio está correto
        if len(self.tickers) != len(self.weights):
            raise ValueError("Número de tickers deve ser igual ao número de pesos")
        
        if not np.isclose(self.weights.sum(), 1.0, atol=1e-6):
            raise ValueError("Pesos devem somar 1.0")
        
        if np.any(self.weights < 0):
            raise ValueError("Pesos não podem ser negativos")
    
    def analyze_portfolio(self) -> Dict[str, any]:
        #Análise completa do portfólio
        analysis = {
            'allocation': self._analyze_allocation(),
            'concentration_risk': self._calculate_concentration_risk(),
            'diversification_score': self._calculate_diversification_score(),
            'summary': self._generate_portfolio_summary()
        }
        
        return analysis
    
    def _analyze_allocation(self) -> Dict[str, float]:
        #Analisa a alocação atual do portfólio
        allocation = {}
        for ticker, weight in zip(self.tickers, self.weights):
            allocation[ticker] = weight * 100
        return allocation
    
    def _calculate_concentration_risk(self) -> Dict[str, float]:
        #Calcula risco de concentração
        hhi = np.sum(self.weights ** 2)
        
        return {
            'HHI': hhi,
            'Effective_Assets': 1 / hhi,
            'Concentration_Level': self._classify_concentration(hhi)
        }
    
    def _classify_concentration(self, hhi: float) -> str:
        #Classifica o nível de concentração
        if hhi < 0.15:
            return "Baixa"
        elif hhi < 0.25:
            return "Média"
        elif hhi < 0.50:
            return "Alta"
        else:
            return "Muito Alta"
    
    def _calculate_diversification_score(self) -> float:
        #Calcula score de diversificação (0-100)
        num_assets = len(self.tickers)
        weight_balance = 1 - np.std(self.weights)
        diversification_score = min(100, (num_assets * 20) + (weight_balance * 50))
        return round(diversification_score, 1)
    
    def _generate_portfolio_summary(self) -> Dict[str, any]:
        #Gera resumo executivo do portfólio
        allocation = self._analyze_allocation()
        concentration = self._calculate_concentration_risk()
        diversification = self._calculate_diversification_score()
        
        max_weight_idx = np.argmax(self.weights)
        min_weight_idx = np.argmin(self.weights)
        
        return {
            'total_assets': len(self.tickers),
            'largest_position': {
                'ticker': self.tickers[max_weight_idx],
                'weight': allocation[self.tickers[max_weight_idx]]
            },
            'smallest_position': {
                'ticker': self.tickers[min_weight_idx],
                'weight': allocation[self.tickers[min_weight_idx]]
            },
            'concentration_level': concentration['Concentration_Level'],
            'diversification_score': diversification,
            'recommendation': self._generate_recommendation(concentration, diversification)
        }
    
    def _generate_recommendation(self, concentration: Dict, diversification: float) -> str:
        #G recomendação baseada na análise
        if concentration['Concentration_Level'] == "Muito Alta" or diversification < 30:
            return "Considerar diversificar mais o portfólio"
        elif concentration['Concentration_Level'] == "Alta" or diversification < 50:
            return "Portfólio moderadamente concentrado"
        elif concentration['Concentration_Level'] == "Média" or diversification < 70:
            return "Portfólio bem diversificado"
        else:
            return "Portfólio muito bem diversificado"
    
    def suggest_rebalancing(self, target_weights: List[float]) -> Dict[str, Dict[str, float]]:
        #Sugere rebalanceamento para pesos alvo
        target_weights = np.array(target_weights)
        
        if len(target_weights) != len(self.weights):
            raise ValueError("Pesos alvo devem ter mesmo tamanho dos pesos atuais")
        
        weight_diff = target_weights - self.weights
        
        rebalancing = {}
        for i, ticker in enumerate(self.tickers):
            current_weight = self.weights[i] * 100
            target_weight = target_weights[i] * 100
            weight_change = weight_diff[i] * 100
            
            rebalancing[ticker] = {
                'Current_Weight': current_weight,
                'Target_Weight': target_weight,
                'Weight_Change': weight_change,
                'Action': 'Comprar' if weight_change > 0 else 'Vender' if weight_change < 0 else 'Manter'
            }
        
        return rebalancing
    
    def calculate_risk_metrics(self) -> Dict[str, float]:
        #calcula métricas de risco do portfólio
        risk_metrics = {
            'concentration_risk': self._calculate_concentration_risk()['HHI'],
            'diversification_score': self._calculate_diversification_score(),
            'num_assets': len(self.tickers),
            'weight_std': float(np.std(self.weights))
        }
        
        return risk_metrics

class PortfolioFactory:
    @staticmethod
    def create_portfolio_manager(tickers: List[str], weights: List[float]) -> PortfolioManager:
        #Cria um gerenciador de portfólio
        return PortfolioManager(tickers, weights)

# Funções de conveniência
def create_portfolio_manager(tickers: List[str], weights: List[float]) -> PortfolioManager:
    return PortfolioFactory.create_portfolio_manager(tickers, weights)