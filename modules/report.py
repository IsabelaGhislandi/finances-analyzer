#Módulo para geração de relatórios e visualizações
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import logging
from typing import Dict
import os

logger = logging.getLogger(__name__)

# Configurar estilo dos gráficos
plt.style.use('default')
sns.set_palette("husl")

class ReportInterface(ABC):
    #Classe para gerar relatórios e gráficos

    @abstractmethod
    def generate_report(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        #Gera o relatório
        pass
    
    @abstractmethod
    def get_report_name(self) -> str:
        #Retorna o nome do tipo de relatório
        pass

class StockReport(ReportInterface):
    def __init__(self, output_dir: str = "outputs/plots", 
                 include_advanced_analysis: bool = True,
                 include_summary_stats: bool = True,
                 figsize: tuple = (12, 8)):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.figsize = figsize
        self.colors = ['#2E86C1', '#28B463', '#F39C12', '#E74C3C', '#8E44AD']
        
        # Configurações flexíveis
        self.include_advanced_analysis = include_advanced_analysis
        self.include_summary_stats = include_summary_stats

    def get_report_name(self) -> str:
        if self.include_advanced_analysis and self.include_summary_stats:
            return "Default Report"
        else:
            return "Custom Report"

    def generate_report(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        logger.info(f"Gerando {self.get_report_name()}")
        # Análises básicas (sempre incluídas)
        self.plot_comparison(data_dict)
        self.plot_correlation_heatmap(data_dict)
        
        # Análises condicionais baseadas nas configurações
        if self.include_summary_stats:
            self.generate_summary_stats(data_dict)
        
        if self.include_advanced_analysis:
            self.plot_rolling_volatility(data_dict)
            self.plot_drawdown_analysis(data_dict)
            self.plot_risk_return_scatter(data_dict)
    
    def plot_single_stock(self, data: pd.DataFrame, ticker: str, save: bool = True) -> None:
        #Gráfico de preço de uma ação individual
        #Gráfico de preço de uma ação individual
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Gráfico principal - Preço
        ax1.plot(data.index, data['Close'], linewidth=2, color=self.colors[0])
        ax1.set_title(f'{ticker} - Evolução do Preço Ajustado', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Preço (R$)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', labelbottom=False)
        
        # Adicionar informações básicas
        price_start = data['Close'].iloc[0]
        price_end = data['Close'].iloc[-1]
        total_return = (price_end / price_start - 1) * 100
        
        ax1.text(0.02, 0.95, f'Retorno Total: {total_return:.1f}%', 
                transform=ax1.transAxes, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Gráfico secundário - Volume
        ax2.bar(data.index, data['Volume'], alpha=0.6, color=self.colors[1])
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Data', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
      
        fig.tight_layout()
        
        if save:
            filename = os.path.join(self.output_dir, f'{ticker}_analysis.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico salvo: {filename}")
        
        fig.show()
    
    def plot_comparison(self, data_dict: Dict[str, pd.DataFrame], 
                       normalize: bool = True, save: bool = True) -> None:
        #Compara múltiplas ações em um gráfico

        fig, ax = plt.subplots(figsize=self.figsize)
        
        returns_data = {}
        
        for i, (ticker, data) in enumerate(data_dict.items()):
            if normalize:
                # Normalizar para base 100
                normalized_price = (data['Close'] / data['Close'].iloc[0]) * 100
                ax.plot(data.index, normalized_price, 
                       label=ticker, linewidth=2.5, color=self.colors[i % len(self.colors)])
                
                # Calcular retorno para a tabela
                final_return = (normalized_price.iloc[-1] / 100 - 1) * 100
                returns_data[ticker] = {
                    'Retorno (%)': final_return,
                    'Preço Inicial (R$)': data['Close'].iloc[0],
                    'Preço Final (R$)': data['Close'].iloc[-1]
                }
            else:
                ax.plot(data.index, data['Close'], 
                       label=ticker, linewidth=2.5, color=self.colors[i % len(self.colors)])
        
        # Configurar gráfico
        title = 'Comparação de Performance (Base 100)' if normalize else 'Comparação de Preços'
        ylabel = 'Performance Normalizada' if normalize else 'Preço (R$)'
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Data', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Rotacionar labels do eixo x
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.output_dir, 'comparison_plot.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico salvo: {filename}")
        
        plt.show()
        
        # Mostrar tabela de retornos se normalizado
        if normalize and returns_data:
            self._print_returns_table(returns_data)
    
    def plot_correlation_heatmap(self, data_dict: Dict[str, pd.DataFrame], 
                                save: bool = True) -> None:
        try:
        # Usa função de correlação robusta
            from modules.data import calculate_correlation_matrix
            corr_matrix = calculate_correlation_matrix(data_dict)
            
            if corr_matrix is None:
                logger.warning("Não foi possível calcular correlação - dados insuficientes")
                return
            
            # Plotar heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5)
            plt.title('Matriz de Correlação entre Ativos')
            plt.tight_layout()
            
            if save:
                plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
                logger.info("Heatmap de correlação salvo")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Erro ao plotar correlação: {e}")
    
    def _print_returns_table(self, returns_data: Dict) -> None:
        #Imprime tabela formatada de retornos
        
        df = pd.DataFrame.from_dict(returns_data, orient='index')
        df = df.sort_values('Retorno (%)', ascending=False)
        
        print("\n" + "="*60)
        print("📊 RANKING DE PERFORMANCE")
        print("="*60)
        print(df.round(2))
        print("="*60)
    
    def generate_summary_stats(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        #Gera estatísticas resumidas para cada ação
  
        stats = {}
        
        for ticker, data in data_dict.items():
            price_start = data['Close'].iloc[0]
            price_end = data['Close'].iloc[-1]
            
            # Calcular retornos diários
            daily_returns = data['Close'].pct_change().dropna()
            
            stats[ticker] = {
                'Preço Inicial (R$)': price_start,
                'Preço Final (R$)': price_end,
                'Retorno Total (%)': (price_end / price_start - 1) * 100,
                'Volatilidade Diária (%)': daily_returns.std() * 100,
                'Volatilidade Anual (%)': daily_returns.std() * np.sqrt(252) * 100,
                'Maior Alta (%)': daily_returns.max() * 100,
                'Maior Baixa (%)': daily_returns.min() * 100,
                'Dias de Dados': len(data)
            }
        
        df_stats = pd.DataFrame.from_dict(stats, orient='index')
        
        print("\n" + "="*80)
        print("📈 ESTATÍSTICAS RESUMIDAS")
        print("="*80)
        print(df_stats.round(2))
        print("="*80)
        
        return df_stats

class ReportFactory:
    @staticmethod
    def create_default_report(output_dir: str = "outputs/plots") -> StockReport:
        #Cria relatório com todas as análises
        return StockReport(
            output_dir=output_dir,
            include_advanced_analysis=True,
            include_summary_stats=True,
            figsize=(14, 10)
        )
    
    @staticmethod
    def create_custom_report(output_dir: str = "outputs/plots",
                           include_advanced_analysis: bool = True,
                           include_summary_stats: bool = True,
                           figsize: tuple = (12, 8)) -> StockReport:
    #Cria relatório personalizado com configurações específicas"""
        return StockReport(
            output_dir=output_dir,
            include_advanced_analysis=include_advanced_analysis,
            include_summary_stats=include_summary_stats,
            figsize=figsize
        )
    
    @staticmethod
    def get_available_configurations() -> Dict[str, str]:
        #Retorna descrições das configurações disponíveis"""
        return {
            'default': 'Relatório padrão completo com todas as análises avançadas',
            'custom': 'Relatório personalizado com configurações específicas'
        }