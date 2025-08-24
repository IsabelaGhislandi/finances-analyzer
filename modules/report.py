# modules/reports.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
import logging
import os

logger = logging.getLogger(__name__)

class ReportGenerator(ABC):
    #Interface base para geradores de relatórios
    @abstractmethod
    def generate_report(self, data: Dict, **kwargs) -> bool:

        pass

class ReportGenerator(ReportGenerator):
    #Gerador de relatórios simples e unificado
    
    def __init__(self, output_dir: str = 'outputs'):
        self.output_dir = output_dir
        self.setup_style()
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"SimpleReportGenerator inicializado - output: {output_dir}")
    
    def setup_style(self):
        #Configura estilo padrão dos gráficos
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Configurações para melhor qualidade
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 10
    
    def generate_report(self, data: Dict, **kwargs) -> bool:
        #Gera relatório baseado no tipo de dados"""
        report_type = kwargs.get('report_type', 'auto')
        
        try:
            if report_type == 'phase1' or 'stock_data' in data:
                self._generate_phase1_report(data, **kwargs)
            elif report_type == 'phase2' or 'comparacao' in data:
                self._generate_phase2_report(data, **kwargs)
            else:
                self._generate_generic_report(data, **kwargs)
            
            logger.info("Relatório gerado com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {e}")
            return False
    
    def _generate_phase1_report(self, data: Dict, **kwargs):
        #Gera relatório da Fase 1: Análise exploratória
        
        # Gráfico de preços das ações
        if 'stock_data' in data:
            self._plot_stock_prices(data['stock_data'], **kwargs)
        
        # Gráfico de correlação
        if 'stock_data' in data and len(data['stock_data']) > 1:
            self._plot_correlation_heatmap(data['stock_data'], **kwargs)
        
        #  Gráfico de retornos
        if 'stock_data' in data:
            self._plot_returns(data['stock_data'], **kwargs)
    
    def _generate_phase2_report(self, data: Dict, **kwargs):
        #Gera relatório da Fase 2: Simulação de investimentos"""
        print(f"🔍 DEBUG: Gerando relatório Fase 2 com dados: {list(data.keys())}")
        
        # 1. Gráfico comparativo dos cenários
        if 'comparacao' in data:
            print(f"✅ Gerando gráfico comparativo...")
            self._plot_scenario_comparison(data['comparacao'], **kwargs)
        else:
            print(f"❌ Dados de comparação não encontrados")
        
        # 2. Gráfico de evolução do capital
        if 'juros_fixos' in data and 'carteira_acoes' in data:
            print(f"✅ Gerando gráfico de evolução do capital...")
            self._plot_capital_evolution(data, **kwargs)
        else:
            print(f"❌ Dados de evolução não encontrados: juros_fixos={('juros_fixos' in data)}, carteira_acoes={('carteira_acoes' in data)}")
        
        # 3. Gráfico de métricas comparativas
        if 'metricas_juros' in data and 'metricas_carteira' in data:
            print(f"✅ Gerando gráfico de métricas...")
            self._plot_metrics_comparison(data, **kwargs)
        else:
            print(f"❌ Dados de métricas não encontrados: metricas_juros={('metricas_juros' in data)}, metricas_carteira={('metricas_carteira' in data)}")

    def _plot_stock_prices(self, stock_data: Dict, **kwargs):
        #Plota preços das ações
        fig, ax = plt.subplots(figsize=(12, 6))
        
        valid_data_count = 0
        for ticker, data in stock_data.items():
            if 'Close' in data.columns:
                ax.plot(data.index, data['Close'], label=ticker, linewidth=2)
                valid_data_count += 1
                print(f"✅ Plotando {ticker} usando preços de fechamento")
            else:
                print(f"❌ Coluna 'Close' não encontrada para {ticker}")
        
        if valid_data_count == 0:
            print("⚠️ Nenhum dado válido encontrado para plotar preços")
            plt.close(fig)
            return
        
        ax.set_title('Evolução dos Preços das Ações', fontsize=14, fontweight='bold')
        ax.set_xlabel('Data')
        ax.set_ylabel('Preço (R$)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Corrigir rotação e layout
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Salvar e exibir
        filename = f"{self.output_dir}/precos_acoes.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print(f"📊 Gráfico de preços salvo: {filename}")
    
    def _plot_correlation_heatmap(self, stock_data: Dict, **kwargs):
        #Plota mapa de correlação
        if len(stock_data) < 2:
            return
        
        # Calcular correlações
        returns_data = {}
        for ticker, data in stock_data.items():
            if 'Close' in data.columns:
                returns_data[ticker] = data['Close'].pct_change().dropna()
                print(f"✅ Calculando correlação para {ticker}")
            else:
                print(f"❌ Coluna 'Close' não encontrada para {ticker}")
        
        if len(returns_data) < 2:
            print("⚠️ Dados insuficientes para mapa de correlação")
            return
        
        correlation_df = pd.DataFrame(returns_data).corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_df, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        ax.set_title('Correlação entre Ações', fontsize=14, fontweight='bold')
        
        # Corrigir layout
        plt.tight_layout()
        
        # Salvar e exibir
        filename = f"{self.output_dir}/correlacao_acoes.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print(f"📊 Mapa de correlação salvo: {filename}")
    
    def _plot_returns(self, stock_data: Dict, **kwargs):
        """Plota retornos das ações"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        valid_data_count = 0
        for ticker, data in stock_data.items():
            if 'Close' in data.columns:
                returns = data['Close'].pct_change().dropna()
                ax.plot(returns.index, returns, label=ticker, alpha=0.7)
                valid_data_count += 1
                print(f"✅ Plotando retornos de {ticker}")
            else:
                print(f"❌ Coluna 'Close' não encontrada para {ticker}")
        
        if valid_data_count == 0:
            print("⚠️ Nenhum dado válido encontrado para plotar retornos")
            plt.close(fig)
            return
        
        ax.set_title('Retornos Diários das Ações', fontsize=14, fontweight='bold')
        ax.set_xlabel('Data')
        ax.set_ylabel('Retorno (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Corrigir rotação e layout
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Salvar e exibir
        filename = f"{self.output_dir}/retornos_acoes.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print(f"📊 Gráfico de retornos salvo: {filename}")
    
    def _plot_scenario_comparison(self, comparacao: pd.DataFrame, **kwargs):
        #Plota comparação dos cenários
        if comparacao.empty:
            return
        
        # Identificar colunas
        juros_col = self._find_column(comparacao, ['Fixed_Interest_Capital', 'Juros_Fixos'])
        carteira_col = self._find_column(comparacao, ['Stock_Portfolio_Capital', 'Carteira_Acoes'])
        
        if juros_col is None or carteira_col is None:
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        ax.plot(comparacao.index, comparacao[juros_col], 
               label='Juros Fixos', linewidth=2, color='green', marker='o')
        ax.plot(comparacao.index, comparacao[carteira_col], 
               label='Carteira de Ações', linewidth=2, color='blue', marker='s')
        
        ax.set_title('Comparação: Juros Fixos vs Carteira de Ações', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Data')
        ax.set_ylabel('Capital (R$)')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Corrigir rotação e layout
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Salvar e exibir
        filename = f"{self.output_dir}/comparacao_cenarios.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print(f"📊 Gráfico comparativo salvo: {filename}")
    
    def _plot_capital_evolution(self, data: Dict, **kwargs):
        #Plota evolução do capital
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Gráfico 1: Juros Fixos
        if 'juros_fixos' in data:
            juros_data = data['juros_fixos']
            capital_col = self._find_column(juros_data, ['Capital_Acumulado', 'Accumulated_Capital'])
            if capital_col:
                ax1.plot(juros_data.index, juros_data[capital_col], 
                        color='green', linewidth=2, marker='o')
                ax1.set_title('Evolução do Capital - Juros Fixos', fontweight='bold')
                ax1.set_ylabel('Capital (R$)')
                ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Carteira de Ações
        if 'carteira_acoes' in data:
            carteira_data = data['carteira_acoes']
            capital_col = self._find_column(carteira_data, ['Capital_Acumulado', 'Accumulated_Capital'])
            if capital_col:
                ax2.plot(carteira_data.index, carteira_data[capital_col], 
                        color='blue', linewidth=2, marker='s')
                ax2.set_title('Evolução do Capital - Carteira de Ações', fontweight='bold')
                ax2.set_xlabel('Data')
                ax2.set_ylabel('Capital (R$)')
                ax2.grid(True, alpha=0.3)
        
        # Corrigir rotação e layout
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Salvar e exibir
        filename = f"{self.output_dir}/evolucao_capital.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print(f"📊 Gráfico de evolução salvo: {filename}")
    
    def _plot_metrics_comparison(self, data: Dict, **kwargs):
        #Plota comparação de métricas
        juros_metrics = data.get('metricas_juros', {})
        carteira_metrics = data.get('metricas_carteira', {})
        
        if not juros_metrics or not carteira_metrics:
            return
        
        # Selecionar métricas para comparar
        metrics_to_compare = ['CAGR', 'Annual_Volatility', 'Max_Drawdown', 'Sharpe_Ratio']
        available_metrics = [m for m in metrics_to_compare 
                           if m in juros_metrics and m in carteira_metrics]
        
        if not available_metrics:
            return
        
        # Criar gráfico de barras
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(available_metrics))
        width = 0.35
        
        juros_values = [juros_metrics[m] for m in available_metrics]
        carteira_values = [carteira_metrics[m] for m in available_metrics]
        
        ax.bar(x - width/2, juros_values, width, label='Juros Fixos', color='green', alpha=0.7)
        ax.bar(x + width/2, carteira_values, width, label='Carteira de Ações', color='blue', alpha=0.7)
        
        ax.set_title('Comparação de Métricas', fontsize=14, fontweight='bold')
        ax.set_ylabel('Valor')
        ax.set_xticks(x)
        ax.set_xticklabels(available_metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salvar e exibir
        filename = f"{self.output_dir}/comparacao_metricas.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print(f"📊 Gráfico de métricas salvo: {filename}")
    
    def _plot_simple_line(self, df: pd.DataFrame, title: str, **kwargs):
        #Plota gráfico de linha simples"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Encontrar coluna numérica
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return
        
        # Plotar primeira coluna numérica
        col = numeric_cols[0]
        ax.plot(df.index, df[col], linewidth=2, marker='o')
        ax.set_title(f'{title}', fontweight='bold')
        ax.set_xlabel('Data')
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
        
        # Corrigir rotação e layout
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Salvar e exibir
        filename = f"{self.output_dir}/{title.lower().replace(' ', '_')}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print(f"📊 Gráfico simples salvo: {filename}")
    
    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        for name in possible_names:
            if name in df.columns:
                return name
        return None

class ReportFactory:
    #Factory para criar geradores de relatórios
    @staticmethod
    def create_generator(report_type: str = 'simple', **kwargs) -> ReportGenerator:
        if report_type.lower() == 'simple':
            return ReportGenerator(**kwargs)
        else:
            raise ValueError(f"Tipo de relatório não suportado: {report_type}")
