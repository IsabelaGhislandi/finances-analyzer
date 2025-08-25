# modules/reports.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class ReportGenerator(ABC):
    #Interface base para geradores de relatórios
    @abstractmethod
    def generate_report(self, data: Dict, **kwargs) -> bool:

        pass

class SimpleReportGenerator(ReportGenerator):
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
            if report_type == 'integrated':
                self._generate_integrated_report(data, **kwargs)
            elif report_type == 'phase1' or 'stock_data' in data:
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

    def generate_phase3_report(self, data: dict, report_type: str = 'phase3'):
        """Gera relatório da Fase 3: Previsões e Backtesting"""
        print(f"🔮 Gerando relatório da Fase 3...")
        
        if report_type == 'phase3':
            self._generate_phase3_report(data)
        else:
            print(f"⚠️ Tipo de relatório não suportado: {report_type}")

    def _generate_phase3_report(self, data: dict):
        """Gera relatório completo da Fase 3"""
        try:
            # Extrair dados
            forecasts = data.get('forecasts', {})
            backtests = data.get('backtests', {})
            stock_data = data.get('stock_data', {})
            
            if not forecasts:
                print("⚠️ Nenhuma previsão encontrada para gerar gráficos")
                return
            
            print(f"📊 Gerando {len(forecasts)} gráficos de previsões...")
            
            # 1. Gráfico de previsões para cada ativo
            self._plot_forecasts(forecasts, stock_data)
            
            # 2. Gráfico de backtesting (se houver)
            if backtests:
                self._plot_backtest_results(backtests)
            
            # 3. Gráfico comparativo de métricas
            if backtests:
                self._plot_forecast_metrics(backtests)
            
            print("✅ Gráficos da Fase 3 gerados com sucesso!")
            
        except Exception as e:
            print(f"❌ Erro ao gerar gráficos da Fase 3: {e}")

    def _plot_forecasts(self, forecasts: dict, stock_data: dict):
        # Plota previsões vs dados históricos
        try:
            fig, axes = plt.subplots(len(forecasts), 1, figsize=(12, 4*len(forecasts)))
            if len(forecasts) == 1:
                axes = [axes]
            
            for i, (ticker, forecast) in enumerate(forecasts.items()):
                ax = axes[i]
                
                # Dados históricos
                if ticker in stock_data and 'Close' in stock_data[ticker].columns:
                    historical = stock_data[ticker]['Close']
                    ax.plot(historical.index, historical.values, 
                           label='Dados Históricos', color='blue', linewidth=2)
                
                # Previsões
                ax.plot(forecast.index, forecast.values, 
                       label='Previsões', color='red', linestyle='--', linewidth=2)
                
                # Configurações
                ax.set_title(f'Previsões para {ticker} - Próximos {len(forecast)} dias')
                ax.set_xlabel('Data')
                ax.set_ylabel('Preço (R$)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Rotacionar labels do eixo X
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Salvar e mostrar
            filename = f'outputs/phase3_forecasts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            os.makedirs('outputs', exist_ok=True)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig)
            
            print(f"📈 Gráfico de previsões salvo: {filename}")
            
        except Exception as e:
            print(f"❌ Erro ao plotar previsões: {e}")

    def _plot_backtest_results(self, backtests: dict):
        #Plota resultados do backtesting
        try:
            fig, axes = plt.subplots(len(backtests), 1, figsize=(12, 4*len(backtests)))
            if len(backtests) == 1:
                axes = [axes]
            
            for i, (ticker, results) in enumerate(backtests.items()):
                ax = axes[i]
                
                # Valores reais vs previstos
                actual = results.get('actual_values', [])
                predicted = results.get('predicted_values', [])
                
                if actual and predicted:
                    # Plotar apenas os primeiros 100 pontos para clareza
                    max_points = min(100, len(actual), len(predicted))
                    x = range(max_points)
                    
                    ax.plot(x, actual[:max_points], 
                           label='Valores Reais', color='blue', linewidth=2)
                    ax.plot(x, predicted[:max_points], 
                           label='Valores Previstos', color='red', linestyle='--', linewidth=2)
                    
                    # Configurações
                    ax.set_title(f'Backtesting para {ticker}')
                    ax.set_xlabel('Período')
                    ax.set_ylabel('Preço (R$)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Salvar e mostrar
            filename = f'outputs/phase3_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig)
            
            print(f"🔄 Gráfico de backtesting salvo: {filename}")
            
        except Exception as e:
            print(f"❌ Erro ao plotar backtesting: {e}")

    def _plot_forecast_metrics(self, backtests: dict):
        # Plota métricas comparativas de previsão
        try:
            # Preparar dados
            tickers = list(backtests.keys())
            mape_values = [backtests[t]['metrics']['avg_mape'] for t in tickers]
            rmse_values = [backtests[t]['metrics']['avg_rmse'] for t in tickers]
            
            # Criar gráfico de barras
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # MAPE
            bars1 = ax1.bar(tickers, mape_values, color='skyblue', alpha=0.7)
            ax1.set_title('MAPE por Ativo (Menor = Melhor)')
            ax1.set_ylabel('MAPE (%)')
            ax1.grid(True, alpha=0.3)
            
            # Adicionar valores nas barras
            for bar, value in zip(bars1, mape_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{value:.1f}%', ha='center', va='bottom')
            
            # RMSE
            bars2 = ax2.bar(tickers, rmse_values, color='lightcoral', alpha=0.7)
            ax2.set_title('RMSE por Ativo (Menor = Melhor)')
            ax2.set_ylabel('RMSE (R$)')
            ax2.grid(True, alpha=0.3)
            
            # Adicionar valores nas barras
            for bar, value in zip(bars2, rmse_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{value:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Salvar e mostrar
            filename = f'outputs/phase3_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig)
            
            print(f"📊 Gráfico de métricas salvo: {filename}")
            
        except Exception as e:
            print(f"❌ Erro ao plotar métricas: {e}")

    def _generate_integrated_report(self, data: Dict, **kwargs):
        """Gera relatório integrado com todas as fases"""
        print(f"🔍 DEBUG: Gerando relatório integrado com dados: {list(data.keys())}")
        
        # Fase 1: Análise exploratória
        if 'stock_data' in data:
            print("📊 Gerando relatório Fase 1...")
            self._generate_phase1_report(data, **kwargs)
        
        # Fase 2: Simulação de investimentos
        if 'juros_fixos' in data and 'carteira_acoes' in data:
            print("💰 Gerando relatório Fase 2...")
            self._generate_phase2_report(data, **kwargs)
        
        # Fase 3: Previsões (se disponível)
        if 'forecasts' in data:
            print("🔮 Gerando relatório Fase 3...")
            self._generate_phase3_report(data)
        
        # Relatório executivo integrado
        self._generate_executive_summary(data, **kwargs)

    def _generate_executive_summary(self, data: Dict, **kwargs):
        """Gera resumo executivo integrado"""
        print("\n" + "="*80)
        print("📋 RELATÓRIO EXECUTIVO INTEGRADO")
        print("="*80)
        
        if 'stock_data' in data:
            print(f"📊 FASE 1: Análise de {len(data['stock_data'])} ativos")
            print(f"   - Período: {data.get('start_date', 'N/A')} a {data.get('end_date', 'N/A')}")
        
        if 'juros_fixos' in data and 'carteira_acoes' in data:
            print(f"💰 FASE 2: Simulação de investimentos")
            print(f"   - Capital inicial: R$ {data.get('capital_inicial', 'N/A'):,.2f}")
            print(f"   - Aporte mensal: R$ {data.get('aporte_mensal', 'N/A'):,.2f}")
        
        if 'forecasts' in data:
            print(f"🔮 FASE 3: Previsões futuras")
            print(f"   - Horizonte: {data.get('forecast_horizon', 'N/A')} dias")
        
        print("="*80)

class ReportFactory:
    #Factory para criar geradores de relatórios
    @staticmethod
    def create_generator(report_type: str = 'simple', **kwargs) -> 'SimpleReportGenerator':
        if report_type.lower() == 'simple':
            return SimpleReportGenerator(**kwargs)
        else:
            raise ValueError(f"Tipo de relatório não suportado: {report_type}")

def generate_comparison_table(df_juros, df_carteira, stock_data=None):
    # Gera tabela comparativa profissional entre cenários
    try:
        # Debug: verificar estrutura dos DataFrames
        print(f"🔍 DEBUG: df_juros colunas: {list(df_juros.columns)}")
        print(f"🔍 DEBUG: df_carteira colunas: {list(df_carteira.columns)}")
        print(f"🔍 DEBUG: df_juros shape: {df_juros.shape}")
        print(f"🔍 DEBUG: df_carteira shape: {df_carteira.shape}")
        
        # Importar o calculador de métricas
        from modules.metrics import create_default_metrics_calculator
        metrics_calc = create_default_metrics_calculator()
        
        # Calcular métricas para Juros Compostos
        juros_metrics = metrics_calc.calculate_metrics(df_juros)
        print(f"🔍 DEBUG: juros_metrics: {juros_metrics}")
        
        # Calcular métricas para Carteira de Ações
        carteira_metrics = metrics_calc.calculate_metrics(df_carteira)
        print(f"🔍 DEBUG: carteira_metrics: {carteira_metrics}")
        
        # Debug: verificar se as métricas têm valores válidos
        print(f"🔍 DEBUG: juros_metrics keys: {list(juros_metrics.keys())}")
        print(f"🔍 DEBUG: carteira_metrics keys: {list(carteira_metrics.keys())}")
        
        # Criar tabela comparativa
        comparison_data = {
            'Métrica': [
                'Valor Final (R$)',
                'CAGR (%)',
                'Volatilidade Anualizada (%)',
                'Máximo Drawdown (%)',
                'Sharpe Ratio',
                'Retorno Total (%)'
            ],
            'Juros Compostos': [
                f"R$ {juros_metrics.get('Final_Capital', 0):,.2f}",
                f"{juros_metrics.get('CAGR', 0)*100:.2f}%",
                f"{juros_metrics.get('Annual_Volatility', 0):.2f}%",
                f"{juros_metrics.get('Max_Drawdown', 0):.2f}%",
                f"{juros_metrics.get('Sharpe_Ratio', 0):.3f}",
                f"{juros_metrics.get('Total_Return', 0):.2f}%"
            ],
            'Carteira de Ações': [
                f"R$ {carteira_metrics.get('Final_Capital', 0):,.2f}",
                f"{carteira_metrics.get('CAGR', 0)*100:.2f}%",
                f"{carteira_metrics.get('Annual_Volatility', 0):.2f}%",
                f"{carteira_metrics.get('Max_Drawdown', 0):.2f}%",
                f"{carteira_metrics.get('Sharpe_Ratio', 0):.3f}",
                f"{carteira_metrics.get('Total_Return', 0):.2f}%"
            ]
        }
        
        # Criar DataFrame da tabela
        comparison_df = pd.DataFrame(comparison_data)
        print(f"🔍 DEBUG: Tabela criada com {len(comparison_df)} linhas")
        print(f"🔍 DEBUG: Colunas da tabela: {list(comparison_df.columns)}")
        
        # Calcular diferenças (extrair valores numéricos)
        def extract_numeric(value_str):
            # Extrai valor numérico de string formatada
            try:
                # Remove R$, % e converte para float
                clean_value = value_str.replace('R$ ', '').replace('%', '').replace(',', '')
                return float(clean_value)
            except:
                return 0.0
        
        # Calcular diferenças (extrair valores numéricos)
        juros_values = [extract_numeric(val) for val in comparison_df['Juros Compostos']]
        carteira_values = [extract_numeric(val) for val in comparison_df['Carteira de Ações']]
        differences = [c - j for c, j in zip(carteira_values, juros_values)]
        
        # Formatar diferenças baseado no tipo de métrica
        formatted_differences = []
        for i, (metric, diff) in enumerate(zip(comparison_df['Métrica'], differences)):
            if 'R$' in comparison_df['Juros Compostos'].iloc[i]:
                # Para valores monetários
                formatted_differences.append(f"+{diff:,.2f}" if diff > 0 else f"{diff:,.2f}")
            elif '%' in comparison_df['Juros Compostos'].iloc[i]:
                # Para percentuais
                formatted_differences.append(f"+{diff:.2f}%" if diff > 0 else f"{diff:.2f}%")
            else:
                # Para outros valores
                formatted_differences.append(f"+{diff:.3f}" if diff > 0 else f"{diff:.3f}")
        
        comparison_df['Diferença'] = formatted_differences
        print(f"🔍 DEBUG: Tabela final com diferenças calculadas")
        print(f"🔍 DEBUG: Primeiras linhas da tabela:")
        print(comparison_df.head())
        
        return comparison_df
        
    except Exception as e:
        logger.error(f"Erro ao gerar tabela comparativa: {e}")
        return pd.DataFrame()

def print_comparison_table(comparison_df):
    #Imprime tabela comparativa formatada
    if comparison_df.empty:
        print("❌ Erro ao gerar tabela comparativa")
        return
    
    print("\n" + "="*100)
    print("📊 TABELA COMPARATIVA: JUROS COMPOSTOS vs CARTEIRA DE AÇÕES")
    print("="*100)
    
    # Imprimir tabela formatada
    for idx, row in comparison_df.iterrows():
        print(f"{row['Métrica']:<30} | {row['Juros Compostos']:<20} | {row['Carteira de Ações']:<20} | {row['Diferença']:<15}")
        if idx == 0:  # Separador após cabeçalho
            print("-"*100)
    
    print("="*100)
    
    # Resumo executivo
    print("\n🏆 RESUMO EXECUTIVO:")
    
    # Extrair valor final para comparação
    try:
        juros_final_str = comparison_df.iloc[0]['Juros Compostos']
        carteira_final_str = comparison_df.iloc[0]['Carteira de Ações']
        
        juros_final = float(juros_final_str.replace('R$ ', '').replace(',', ''))
        carteira_final = float(carteira_final_str.replace('R$ ', '').replace(',', ''))
        
        if carteira_final > juros_final:
            diferenca = carteira_final - juros_final
            print(f"✅ CARTEIRA SUPERIOR em R$ {diferenca:,.2f}")
            print(f"📈 Vantagem: {(diferenca/juros_final)*100:.1f}% sobre juros fixos")
        else:
            diferenca = juros_final - carteira_final
            print(f"⚠️ JUROS FIXOS SUPERIOR em R$ {diferenca:,.2f}")
            print(f"📉 Desvantagem: {(diferenca/carteira_final)*100:.1f}% sobre carteira")
    except Exception as e:
        print(f"⚠️ Não foi possível calcular resumo executivo: {e}")
