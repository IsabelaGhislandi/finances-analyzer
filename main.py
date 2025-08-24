from modules.report import ReportFactory
from modules.data import parse_tickers, parse_weights
from modules.data import DataCollectorFactory
from modules.interests import InvestmentSimulator
from modules.metrics import MetricsCalculatorFactory
from modules.portfolio import PortfolioFactory
import argparse
import sys
import os
from datetime import datetime
import logging
import pandas as pd 

# Adicionar o diretório atual ao path para imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_argument_parser():
    #Cria o parser de argumentos da linha de comando
    parser = argparse.ArgumentParser(description='Investment Analysis Tool')

    parser.add_argument('--tickers', nargs='+', required=True, help='Stock tickers to analyze')
    parser.add_argument('--weights', nargs='+', type=float, help='Portfolio weights')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')

    parser.add_argument('--aporte-mensal', type=float, help='Monthly contribution amount')
    parser.add_argument('--capital-inicial', type=float, help='Initial capital amount')
    parser.add_argument('--taxa-juros', type=float, default=0.01, help='Monthly interest rate (default: 1%)')
    parser.add_argument('--retorno-fixo', type=float, default=0.01, help='Fixed monthly return rate for portfolio (default: 1%)')
    
    # Argumentos opcionais para Fase 1
    parser.add_argument('--save-data', action='store_true', help='Save raw data to CSV files')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    parser.add_argument('--quick', action='store_true', help='Quick mode (no plot saving)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output with debug info')
    
    return parser

def validate_arguments(args):
    """Valida e ajusta argumentos"""
    # Validar datas
    try:
        datetime.strptime(args.start_date, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Data inicial inválida: {args.start_date}. Use formato YYYY-MM-DD")
    
    # Data final default é hoje
    if args.end_date is None:
        args.end_date = datetime.now().strftime('%Y-%m-%d')
    else:
        try:
            datetime.strptime(args.end_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Data final inválida: {args.end_date}. Use formato YYYY-MM-DD")
    
    # Validar período
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    if start_date >= end_date:
        raise ValueError("Data inicial deve ser anterior à data final")
    
    # Validar valores monetários (só se estiver na Fase 2)
    if hasattr(args, 'aporte_mensal') and args.aporte_mensal is not None:
        if args.aporte_mensal < 0:
            raise ValueError("Aporte mensal deve ser positivo")
    
    if hasattr(args, 'capital_inicial') and args.capital_inicial is not None:
        if args.capital_inicial < 0:
            raise ValueError("Capital inicial deve ser positivo")
    
    if hasattr(args, 'taxa_juros') and args.taxa_juros is not None:
        if args.taxa_juros < 0:
            raise ValueError("Taxa de juros deve ser positiva")
    
    return args

def execute_analysis_phase_1(args, logger):
    #Executa análise da Fase 1
    print("\n" + "="*80)
    print("📊 INVESTMENT SIMULATOR - FASE 1: ANÁLISE EXPLORATÓRIA")
    print("="*80)
    
    # 1. Processar tickers
    tickers = parse_tickers(args.tickers)
    logger.info(f"Tickers solicitados: {tickers}")
    print(f"🎯 Ativos para análise: {', '.join(tickers)}")
    print(f"📅 Período: {args.start_date} até {args.end_date}")
    
    # 2. Validar tickers
    collector = DataCollectorFactory.create_collector('yfinance')
    print(f"\n🔍 Validando ativos disponíveis...")
    
    valid_tickers, invalid_tickers = collector.validate_tickers(tickers)
    
    if invalid_tickers:
        print(f"Ativos não encontrados: {', '.join(invalid_tickers)}")
    
    if not valid_tickers:
        print("Nenhum ativo válido encontrado!")
        return False
    
    print(f"✅ Ativos validados: {', '.join(valid_tickers)}")
    
    # 3. Coletar dados
    print(f"\n📈 Coletando dados históricos...")
    data_dict = collector.get_multiple_stocks(valid_tickers, args.start_date, args.end_date)
    
    if not data_dict:
        print("Falha na coleta de dados!")
        return False
    
    print(f"✅ Dados coletados para {len(data_dict)} ativo(s)")
    
    # 4. Resumo dos dados
    print(f"\n📋 RESUMO DOS DADOS:")
    print("-" * 60)
    
    for ticker, data in data_dict.items():
        price_start = data['Close'].iloc[0]
        price_end = data['Close'].iloc[-1]
        retorno = (price_end / price_start - 1) * 100
        
        print(f"{ticker:>12} | {len(data):>4} dias | "
              f"R$ {price_start:>8.2f} → R$ {price_end:>8.2f} | "
              f"{retorno:>6.1f}%")
    
    # 5. Processaar pesos da carteira
    weights = parse_weights(args.weights, len(valid_tickers))
    
    print(f"\n⚖️  COMPOSIÇÃO DA CARTEIRA:")
    print("-" * 40)
    for ticker, weight in zip(valid_tickers, weights):
        print(f"{ticker:>12}: {weight:>6.1%}")
    
    # 6. Salvar dados se solicitado
    if args.save_data:
        print(f"\n💾 Salvando dados...")
        os.makedirs('data/raw', exist_ok=True)
        
        for ticker, data in data_dict.items():
            filename = f'data/raw/{ticker}_{args.start_date}_{args.end_date}.csv'
            data.to_csv(filename)
            logger.info(f"Dados salvos: {filename}")
        
        print(f"✅ Dados salvos em data/raw/")
    
    # 7. Gerar relatórios visuais
    if not args.no_plots:
        print(f"\n📊 Gerando análises visuais...")
        
        report = ReportFactory.create_default_report()
        save_plots = not args.quick
        
        # Análise individual da primeira ação (ou única)
        first_ticker = valid_tickers[0]
        first_data = data_dict[first_ticker]
        report.plot_single_stock(first_data, first_ticker, save=save_plots)
        
        # Se múltiplas ações, fazer comparação
        if len(data_dict) > 1:
            report.plot_comparison(data_dict, save=save_plots)
            report.plot_correlation_heatmap(data_dict, save=save_plots)
        
        # Estatísticas detalhadas
        stats_df = report.generate_summary_stats(data_dict)
    
    # Informações para debug
    if args.verbose:
        print(f"\n🔧 DEBUG INFO:")
        print(f"  - Cache ativo: {len(collector.cache)} entradas")
        print(f"  - Memória de dados: {sum(data.memory_usage().sum() for data in data_dict.values()) / 1024 / 1024:.1f} MB")
    
    print("="*80)
    logger.info("Análise da Fase 1 concluída com sucesso")
    
    return True

def execute_analysis_phase2(args, logger):
    """Executa Fase 2: Simulação de investimentos COMPLETA"""
    logger.info("=== FASE 2: Simulação de investimentos COMPLETA ===")
    
    # 1. COLETAR DADOS REAIS (Yahoo Finance)
    print("\n COLETANDO DADOS REAIS DAS AÇÕES:")
    print("-" * 50)
    
    collector = DataCollectorFactory.create_collector('yfinance')
    stock_data = collector.get_multiple_stocks(args.tickers, args.start_date, args.end_date)
    
    if not stock_data:
        print("❌ Falha na coleta de dados! Usando simulação com retorno fixo.")
        logger.warning("Falha na coleta de dados - usando retorno fixo")
        stock_data = {}  # Vazio para usar retorno fixo
    else:
        print(f"✅ Dados coletados para {len(stock_data)} ativo(s)")
        
        # Mostrar resumo dos dados
        for ticker, data in stock_data.items():
            price_start = data['Close'].iloc[0]
            price_end = data['Close'].iloc[-1]
            retorno = (price_end / price_start - 1) * 100
            print(f"  {ticker}: R$ {price_start:.2f} → R$ {price_end:.2f} ({retorno:.1f}%)")
    
    # 2. Criar simulador
    simulador = InvestmentSimulator(
        initial_capital=args.capital_inicial,
        monthly_contribution=args.aporte_mensal,
        monthly_interest_rate=args.taxa_juros
    )
    
    # 3. Simular cenários
    print("\n SIMULANDO CENÁRIOS:")
    print("-" * 50)
    
    df_juros = simulador.simulate_fixed_interest_scenario(
        start_date=args.start_date,
        end_date=args.end_date
    )
    print("✅ Simulação de juros fixos concluída")
    
    # ✅ CORREÇÃO: Passar stock_data como primeiro parâmetro
    df_carteira = simulador.simulate_stock_portfolio_scenario(
        stock_data=stock_data,  # ✅ CORRETO! Primeiro parâmetro
        weights=args.weights,
        start_date=args.start_date,
        end_date=args.end_date,
        monthly_return_rate=args.retorno_fixo
    )
    
    # Verificar tipo de simulação
    if not stock_data:
        print("✅ Simulação de carteira com retorno fixo concluída")
    else:
        print("✅ Simulação de carteira com dados reais concluída")
    
    # 4. Comparar e gerar relatório
    comparacao = simulador.compare_scenarios(df_juros, df_carteira)
    metrics_calculator = MetricsCalculatorFactory.create_calculator('performance')
    metricas_juros = metrics_calculator.calculate_metrics(df_juros)
    metricas_carteira = metrics_calculator.calculate_metrics(df_carteira)
    
    # 5. Análise de portfólio
    portfolio_manager = PortfolioFactory.create_portfolio_manager(args.tickers, args.weights)
    portfolio_analysis = portfolio_manager.analyze_portfolio()
    pesos_iguais = [1/len(args.tickers)] * len(args.tickers)
    rebalanceamento = portfolio_manager.suggest_rebalancing(pesos_iguais)

    print("\n" + "="*80)
    print("🏆 INVESTMENT SIMULATOR - FASE 2: RELATÓRIO EXECUTIVO")
    print("="*80)
    
    # A) PARÂMETROS DA SIMULAÇÃO
    print("\n📋 PARÂMETROS DA SIMULAÇÃO:")
    print("-" * 50)
    print(f"�� Capital Inicial: R$ {args.capital_inicial:,.2f}")
    print(f"📈 Aporte Mensal: R$ {args.aporte_mensal:,.2f}")
    print(f"📅 Período: {args.start_date} até {args.end_date}")
    print(f"�� Ativos: {', '.join(args.tickers)}")
    print(f"⚖️  Pesos: {', '.join([f'{w:.1%}' for w in args.weights])}")
    
    # B) RESULTADO FINAL COMPARATIVO
    if not comparacao.empty:
        ultima_linha = comparacao.iloc[-1]
        print("\n🏆 RESULTADO FINAL COMPARATIVO:")
        print("-" * 50)
        print(f"💰 CAPITAL FINAL:")
        print(f"  �� Juros Fixos:     R$ {ultima_linha['Fixed_Interest_Capital']:>12,.2f}")
        print(f"  �� Carteira Ações:  R$ {ultima_linha['Stock_Portfolio_Capital']:>12,.2f}")
        print(f"  �� Diferença:       R$ {ultima_linha['Capital_Difference']:>12,.2f}")
        
        # Calcular vantagem
        if ultima_linha['Capital_Difference'] > 0:
            vantagem_percentual = (ultima_linha['Capital_Difference'] / ultima_linha['Fixed_Interest_Capital']) * 100
            print(f"\n🏆 Carteira de Ações venceu por {vantagem_percentual:.1f}%")
        else:
            vantagem_percentual = (abs(ultima_linha['Capital_Difference']) / ultima_linha['Stock_Portfolio_Capital']) * 100
            print(f"\n Juros Fixos venceram por {vantagem_percentual:.1f}%")
    
    # C) MÉTRICAS DETALHADAS
    print("\n📊 MÉTRICAS DETALHADAS:")
    print("-" * 50)
    
    # Métricas do Juros Fixos
    print("�� JUROS FIXOS:")
    if 'Final_Capital' in metricas_juros:
        print(f"  Capital Final: R$ {metricas_juros['Final_Capital']:>12,.2f}")
    if 'Total_Return' in metricas_juros:
        print(f"  Retorno Total: {metricas_juros['Total_Return']:>12,.2f}%")
    if 'CAGR' in metricas_juros:
        print(f"  CAGR: {metricas_juros['CAGR']*100:>12,.2f}%")
    if 'Annual_Volatility' in metricas_juros:
        print(f"  Volatilidade: {metricas_juros['Annual_Volatility']:>12,.2f}%")
    
    # Métricas da Carteira
    print("\n🟥 CARTEIRA DE AÇÕES:")
    if 'Final_Capital' in metricas_carteira:
        print(f"  Capital Final: R$ {metricas_carteira['Final_Capital']:>12,.2f}")
    if 'Total_Return' in metricas_carteira:
        print(f"  Retorno Total: {metricas_carteira['Total_Return']:>12,.2f}%")
    if 'CAGR' in metricas_carteira:
        print(f"  CAGR: {metricas_carteira['CAGR']*100:>12,.2f}%")
    if 'Annual_Volatility' in metricas_carteira:
        print(f"  Volatilidade: {metricas_carteira['Annual_Volatility']:>12,.2f}%")
    if 'Max_Drawdown' in metricas_carteira:
        print(f"  Max Drawdown: {metricas_carteira['Max_Drawdown']:>12,.2f}%")
    if 'Sharpe_Ratio' in metricas_carteira:
        print(f"  Sharpe Ratio: {metricas_carteira['Sharpe_Ratio']:>12,.3f}")
    
    # D) ANÁLISE DO PORTFÓLIO
    print("\n�� ANÁLISE DO PORTFÓLIO:")
    print("-" * 50)
    print(f"�� Total de Ativos: {portfolio_analysis['summary']['total_assets']}")
    print(f"📊 Maior Posição: {portfolio_analysis['summary']['largest_position']['ticker']} ({portfolio_analysis['summary']['largest_position']['weight']:.1f}%)")
    print(f"📊 Menor Posição: {portfolio_analysis['summary']['smallest_position']['ticker']} ({portfolio_analysis['summary']['smallest_position']['weight']:.1f}%)")
    print(f"⚠️  Nível de Concentração: {portfolio_analysis['summary']['concentration_level']}")
    print(f"🎯 Score de Diversificação: {portfolio_analysis['summary']['diversification_score']}/100")
    print(f"💡 Recomendação: {portfolio_analysis['summary']['recommendation']}")
    
    # E) SUGESTÕES DE REBALANCEAMENTO
    print("\n⚖️ SUGESTÕES DE REBALANCEAMENTO (pesos iguais):")
    print("-" * 50)
    for ticker, info in rebalanceamento.items():
        action_emoji = "��" if info['Action'] == 'Comprar' else "��" if info['Action'] == 'Vender' else "��"
        print(f"{action_emoji} {ticker:>8}: {info['Action']:>6} {abs(info['Weight_Change']):>6.1f}%")
    
    # F) COMPARAÇÃO DETALHADA
    print("\n" + metrics_calculator.compare_metrics(
        metricas_juros, metricas_carteira, 
        "Juros Fixos", "Carteira de Ações"
    ))
    
    # G) EVOLUÇÃO COMPARATIVA (TABELA)
    if not comparacao.empty:
        print("\n📊 EVOLUÇÃO COMPARATIVA:")
        print("-" * 80)
        print(f"{'Data':<12} {'Juros Fixos':<15} {'Carteira Ações':<15} {'Diferença':<15}")
        print("-" * 80)
        
        # Mostrar algumas linhas representativas
        for i in range(0, len(comparacao), max(1, len(comparacao)//6)):  
            linha = comparacao.iloc[i]
            if pd.notna(linha['Date']):  
                date_str = linha['Date'].strftime('%Y-%m')
            else:
                date_str = "Data Inválida"

            print(f"{date_str:<12} "
                f"R$ {linha['Fixed_Interest_Capital']:<13,.0f} "
                f"R$ {linha['Stock_Portfolio_Capital']:<13,.0f} "
                f"R$ {linha['Capital_Difference']:<13,.0f}")
    
    # H) ANÁLISE DE CRUZAMENTOS
    if not comparacao.empty:
        comparacao['Cruzamento'] = comparacao['Capital_Difference'].abs() < 1000
        cruzamentos = comparacao[comparacao['Cruzamento']]
        
        if not cruzamentos.empty:
            print("\n🔄 PONTOS DE CRUZAMENTO:")
            print("-" * 50)
            for _, linha in cruzamentos.iterrows():
                print(f"�� {linha['Date'].strftime('%Y-%m')} - Capital: R$ {linha['Fixed_Interest_Capital']:,.2f}")
        else:
            print("\n🔄 Nenhum ponto de cruzamento encontrado no período")
    
    print("\n" + "="*80)
    print("✅ RELATÓRIO DA FASE 2 CONCLUÍDO!")
    print("="*80)

   # report = ReportFactory.create_report('stock')
   # report.plot_comparison_scenarios(df_juros, df_carteira)
   # report.plot_aportes_analysis(df_juros, df_carteira)
   # report.print_metrics_summary(metricas_juros, metricas_carteira)
    
    logger.info("✅ Fase 2 concluída!")

def execute_analysis(args, logger):
    #Dispatcher: escolhe entre Fase 1 ou Fase 2 baseado nos argumentos"""
    if args.aporte_mensal and args.capital_inicial:
        # Tem argumentos de simulação = Fase 2
        execute_analysis_phase2(args, logger)
    else:
        # Só análise básica = Fase 1
        execute_analysis_phase_1(args, logger)

def main():
    #Função principal
    # Parser de argumentos
    parser = create_argument_parser()
    try:
        args = parser.parse_args()
        args = validate_arguments(args)
    except ValueError as e:
        print(f"Erro nos argumentos: {e}")
        parser.print_help()
        sys.exit(1)
    except SystemExit:
        # argparse chamou sys.exit (--help, argumentos inválidos)
        sys.exit(1)
    
    logger.info("🚀 Investment Simulator iniciado")
    logger.info(f"Argumentos: {vars(args)}")
    
    try:
        execute_analysis(args, logger)  
    except KeyboardInterrupt:
        print(f"\n\n Execução interrompida pelo usuário")
        logger.info("Execução interrompida pelo usuário")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n Erro inesperado: {e}")
        logger.exception("Erro inesperado durante a execução")
        sys.exit(1)
if __name__ == "__main__":
    main()