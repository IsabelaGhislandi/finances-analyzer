from modules import InvestmentAnalysisFactory
from modules.data import parse_tickers, parse_weights
from modules.interests import InvestmentSimulator
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
    parser.add_argument('--taxa-juros', type=float, default=0.01, help='Monthly interest rate default 1 percent')
    parser.add_argument('--retorno-fixo', type=float, default=0.01, help='Fixed monthly return rate default 1 percent')
    
    # Argumentos opcionais para Fase 1
    parser.add_argument('--save-data', action='store_true', help='Save raw data to CSV files')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    parser.add_argument('--quick', action='store_true', help='Quick mode (no plot saving)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output with debug info')
    
    return parser

def validate_arguments(args):
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

def create_system_config(args):
    #Cria configuração do sistema baseada nos argumentos

    config = {
            'data_source': 'yfinance',
            'metrics_type': 'performance',
            'portfolio_type': 'standard',
            'report_type': 'simple',
            'simulator_type': 'compound',
            'data_config': {'cache_enabled': True},  # ✅ Adicionar esta linha
            'report_config': {'include_charts': True}  # ✅ E esta também
    }
    
    # Ajustar configuração baseada nos argumentos
    if args.quick:
        config['report_config']['include_charts'] = False
    
    if args.verbose:
        config['data_config']['cache_enabled'] = False
    
    return config

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
    
    # 2. Processar pesos da carteira
    weights = parse_weights(args.weights, len(tickers))
    
    # 3. Validar tickers
    main_factory = InvestmentAnalysisFactory()
    config = create_system_config(args)
    system = main_factory.create_complete_system(config, tickers, weights)
    
    # Usar componentes do sistema
    collector = system['data_collector']
    report = system['report_generator']
    
    print(f"\n🔍 Validando ativos disponíveis...")
    
    valid_tickers, invalid_tickers = collector.validate_tickers(tickers)
    
    if invalid_tickers:
        print(f"Ativos não encontrados: {', '.join(invalid_tickers)}")
    
    if not valid_tickers:
        print("Nenhum ativo válido encontrado!")
        return False
    
    print(f"✅ Ativos validados: {', '.join(valid_tickers)}")
    
    # 4. Coletar dados
    print(f"\n📈 Coletando dados históricos...")
    data_dict = collector.get_multiple_stocks(valid_tickers, args.start_date, args.end_date)
    
    if not data_dict:
        print("Falha na coleta de dados!")
        return False
    
    print(f"✅ Dados coletados para {len(data_dict)} ativo(s)")
    
    # 5. Resumo dos dados
    print(f"\n📋 RESUMO DOS DADOS:")
    print("-" * 60)
    
    for ticker, data in data_dict.items():
        price_start = data['Close'].iloc[0]
        price_end = data['Close'].iloc[-1]
        retorno = (price_end / price_start - 1) * 100
        
        print(f"{ticker:>12} | {len(data):>4} dias | "
              f"R$ {price_start:>8.2f} → R$ {price_end:>8.2f} | "
              f"{retorno:>6.1f}%")
    
    # 6. Exibir composição da carteira
    print(f"\n⚖️  COMPOSIÇÃO DA CARTEIRA:")
    print("-" * 40)
    for ticker, weight in zip(valid_tickers, weights):
        print(f"{ticker:>12}: {weight:>6.1%}")
    
    # 7. Salvar dados se solicitado
    if args.save_data:
        print(f"\n💾 Salvando dados...")
        os.makedirs('data/raw', exist_ok=True)
        
        for ticker, data in data_dict.items():
            filename = f'data/raw/{ticker}_{args.start_date}_{args.end_date}.csv'
            data.to_csv(filename)
            logger.info(f"Dados salvos: {filename}")
        
        print(f"✅ Dados salvos em data/raw/")
    
    # 8. Gerar relatórios visuais
    if not args.no_plots:
        print(f"\n📊 Gerando análises visuais...")
        
        report = system['report_generator']
        save_plots = not args.quick
        
        # Gerar relatório da Fase 1 usando o método correto
        report.generate_report({
            'stock_data': data_dict,
            'tickers': valid_tickers,
            'weights': weights
        }, report_type='phase1')
        
        print("✅ Análises visuais geradas com sucesso!")
    
    # 9. Informações para debug
    if args.verbose:
        print(f"\n🔧 DEBUG INFO:")
        print(f"  - Cache ativo: {len(collector.cache)} entradas")
        print(f"  - Memória de dados: {sum(data.memory_usage().sum() for data in data_dict.values()) / 1024 / 1024:.1f} MB")
    
    print("="*80)
    logger.info("Análise da Fase 1 concluída com sucesso")
    
    return True

def execute_analysis_phase2(args, logger):
    #Executa Fase 2: Simulação de investimentos COMPLETA"""
    logger.info("=== FASE 2: Simulação de investimentos COMPLETA ===")
    # Criar sistema e executar tudo em uma linha
    system = InvestmentAnalysisFactory().create_complete_system(create_system_config(args), args.tickers, args.weights)
    # Executar simulações e gerar relatório
    run_phase2(system, args)
    logger.info("✅ Fase 2 concluída!")

def run_phase2(system, args):
    stock_data = system['data_collector'].get_multiple_stocks(args.tickers, args.start_date, args.end_date)
    print(f"\n📊 Dados coletados: {'✅' if stock_data else '(usando simulação)'}")

    # Criar simuladores específicos para cada cenário
    simulator_factory = system['investment_simulator']
    
    # Simular juros fixos
    juros_simulator = simulator_factory.create_simulator(
        'compound_interest',
        initial_capital=args.capital_inicial,
        monthly_rate=args.taxa_juros * 100  # Converter para percentual
    )
    df_juros = juros_simulator.simulate(
        monthly_contribution=args.aporte_mensal,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Simular carteira de ações
    carteira_simulator = simulator_factory.create_simulator(
        'stock_portfolio',
        tickers=args.tickers,
        weights=args.weights,
        initial_capital=args.capital_inicial,
        monthly_return_rate=args.retorno_fixo
    )
    df_carteira = carteira_simulator.simulate(
        monthly_contribution=args.aporte_mensal,
        start_date=args.start_date,
        end_date=args.end_date,
        stock_data=stock_data
    )
    
    # Criar instância do InvestmentSimulator para comparar cenários
    main_simulator = InvestmentSimulator(
        initial_capital=args.capital_inicial,
        monthly_contribution=args.aporte_mensal,
        monthly_interest_rate=args.taxa_juros * 100
    )
    comparacao = main_simulator.compare_scenarios(df_juros, df_carteira)

    metrics_calc = system['metrics_calculator']
    metricas_juros = metrics_calc.calculate_metrics(df_juros)
    metricas_carteira = metrics_calc.calculate_metrics(df_carteira)

     # 4. Análise de portfólio
    portfolio_mgr = system.get('portfolio_analyzer')
    portfolio_analysis = portfolio_mgr.analyze_portfolio()
    rebalanceamento = portfolio_mgr.suggest_rebalancing([1/len(args.tickers)] * len(args.tickers))

    generate_report(args, comparacao, metricas_juros, metricas_carteira, portfolio_analysis, rebalanceamento, system, df_juros, df_carteira)

def generate_report(args, comparacao, metricas_juros, metricas_carteira, portfolio_analysis, rebalanceamento, system, df_juros, df_carteira):
    print("\n" + "="*80)
    print("�� RELATÓRIO EXECUTIVO - FASE 2")
    print("="*80)
    
    if system and not args.no_plots:
        print(f"\n📊 Gerando gráficos...")
        try:
            report_gen = system['report_generator']
            report_gen.generate_report({
                'juros_fixos': df_juros,
                'carteira_acoes': df_carteira,
                'comparacao': comparacao,
                'metricas_juros': metricas_juros,
                'metricas_carteira': metricas_carteira,
                'portfolio_analysis': portfolio_analysis,
                'rebalanceamento': rebalanceamento
            }, report_type='phase2')
            print("✅ Gráficos gerados com sucesso!")
        except Exception as e:
            print(f"⚠️ Gráficos não puderam ser gerados: {e}")
    # Parâmetros em uma linha
    print(f"💰 Capital: R$ {args.capital_inicial:,.2f} | Aporte: R$ {args.aporte_mensal:,.2f} | Período: {args.start_date} → {args.end_date}")
    print(f"�� Ativos: {', '.join(args.tickers)} | Pesos: {', '.join([f'{w:.1%}' for w in args.weights])}")
    
    # Resultado final comparativo
    if not comparacao.empty:
        ultima = comparacao.iloc[-1]
        juros_final = ultima['Fixed_Interest_Capital']
        carteira_final = ultima['Stock_Portfolio_Capital']
        diferenca = ultima['Capital_Difference']
        
        print(f"\n🏆 RESULTADO FINAL:")
        print(f"  �� Juros Fixos: R$ {juros_final:>12,.0f}")
        print(f"  �� Carteira:    R$ {carteira_final:>12,.0f}")
        print(f"  �� Diferença:   R$ {diferenca:>12,.0f}")
        
        # Vantagem em uma linha
        if diferenca > 0:
            vantagem = (diferenca / juros_final) * 100
            print(f"  🏆 Carteira venceu por {vantagem:.1f}%")
        else:
            vantagem = (abs(diferenca) / carteira_final) * 100
            print(f"  �� Juros fixos venceram por {vantagem:.1f}%")
    
    # Métricas resumidas
    print(f"\n�� MÉTRICAS RESUMIDAS:")
    print(f"  �� Juros Fixos - CAGR: {metricas_juros.get('CAGR', 0):.1%} | Vol: {metricas_juros.get('Annual_Volatility', 0):.1%}")
    print(f"  �� Carteira - CAGR: {metricas_carteira.get('CAGR', 0):.1%} | Vol: {metricas_carteira.get('Annual_Volatility', 0):.1%}")
    
    # Portfólio em uma linha
    print(f"\n⚖️ PORTFÓLIO - Score: {portfolio_analysis['summary']['diversification_score']}/100 | Recomendação: {portfolio_analysis['summary']['recommendation']}")
    
    # Rebalanceamento compacto
    if rebalanceamento:
        print(f"�� Rebalanceamento: {', '.join([f'{t}:{info['Action'][:3]}' for t, info in rebalanceamento.items()])}")
    
    print("="*80)

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