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

def create_argument_parser() -> argparse.ArgumentParser:
    """Cria o parser de argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description='Investment Analysis Tool - Sistema Integrado')

    # Argumentos obrigatórios
    parser.add_argument('--tickers', nargs='+', required=True, help='Stock tickers to analyze')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    
    # Argumentos opcionais
    parser.add_argument('--weights', nargs='+', type=float, help='Portfolio weights')
    parser.add_argument('--aporte_mensal', type=float, help='Monthly contribution amount')
    parser.add_argument('--capital_inicial', type=float, help='Initial capital amount')
    parser.add_argument('--taxa_juros_mensal', type=float, default=0.01, help='Monthly interest rate (default: 1%)')
    parser.add_argument('--forecast_horizon', type=int, default=30, help='Days to forecast (default: 30)')
    
    # Modos de execução
    parser.add_argument('--mode', choices=['explore', 'simulate', 'forecast', 'integrated'], 
                       default='integrated', help='Execution mode (default: integrated)')
    
    # Argumentos opcionais
    parser.add_argument('--save-data', action='store_true', help='Save data to CSV files')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    parser.add_argument('--quick', action='store_true', help='Quick mode (no plot saving)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    return parser

def validate_arguments(args: argparse.Namespace) -> argparse.Namespace:
    """Valida os argumentos da linha de comando."""
    try:
        datetime.strptime(args.start, '%Y-%m-%d')
        datetime.strptime(args.end, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Datas devem estar no formato YYYY-MM-DD")
    
    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')
    
    if start_date >= end_date:
        raise ValueError("Data inicial deve ser anterior à data final")
    
    # Validar pesos se fornecidos
    if args.weights:
        if len(args.weights) != len(args.tickers):
            raise ValueError("Número de pesos deve ser igual ao número de tickers")
        if abs(sum(args.weights) - 1.0) > 0.01:
            raise ValueError("Pesos devem somar 1.0")
    
    # Validar valores monetários se fornecidos
    if args.aporte_mensal is not None and args.aporte_mensal < 0:
        raise ValueError("Aporte mensal deve ser positivo")
    if args.capital_inicial is not None and args.capital_inicial < 0:
        raise ValueError("Capital inicial deve ser positivo")
    
    return args

def create_system_config(args: argparse.Namespace) -> dict:
    """Cria configuração do sistema."""
    config = {
        'data_source': 'yfinance',
        'metrics_type': 'performance',
        'portfolio_type': 'standard',
        'report_type': 'integrated',
        'simulator_type': 'compound',
        'data_config': {'cache_enabled': True},
        'report_config': {'include_charts': not args.no_plots}
    }
    
    if args.quick:
        config['report_config']['include_charts'] = False
    
    return config

def execute_integrated_analysis(args, logger):
    """Executa análise completa integrada."""
    logger.info("=== EXECUÇÃO INTEGRADA - TODAS AS FASES ===")
    
    print("\n" + "="*80)
    print("🚀 INVESTMENT SIMULATOR - SISTEMA INTEGRADO")
    print("="*80)
    
    # Criar sistema
    system = InvestmentAnalysisFactory().create_complete_system(
        create_system_config(args), 
        args.tickers, 
        args.weights or [1.0/len(args.tickers)] * len(args.tickers)
    )
    
    # FASE 1: Análise Exploratória
    print(f"\n📊 FASE 1: ANÁLISE EXPLORATÓRIA")
    print("-" * 60)
    
    stock_data = system['data_collector'].get_multiple_stocks(args.tickers, args.start, args.end)
    if not stock_data:
        print("❌ Falha na coleta de dados")
        return False
    
    print(f"✅ Dados coletados para {len(stock_data)} ativo(s)")
    
    # Resumo dos dados
    print(f"\n📋 RESUMO DOS DADOS:")
    print("-" * 60)
    for ticker, data in stock_data.items():
        if 'Close' in data.columns and len(data) > 0:
            price_start = data['Close'].iloc[0]
            price_end = data['Close'].iloc[-1]
            retorno = (price_end / price_start - 1) * 100
            print(f"{ticker:>12} | {len(data):>4} dias | "
                  f"R$ {price_start:>8.2f} → R$ {price_end:>8.2f} | "
                  f"{retorno:>6.1f}%")
    
    # FASE 2: Simulação de Investimentos (se parâmetros fornecidos)
    if args.aporte_mensal and args.capital_inicial:
        print(f"\n💰 FASE 2: SIMULAÇÃO DE INVESTIMENTOS")
        print("-" * 60)
        
        simulator_factory = system['investment_simulator']
        
        # Simular juros fixos
        juros_simulator = simulator_factory.create_simulator(
            'compound_interest',
            initial_capital=args.capital_inicial,
            monthly_rate=args.taxa_juros_mensal * 100
        )
        df_juros = juros_simulator.simulate(
            monthly_contribution=args.aporte_mensal,
            start_date=args.start,
            end_date=args.end
        )
        
        # Simular carteira de ações
        carteira_simulator = simulator_factory.create_simulator(
            'stock_portfolio',
            tickers=args.tickers,
            weights=args.weights or [1.0/len(args.tickers)] * len(args.tickers),
            initial_capital=args.capital_inicial,
            monthly_return_rate=0.01
        )
        df_carteira = carteira_simulator.simulate(
            monthly_contribution=args.aporte_mensal,
            start_date=args.start,
            end_date=args.end,
            stock_data=stock_data
        )
        
        # Calcular métricas
        metrics_calc = system['metrics_calculator']
        metricas_juros = metrics_calc.calculate_metrics(df_juros)
        metricas_carteira = metrics_calc.calculate_metrics(df_carteira)
        
        print(f"✅ Simulações concluídas:")
        print(f"   Juros Fixos: R$ {metricas_juros.get('Final_Capital', 0):,.2f}")
        print(f"   Carteira:    R$ {metricas_carteira.get('Final_Capital', 0):,.2f}")
        
        # Gerar tabela comparativa
        print(f"\n📊 Gerando tabela comparativa...")
        try:
            from modules.report import generate_comparison_table, print_comparison_table
            comparison_table = generate_comparison_table(df_juros, df_carteira, None)
            if not comparison_table.empty:
                print_comparison_table(comparison_table)
                
                # Salvar se solicitado
                if args.save_data:
                    os.makedirs('outputs', exist_ok=True)
                    comparison_table.to_csv('outputs/comparison_table.csv', index=False)
                    print("💾 Tabela comparativa salva: outputs/comparison_table.csv")
            else:
                print("⚠️ Não foi possível gerar tabela comparativa")
        except Exception as e:
            print(f"⚠️ Erro ao gerar tabela comparativa: {e}")
    else:
        print(f"\n⚠️ FASE 2: Parâmetros de simulação não fornecidos")
        print("   Use --capital_inicial e --aporte_mensal para simular investimentos")
    
    # FASE 3: Previsões e Backtesting
    if args.forecast_horizon:
        print(f"\n🔮 FASE 3: PREVISÕES E BACKTESTING")
        print("-" * 60)
        
        forecast_manager = system['forecast_manager']
        backtest_manager = system['backtest_manager']
        
        print(f"🔮 PREVISÕES FUTURAS ({args.forecast_horizon} dias):")
        print("-" * 50)
        
        all_forecasts = {}
        all_backtests = {}
        
        for ticker, data in stock_data.items():
            print(f"\n📈 {ticker}:")
            
            if 'Close' in data.columns and len(data) > 0:
                price_series = data['Close']
                
                # Treinar modelo
                if forecast_manager.train(price_series):
                    forecast = forecast_manager.predict(args.forecast_horizon)
                    all_forecasts[ticker] = forecast
                    
                    print(f"  ✅ Modelo treinado: {forecast_manager.get_model_info()}")
                    print(f"  📊 Previsão final: R$ {forecast.iloc[-1]:.2f}")
                    
                    # Executar backtesting
                    backtest_results = backtest_manager.run_backtest(price_series, forecast_manager)
                    if backtest_results:
                        all_backtests[ticker] = backtest_results
                        metrics = backtest_results['metrics']
                        print(f"  🔄 Backtesting: MAPE {metrics['avg_mape']:.1f}%, RMSE {metrics['avg_rmse']:.2f}")
                    else:
                        print(f"  ⚠️ Backtesting não pôde ser executado")
                else:
                    print(f"  ❌ Falha no treinamento do modelo")
            else:
                print(f"  ❌ Dados insuficientes para previsão")
        
        if all_forecasts:
            print(f"\n✅ Previsões concluídas para {len(all_forecasts)} ativo(s)")
        else:
            print("❌ Nenhuma previsão foi gerada")
    else:
        print(f"\n⚠️ FASE 3: Horizonte de previsão não fornecido")
        print("   Use --forecast_horizon para fazer previsões")
    
    # RELATÓRIO FINAL INTEGRADO
    print(f"\n🏆 RELATÓRIO FINAL INTEGRADO")
    print("="*80)
    
    print(f"📊 RESUMO EXECUTIVO:")
    print(f"   Ativos analisados: {', '.join(args.tickers)}")
    print(f"   Período: {args.start} → {args.end}")
    print(f"   Dados coletados: {len(stock_data)} ativo(s)")
    
    if args.aporte_mensal and args.capital_inicial:
        print(f"   Capital inicial: R$ {args.capital_inicial:,.2f}")
        print(f"   Aporte mensal: R$ {args.aporte_mensal:,.2f}")
        print(f"   Taxa de juros: {args.taxa_juros_mensal*100:.1f}% ao mês")
    
    if args.forecast_horizon:
        print(f"   Previsões: {args.forecast_horizon} dias")
        print(f"   Modelos treinados: {len(all_forecasts) if 'all_forecasts' in locals() else 0}")
    
    # Gerar gráficos integrados
    if not args.no_plots and system.get('report_generator'):
        print(f"\n📊 Gerando gráficos integrados...")
        try:
            report_gen = system['report_generator']
            
            # Dados para relatório integrado
            report_data = {
                'stock_data': stock_data,
                'tickers': args.tickers,
                'weights': args.weights or [1.0/len(args.tickers)] * len(args.tickers)
            }
            
            if args.aporte_mensal and args.capital_inicial:
                report_data.update({
                    'juros_fixos': df_juros,
                    'carteira_acoes': df_carteira,
                    'metricas_juros': metricas_juros,
                    'metricas_carteira': metricas_carteira
                })
            
            if args.forecast_horizon and 'all_forecasts' in locals():
                report_data.update({
                    'forecasts': all_forecasts,
                    'backtests': all_backtests
                })
            
            # Gerar relatório integrado
            report_gen.generate_report(report_data, report_type='integrated')
            print("✅ Gráficos integrados gerados com sucesso!")
            
        except Exception as e:
            print(f"⚠️ Gráficos não puderam ser gerados: {e}")
    
    print("="*80)
    logger.info("Execução integrada concluída com sucesso")
    return True

def execute_analysis(args, logger):
    #Dispatcher principal que escolhe o modo de execução.
    
    if args.mode == 'integrated':
        return execute_integrated_analysis(args, logger)
    elif args.mode == 'explore':
        # Só Fase 1
        args.aporte_mensal = None
        args.capital_inicial = None
        args.forecast_horizon = None
        return execute_integrated_analysis(args, logger)
    elif args.mode == 'simulate':
        # Fases 1 e 2
        args.forecast_horizon = None
        return execute_integrated_analysis(args, logger)
    elif args.mode == 'forecast':
        # Fases 1 e 3
        args.aporte_mensal = None
        args.capital_inicial = None
        return execute_integrated_analysis(args, logger)
    else:
        print(f"❌ Modo inválido: {args.mode}")
        return False

def main():
    #Função principal.
    parser = create_argument_parser()
    
    try:
        args = parser.parse_args()
        args = validate_arguments(args)
    except ValueError as e:
        print(f"❌ Erro nos argumentos: {e}")
        parser.print_help()
        sys.exit(1)
    except SystemExit:
        sys.exit(1)
    
    logger.info("🚀 Investment Simulator iniciado")
    logger.info(f"Modo: {args.mode}")
    logger.info(f"Argumentos: {vars(args)}")
    
    try:
        success = execute_analysis(args, logger)
        if success:
            print("\n🎉 Análise concluída com sucesso!")
        else:
            print("\n❌ Análise falhou!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n\n👋 Execução interrompida pelo usuário")
        logger.info("Execução interrompida pelo usuário")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        logger.exception("Erro inesperado durante a execução")
        sys.exit(1)

if __name__ == "__main__":
    main()