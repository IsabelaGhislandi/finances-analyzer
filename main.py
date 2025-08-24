from modules.report import ReportFactory
from modules.data import parse_tickers, parse_weights
from modules.data import DataCollectorFactory
import argparse
import sys
import os
from datetime import datetime
import logging

# Adicionar o diretório atual ao path para imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_argument_parser():
    #Cria o parser de argumentos da linha de comando
    parser = argparse.ArgumentParser(
        description='Investment Simulator - Compara juros fixos vs carteira de ações',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Exemplos de uso:

            Análise básica (Fase 1):
            python main.py --tickers "PETR4.SA,VALE3.SA,ITUB4.SA" --start 2023-01-01 --end 2024-01-01

            Análise com pesos personalizados:
            python main.py --tickers "PETR4.SA,VALE3.SA" --weights "0.6,0.4" --start 2022-01-01 --end 2024-01-01

            Modo de exploração rápida:
            python main.py --tickers "PETR4.SA" --quick
                    """
        )
    
    # Argumentos obrigatórios
    parser.add_argument('--tickers', type=str, required=True,
                       help='Lista de ativos separados por vírgula (ex: "PETR4.SA,VALE3.SA")')
    
    # Argumentos opcionais com defaults
    parser.add_argument('--start', type=str, default='2023-01-01',
                       help='Data inicial (YYYY-MM-DD) - default: 2019-01-01')
    parser.add_argument('--end', type=str, default=None,
                       help='Data final (YYYY-MM-DD) - default: hoje')
    parser.add_argument('--weights', type=str, default=None,
                       help='Pesos da carteira separados por vírgula (ex: "0.4,0.3,0.3")')
    
    # Argumentos para simulação (Fase 2+)
    parser.add_argument('--aporte-mensal', type=float, default=1000,
                       help='Aporte mensal em R$ (default: 1000)')
    parser.add_argument('--capital-inicial', type=float, default=50000,
                       help='Capital inicial em R$ (default: 50000)')
    parser.add_argument('--taxa-juros-mensal', type=float, default=0.8,
                       help='Taxa de juros mensal em %% (default: 0.8)')
    parser.add_argument('--forecast-horizon', type=int, default=30,
                       help='Dias de previsão (default: 30)')
    
    # Argumentos de controle
    parser.add_argument('--quick', action='store_true',
                       help='Modo rápido: análise básica sem salvar gráficos')
    parser.add_argument('--no-plots', action='store_true',
                       help='Não gerar gráficos')
    parser.add_argument('--save-data', action='store_true',
                       help='Salvar dados baixados em CSV')
    parser.add_argument('--verbose', action='store_true',
                       help='Log detalhado')
    
    return parser


def validate_arguments(args):
    #Valida e ajusta argumentos
    # Validar datas
    try:
        datetime.strptime(args.start, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Data inicial inválida: {args.start}. Use formato YYYY-MM-DD")
    
    # Data final default é hoje
    if args.end is None:
        args.end = datetime.now().strftime('%Y-%m-%d')
    else:
        try:
            datetime.strptime(args.end, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Data final inválida: {args.end}. Use formato YYYY-MM-DD")
    
    # Validar período
    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')
    
    if start_date >= end_date:
        raise ValueError("Data inicial deve ser anterior à data final")
    
    # Validar valores monetários
    if args.aporte_mensal < 0:
        raise ValueError("Aporte mensal deve ser positivo")
    
    if args.capital_inicial < 0:
        raise ValueError("Capital inicial deve ser positivo")
    
    if args.taxa_juros_mensal < 0:
        raise ValueError("Taxa de juros deve ser positiva")
    
    return args


def execute_analysis(args, logger):
    #Executa análise da Fase 1
    
    print("\n" + "="*80)
    print("📊 INVESTMENT SIMULATOR - FASE 1: ANÁLISE EXPLORATÓRIA")
    print("="*80)
    
    # 1. Processar tickers
    tickers = parse_tickers(args.tickers)
    logger.info(f"Tickers solicitados: {tickers}")
    print(f"🎯 Ativos para análise: {', '.join(tickers)}")
    print(f"📅 Período: {args.start} até {args.end}")
    
    # 2. Validar tickers
    collector = DataCollectorFactory.create_collector('yfinance')
    print(f"\n🔍 Validando ativos disponíveis...")
    
    valid_tickers, invalid_tickers = collector.validate_tickers(tickers)
    
    if invalid_tickers:
        print(f"⚠️  Ativos não encontrados: {', '.join(invalid_tickers)}")
    
    if not valid_tickers:
        print("❌ Nenhum ativo válido encontrado!")
        return False
    
    print(f"✅ Ativos validados: {', '.join(valid_tickers)}")
    
    # 3. Coletar dados
    print(f"\n📈 Coletando dados históricos...")
    data_dict = collector.get_multiple_stocks(valid_tickers, args.start, args.end)
    
    if not data_dict:
        print("❌ Falha na coleta de dados!")
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
            filename = f'data/raw/{ticker}_{args.start}_{args.end}.csv'
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
        # Executar análise da Fase 1
        success = execute_analysis(args, logger)
        if success:
            print(f"\n Execução concluída com sucesso!")
            logger.info("Programa finalizado com sucesso")
        else:
            print(f"\n Execução falhou!")
            logger.error(" Programa finalizado com erro")
            sys.exit(1)
            
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