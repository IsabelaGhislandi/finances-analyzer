#Módulo para coleta e tratamento de dados financeiros
import yfinance as yf
import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

#Classe responsável pela coleta de dados financeiros
class DataCollectorInterface(ABC):
    
    @abstractmethod
    def get_stock_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        #Coleta dados de uma ação específica
        # ticker (str): Código da ação (ex: 'PETR4.SA')
        # start_date (str): Data inicial (YYYY-MM-DD)
        # end_date (str): Data final (YYYY-MM-DD)
        pass
    
    @abstractmethod
    def get_multiple_stocks(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        # Coleta dados de múltiplas ações
        # Args:
            # tickers: Lista de códigos das ações
            # start_date: Data inicial
            # end_date: Data final
        pass

    @abstractmethod
    def validate_tickers(self, tickers: List[str]) -> Tuple[List[str], List[str]]:
        #Valida se os tickers existem fazendo download de amostra
        pass
    
class YFinanceDataCollector(DataCollectorInterface):
    def __init__(self):
        self.cache = {}  # Cache simples para evitar downloads desnecessários

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:  
        #Limpa e trata dados ausentes
        #Args: data: DataFrame com dados brutos
        # Forward fill para dias sem negociação
        data = data.ffill()
        
        # Remover linhas onde ainda há NaN (início da série)
        data = data.dropna()
        
        # Garantir que o índice seja datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        return data
        
    def get_stock_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        cache_key = f"{ticker}_{start_date}_{end_date}"
        
        # Verificar cache
        if cache_key in self.cache:
            logger.info(f"Usando dados do cache para {ticker}")
            return self.cache[cache_key]
        
        try:
            logger.info(f"Baixando dados de {ticker} ({start_date} até {end_date})")
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                logger.warning(f"Nenhum dado encontrado para {ticker}")
                return None
            
            # Tratar dados ausentes
            data = self._clean_data(data)
            
            # Salvar no cache
            self.cache[cache_key] = data
            
            logger.info(f"Dados de {ticker} coletados com sucesso: {len(data)} registros")
            return data
            
        except Exception as e:
            logger.error(f"Erro ao coletar dados de {ticker}: {e}")
            return None
    
    def get_multiple_stocks(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        results = {}
        
        for ticker in tickers:
            data = self.get_stock_data(ticker, start_date, end_date)
            if data is not None:
                results[ticker] = data
            else:
                logger.warning(f"Pulando {ticker} - dados não disponíveis")
    
        logger.info(f"Coletados dados de {len(results)}/{len(tickers)} ativos")
        return results

    def validate_tickers(self, tickers: List[str]) -> Tuple[List[str], List[str]]:
        #Valida se os tickers existem fazendo download de amostra
        valid_tickers = []
        invalid_tickers = []
        
        # Data de teste (últimos 5 dias úteis)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(start=start_date, end=end_date)
                
                if not data.empty:
                    valid_tickers.append(ticker)
                    logger.info(f"✅ {ticker} - válido")
                else:
                    invalid_tickers.append(ticker)
                    logger.warning(f"{ticker} - sem dados")
                    
            except Exception as e:
                invalid_tickers.append(ticker)
                logger.error(f"{ticker} - erro: {e}")
        
        return valid_tickers, invalid_tickers
        

def parse_tickers(tickers_input) -> List[str]:
        #Converte string de tickers separados por vírgula em lista ou lista já processada
        #Args:
            # tickers_input: String como "PETR4.SA,VALE3.SA,ITUB4.SA" ou lista já processada

        if not tickers_input:
            return []
        
        # Se já é uma lista, apenas normalizar
        if isinstance(tickers_input, list):
            tickers = [ticker.strip().upper() for ticker in tickers_input]
        else:
            # Se é string, fazer split
            tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
        
        tickers = [ticker for ticker in tickers if ticker]  # Remove strings vazias
        
        return tickers

def parse_weights(weights_input, num_assets: int) -> List[float]:
        #Converte string de pesos em lista normalizada ou lista já processada
        
        if not weights_input:
            # Pesos iguais se não especificado
            weight = 1.0 / num_assets
            return [weight] * num_assets
        
        try:
            # Se já é uma lista, usar diretamente
            if isinstance(weights_input, list):
                weights = [float(w) for w in weights_input]
            else:
                # Se é string, fazer split
                weights = [float(w.strip()) for w in weights_input.split(',')]
            
            if len(weights) != num_assets:
                logger.warning(f"Número de pesos ({len(weights)}) != número de ativos ({num_assets})")
                # Usar pesos iguais
                weight = 1.0 / num_assets
                return [weight] * num_assets
            
            # Normalizar para somar 1.0
            total = sum(weights)
            if total == 0:
                weight = 1.0 / num_assets
                return [weight] * num_assets
            
            normalized_weights = [w / total for w in weights]
            logger.info(f"Pesos normalizados: {normalized_weights}")
            
            return normalized_weights
            
        except ValueError as e:
            logger.error(f"Erro ao processar pesos '{weights_input}': {e}")
            # Fallback para pesos iguais
            weight = 1.0 / num_assets
            return [weight] * num_assets

def align_data_by_date(data_dict):
    # Alinha dados de diferentes mercados por data
    aligned_data = {}
    # Converter todos para mesma frequência (diária)
    for ticker, data in data_dict.items():
        # Normalizar fuso horário
        data.index = data.index.tz_localize(None)
        # Resample para frequência diária
        daily_data = data.resample('D').last()
        aligned_data[ticker] = daily_data
        
        # Criar DataFrame com todos os tickers
        df = pd.DataFrame(aligned_data)
        
        # Remover linhas com NaN (dias sem negociação)
        df = df.dropna()
        
        return df

def calculate_correlation_matrix(data_dict):
    #Calcula matriz de correlação entre tickers de diferentes mercados
    try:
        # Alinhar dados
        aligned_df = align_data_by_date(data_dict)
            
        if len(aligned_df) < 10:
            logger.warning(f"Poucos dados alinhados ({len(aligned_df)} registros) para correlação")
            return None
            
        # Calcular correlação
        correlation_matrix = aligned_df.corr()
            
        logger.info(f"Matriz de correlação calculada com {len(aligned_df)} registros alinhados")
        return correlation_matrix
            
    except Exception as e:
        logger.error(f"Erro ao calcular correlação: {e}")
        return None

    # Função de conveniência para uso direto
def quick_download(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        ## Download rápido para testes e exploração
        # ticker: Código da ação
        # period: Período (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        return data if not data.empty else None
    except Exception as e:
        logger.error(f"Erro no download rápido de {ticker}: {e}")
        return None

class DataCollectorFactory:
    @staticmethod
    def create_collector(source: str = 'yfinance') -> DataCollectorInterface:
        if source == 'yfinance':
            return YFinanceDataCollector()
        else:
            raise ValueError(f"Fonte '{source}' não suportada")