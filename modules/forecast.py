# Módulo para previsões e backtesting de dados financeiros
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import holidays

# Suprimir warnings do statsmodels
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=FutureWarning, module='statsmodels')

# Imports para ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

logger = logging.getLogger(__name__)

class DataPreprocessor:
    
    def __init__(self, country: str = 'BR'):
        self.country = country
        self.holidays_calendar = self._get_holidays_calendar()
    
    def _get_holidays_calendar(self) -> holidays.HolidayBase:
        #Retorna calendário de feriados do país
        if self.country == 'BR':
            return holidays.Brazil()
        elif self.country == 'US':
            return holidays.US()
        else:
            return holidays.Brazil()  # Default
    
    def fill_missing_dates(self, data: pd.Series, method: str = 'ffill') -> pd.Series:
        #Preenche datas ausentes (dias sem negociação)
        if data.empty:
            return data
        
        # Garantir que o índice é datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Criar índice de dias úteis
        start_date = data.index.min()
        end_date = data.index.max()
        business_days = pd.bdate_range(start=start_date, end=end_date)
        
        # Reindexar e preencher valores ausentes
        data_reindexed = data.reindex(business_days)
        
        if method == 'ffill':
            data_filled = data_reindexed.fillna(method='ffill')
        elif method == 'bfill':
            data_filled = data_reindexed.fillna(method='bfill')
        elif method == 'interpolate':
            data_filled = data_reindexed.interpolate(method='linear')
        else:
            data_filled = data_reindexed.fillna(method='ffill')
        
        return data_filled
    
    def normalize_weights(self, weights: pd.Series) -> pd.Series:
        #Normaliza pesos para somarem 1
        if weights.empty:
            return weights
        total_weight = weights.sum()
        if total_weight == 0:
            logger.warning("Todos os pesos são zero, distribuindo igualmente")
            return pd.Series(1.0 / len(weights), index=weights.index)
        
        return weights / total_weight
    
    def adjust_contributions_to_business_days(self, contributions: pd.Series) -> pd.Series:
        # Ajusta aportes para dias úteis
        if contributions.empty:
            return contributions
        
        adjusted_contributions = contributions.copy()
        
        for date in contributions.index:
            if not self._is_business_day(date):
                # Encontrar próximo dia útil
                next_business_day = self._get_next_business_day(date)
                if next_business_day in adjusted_contributions.index:
                    adjusted_contributions[next_business_day] += contributions[date]
                else:
                    adjusted_contributions[next_business_day] = contributions[date]
                adjusted_contributions[date] = 0
        
        return adjusted_contributions
    
    def _is_business_day(self, date: Union[datetime, pd.Timestamp]) -> bool:
        # Verifica se é dia útil
        if isinstance(date, pd.Timestamp):
            date = date.to_pydatetime()
        
        # Verificar se é fim de semana
        if date.weekday() >= 5:  # Sábado = 5, Domingo = 6
            return False
        
        # Verificar se é feriado
        if date in self.holidays_calendar:
            return False
        
        return True
    
    def _get_next_business_day(self, date: Union[datetime, pd.Timestamp]) -> datetime:
        #Encontra o próximo dia útil
        if isinstance(date, pd.Timestamp):
            date = date.to_pydatetime()
        
        next_day = date + timedelta(days=1)
        while not self._is_business_day(next_day):
            next_day += timedelta(days=1)
        
        return next_day
    
    def validate_data_sufficiency(self, data: pd.Series, min_periods: int = 50) -> Tuple[bool, str]:
        # Valida se há dados suficientes para previsão
        if data.empty:
            return False, "Série de dados vazia"
        
        if len(data) < min_periods:
            return False, f"Dados insuficientes: {len(data)} pontos (mínimo: {min_periods})"
        
        # Verificar se há muitos valores ausentes
        missing_ratio = data.isna().sum() / len(data)
        if missing_ratio > 0.3:
            return False, f"Muitos valores ausentes: {missing_ratio:.1%}"
        
        return True, "Dados suficientes para previsão"

class ForecasterInterface(ABC):
    #Interface para previsores
    
    @abstractmethod
    def train(self, data: pd.Series) -> bool:
        #Treina o modelo com dados históricos
        pass
    
    @abstractmethod
    def predict(self, periods: int) -> pd.Series:
        #Faz previsões para os próximos períodos
        pass
    
    @abstractmethod
    def get_model_info(self) -> str:
        #Retorna informações do modelo
        pass
    
    @abstractmethod
    def get_accuracy_metrics(self) -> Dict[str, float]:
        #Retorna métricas de acurácia do modelo
        pass

class NaiveForecaster(ForecasterInterface):
    #Previsor Naive (baseline)
    
    def __init__(self, method: str = 'last_value'):
        self.method = method
        self.is_trained = False
        self.data = None
        self.last_value = None
        self.trend = None
    
    def train(self, data: pd.Series) -> bool:
        try:
            if data.empty:
                return False
            
            self.data = data.dropna()
            if len(self.data) < 2:
                return False
            
            self.last_value = self.data.iloc[-1]
            
            if self.method == 'trend':
                self.trend = self.data.diff().mean()
            elif self.method == 'seasonal':
                # Média dos últimos 7 dias (semana)
                if len(self.data) >= 7:
                    self.seasonal_pattern = self.data.tail(7).values
                else:
                    self.method = 'last_value'
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.error(f"Erro no treinamento Naive: {e}")
            return False
    
    def predict(self, periods: int) -> pd.Series:
        if not self.is_trained:
            raise ValueError("Modelo Naive não treinado")
        
        try:
            predictions = []
            
            if self.method == 'last_value':
                predictions = [self.last_value] * periods
            elif self.method == 'trend':
                for i in range(1, periods + 1):
                    pred = self.last_value + (self.trend * i)
                    predictions.append(max(0, pred))
            elif self.method == 'seasonal':
                for i in range(periods):
                    seasonal_idx = i % len(self.seasonal_pattern)
                    predictions.append(self.seasonal_pattern[seasonal_idx])
            
            # Criar índice de datas futuras
            last_date = self.data.index[-1]
            future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), 
                                       periods=periods)
            
            return pd.Series(predictions, index=future_dates)
            
        except Exception as e:
            logger.error(f"Erro na previsão Naive: {e}")
            return pd.Series()
    
    def get_model_info(self) -> str:
        return f"Naive Forecast ({self.method})"
    
    def get_accuracy_metrics(self) -> Dict[str, float]:
        return {"method": self.method, "last_value": self.last_value}

class HoltWintersForecaster(ForecasterInterface):

    def __init__(self, seasonal_periods: int = 12):
        self.seasonal_periods = seasonal_periods
        self.is_trained = False
        self.data = None
        self.model = None
        self.fitted_model = None
    
    def train(self, data: pd.Series) -> bool:
        #Treina modelo Holt-Winters
        try:
            if not STATSMODELS_AVAILABLE:
                logger.error("statsmodels não instalado")
                return False
            
            self.data = data.dropna()
            if len(self.data) < self.seasonal_periods * 2:
                logger.warning(f"Dados insuficientes para Holt-Winters (mínimo: {self.seasonal_periods * 2})")
                return False
            
            # Definir frequência do índice
            if not self.data.index.freq:
                self.data = self.data.asfreq('B').ffill()
            
            # Ajustar modelo
            self.model = ExponentialSmoothing(
                self.data,
                seasonal_periods=self.seasonal_periods,
                trend='add',
                seasonal='add'
            )
            
            self.fitted_model = self.model.fit()
            self.is_trained = True
            
            return True
            
        except Exception as e:
            logger.error(f"Erro no treinamento Holt-Winters: {e}")
            return False
    
    def predict(self, periods: int) -> pd.Series:
        #Faz previsões com Holt-Winters
        if not self.is_trained:
            raise ValueError("Modelo Holt-Winters não treinado")
        
        try:
            forecast = self.fitted_model.forecast(steps=periods)
            
            # Criar índice de datas futuras
            last_date = self.data.index[-1]
            future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), 
                                       periods=periods)
            
            return pd.Series(forecast, index=future_dates)
            
        except Exception as e:
            logger.error(f"Erro na previsão Holt-Winters: {e}")
            return pd.Series()
    
    def get_model_info(self) -> str:
        if self.is_trained:
            return f"Holt-Winters (seasonal_periods={self.seasonal_periods})"
        return "Holt-Winters - Não treinado"
    
    def get_accuracy_metrics(self) -> Dict[str, float]:
        if self.is_trained:
            return {
                "aic": self.fitted_model.aic,
                "bic": self.fitted_model.bic,
                "seasonal_periods": self.seasonal_periods
            }
        return {}

class ARIMAForecaster(ForecasterInterface):
    #Previsor ARIMA 
    def __init__(self, p: int = 2, d: int = 1, q: int = 2):
        self.is_trained = False
        self.data = None
        self.p, self.d, self.q = p, d, q
        self.model = None
        self.fitted_model = None
        
    def train(self, data: pd.Series) -> bool:
        #Treina modelo ARIMA com parâmetros para previsões variadas
        try:
            self.data = data.dropna()
            if len(self.data) < 40:
                logger.warning("Dados insuficientes para ARIMA (mínimo 40 pontos)")
                return False
            
            if not STATSMODELS_AVAILABLE:
                logger.error("statsmodels não instalado")
                return False
            
            # Definir frequência do índice
            if not self.data.index.freq:
                self.data = self.data.asfreq('B').ffill()
            
            # Teste de estacionaridade - sempre aplicar diferenciação para mais variação
            adf_result = adfuller(self.data)
            if adf_result[1] > 0.05:
                self.d = 1
            else:
                self.d = 1  # Forçar diferenciação para mais variação
            
            # Grid search mais agressivo para previsões variadas
            best_aic = float('inf')
            best_params = (self.p, self.d, self.q)
            
            # Parâmetros mais altos para mais variação
            param_combinations = [
                (2, self.d, 2), (3, self.d, 2), (2, self.d, 3), (3, self.d, 3),
                (4, self.d, 2), (2, self.d, 4), (4, self.d, 3), (3, self.d, 4)
            ]
            
            for p, d, q in param_combinations:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = ARIMA(self.data, order=(p, d, q))
                        fitted = model.fit()
                        
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_params = (p, d, q)
                            self.fitted_model = fitted
                except:
                    continue
            
            self.p, self.d, self.q = best_params
            self.model = ARIMA(self.data, order=best_params)
            
            self.is_trained = True
            logger.info(f"ARIMA{best_params} treinado com AIC: {best_aic:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Erro no treinamento ARIMA: {e}")
            return False
    
    def predict(self, periods: int) -> pd.Series:
        #Faz previsões com ARIMA e adiciona variação realista
        if not self.is_trained:
            raise ValueError("Modelo ARIMA não treinado")
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                forecast = self.fitted_model.forecast(steps=periods)
            
            # Adicionar variação realista baseada na volatilidade dos dados
            varied_forecast = self._add_realistic_variation(forecast)
            
            last_date = self.data.index[-1]
            future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), 
                                       periods=periods)
            
            return pd.Series(varied_forecast, index=future_dates)
            
        except Exception as e:
            logger.error(f"Erro na previsão ARIMA: {e}")
            return self._fallback_predict(periods)
    
    def _fallback_predict(self, periods: int) -> pd.Series:
        #Fallback para modelo simples se ARIMA falhar
        last_value = self.data.iloc[-1]
        trend = self.data.diff().mean()
        
        predictions = []
        for i in range(1, periods + 1):
            pred = last_value + (trend * i)
            predictions.append(max(0, pred))
        
        last_date = self.data.index[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), 
                                    periods=periods)
        
        return pd.Series(predictions, index=future_dates)
    
    def _add_realistic_variation(self, forecast: pd.Series) -> pd.Series:
        #Adiciona variação realista baseada na volatilidade dos dados
        try:
            # Calcular volatilidade dos dados históricos
            returns = self.data.pct_change().dropna()
            volatility = returns.std()
            
            # Adicionar ruído baseado na volatilidade real
            noise_factor = volatility * 0.5  # 50% da volatilidade real
            random_noise = np.random.normal(0, noise_factor, len(forecast))
            
            # Aplicar ruído às previsões
            varied_forecast = forecast * (1 + random_noise)
            
            # Garantir que não seja negativo
            varied_forecast = varied_forecast.clip(lower=0)
            
            return varied_forecast
            
        except Exception as e:
            logger.warning(f"Erro ao adicionar variação: {e}, retornando previsões originais")
            return forecast
    
    def get_model_info(self) -> str:
        if self.is_trained:
            return f"ARIMA({self.p},{self.d},{self.q}) - AIC: {self.fitted_model.aic:.2f}"
        return f"ARIMA({self.p},{self.d},{self.q}) - Não treinado"
    
    def get_accuracy_metrics(self) -> Dict[str, float]:
        if self.is_trained:
            return {
                "aic": self.fitted_model.aic,
                "bic": self.fitted_model.bic,
                "params": (self.p, self.d, self.q)
            }
        return {}

class SensitivityAnalyzer:
    #Analisador de sensibilidade para taxas de juros e aportes
    def __init__(self, base_rate: float = 0.01):
        self.base_rate = base_rate
    
    def analyze_interest_rate_sensitivity(self, 
                                       principal: float, 
                                       periods: int,
                                       rate_variations: List[float] = None) -> Dict[str, List[float]]:
        #Analisa sensibilidade a variações na taxa de juros
        if rate_variations is None:
            rate_variations = [-0.005, -0.002, 0, 0.002, 0.005] 
        
        results = {"rates": [], "final_values": [], "variations": []}
        
        for rate_change in rate_variations:
            new_rate = self.base_rate + rate_change
            final_value = principal * (1 + new_rate) ** periods
            
            results["rates"].append(new_rate)
            results["final_values"].append(final_value)
            results["variations"].append((final_value - principal) / principal)
        
        return results
    
    def analyze_contribution_sensitivity(self, 
                                      initial_principal: float,
                                      monthly_contribution: float,
                                      periods: int,
                                      contribution_variations: List[float] = None) -> Dict[str, List[float]]:
        #Analisa sensibilidade a variações nos aportes mensais
        if contribution_variations is None:
            contribution_variations = [-0.5, -0.2, 0, 0.2, 0.5]  
        
        results = {"contributions": [], "final_values": [], "variations": []}
        
        for contrib_change in contribution_variations:
            new_contribution = monthly_contribution * (1 + contrib_change)
            final_value = initial_principal * (1 + self.base_rate) ** periods
            
            # Adicionar valor dos aportes
            for i in range(periods):
                final_value += new_contribution * (1 + self.base_rate) ** (periods - i - 1)
            
            results["contributions"].append(new_contribution)
            results["final_values"].append(final_value)
            results["variations"].append((final_value - initial_principal) / initial_principal)
        
        return results

class BacktestManager:
    #Gerenciador de backtesting com comparação de modelos
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        
    def run_backtest(self, data: pd.Series, forecasters) -> Dict[str, Any]:
        """Executa backtesting com dados para gráficos"""
        if len(data) < self.window_size * 2:
            logger.warning("Dados insuficientes para backtesting")
            return {}
        
        # Converter para lista se for um único forecaster
        if not isinstance(forecasters, list):
            forecasters = [forecasters]
        
        # Usar tipo do modelo como chave
        results = {type(forecaster).__name__: [] for forecaster in forecasters}
        
        # Reduzir janelas para gráficos mais limpos
        step_size = max(1, len(data) // 8)  # Apenas 8 janelas para gráficos limpos
        
        # Listas para armazenar dados dos gráficos
        all_actual_values = []
        all_predicted_values = []
        
        for i in range(self.window_size, len(data) - 30, step_size): 
            train_data = data.iloc[i-self.window_size:i]
            actual = data.iloc[i:i+30]
            
            for forecaster in forecasters:
                if forecaster.train(train_data):
                    predictions = forecaster.predict(30)
                    
                    if len(actual) == len(predictions):
                        metrics = self._calculate_metrics(actual, predictions)
                        
                        # Armazenar dados para gráficos
                        all_actual_values.extend(actual.values.tolist())
                        all_predicted_values.extend(predictions.values.tolist())
                        
                        results[type(forecaster).__name__].append({
                            'start_date': actual.index[0],
                            'mape': metrics['mape'],
                            'rmse': metrics['rmse']
                        })
        
        # Calcular métricas agregadas
        aggregated_results = {}
        for model_name, model_results in results.items():
            if model_results:
                mape_values = [r['mape'] for r in model_results if r['mape'] != float('inf')]
                rmse_values = [r['rmse'] for r in model_results if r['rmse'] != float('inf')]
                
                aggregated_results[model_name] = {
                    'avg_mape': np.mean(mape_values) if mape_values else float('inf'),
                    'avg_rmse': np.mean(rmse_values) if rmse_values else float('inf'),
                    'total_predictions': len(model_results)
                }
        
        # Retornar estrutura com dados para gráficos
        if aggregated_results:
            first_model = list(aggregated_results.keys())[0]
            first_results = aggregated_results[first_model]
            
            return {
                'metrics': first_results,
                'detailed_results': results[first_model],
                'actual_values': all_actual_values,  # Dados para gráficos
                'predicted_values': all_predicted_values  # Dados para gráficos
            }
        
        return {}
    
    def _calculate_metrics(self, actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
        #Calcula métricas de erro
        if len(actual) == 0 or len(predicted) == 0:
            return {"mape": float('inf'), "rmse": float('inf')}
        # MAPE
        actual_clean = actual[actual != 0]
        if len(actual_clean) == 0:
            mape = float('inf')
        else:
            mape = np.mean(np.abs((actual_clean - predicted[:len(actual_clean)]) / actual_clean)) * 100
        # RMSE
        min_len = min(len(actual), len(predicted))
        rmse = np.sqrt(np.mean((actual.iloc[:min_len] - predicted.iloc[:min_len]) ** 2))
        
        return {"mape": mape, "rmse": rmse}
    
    def _aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        #Agrega métricas dos resultados"""
        mape_values = [r['mape'] for r in results if r['mape'] != float('inf')]
        rmse_values = [r['rmse'] for r in results if r['rmse'] != float('inf')]
        
        return {
            'avg_mape': np.mean(mape_values) if mape_values else float('inf'),
            'avg_rmse': np.mean(rmse_values) if rmse_values else float('inf'),
            'total_predictions': len(results)
        }

class ForecastFactory:
    #Factory para criar previsores e componentes relacionado
    @staticmethod
    def create_forecaster(forecaster_type: str = 'arima') -> ForecasterInterface:
        if forecaster_type.lower() == 'arima':
            return ARIMAForecaster()
        elif forecaster_type.lower() == 'holt_winters':
            return HoltWintersForecaster()
        elif forecaster_type.lower() == 'naive':
            return NaiveForecaster()
        else:
            raise ValueError(f"Tipo de previsor não suportado: {forecaster_type}. Use: arima, holt_winters, naive")
    
    @staticmethod
    def create_backtest_manager(window_size: int = 60) -> BacktestManager:
        return BacktestManager(window_size)
    
    @staticmethod
    def create_data_preprocessor(country: str = 'BR') -> DataPreprocessor:
        return DataPreprocessor(country)
    
    @staticmethod
    def create_sensitivity_analyzer(base_rate: float = 0.01) -> SensitivityAnalyzer:
        return SensitivityAnalyzer(base_rate)