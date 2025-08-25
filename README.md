
# Financces Analyzer

## 📋 Visão Geral

- **📊 FASE 1**: Análise exploratória dos dados históricos de ações
- **💰 FASE 2**: Simulação de cenários de investimento com aportes mensais  
- **🔮 FASE 3**: Previsões futuras usando modelos ARIMA e validação por backtesting

## 🏗️ Arquitetura do Sistema

O projeto utiliza o **Factory Pattern** em todos os módulos, permitindo:
- **Flexibilidade**: Fácil troca de implementações
- **Manutenibilidade**: Código organizado e testável
- **Extensibilidade**: Novos tipos de análise podem ser adicionados facilmente

### **Estrutura dos Módulos**
```
modules/
├── __init__.py                 # Factory principal (InvestmentAnalysisFactory)
├── data.py                     # Coleta de dados via yfinance (DataCollectorFactory)
├── interests.py                # Simuladores de investimento (InvestmentSimulatorFactory)
├── metrics.py                  # Cálculo de métricas (MetricsCalculatorFactory)
├── forecast.py                 # Modelos de previsão ARIMA (ForecastFactory)
├── portfolio.py                # Análise de portfólio (PortfolioFactory)
└── report.py                   # Geração de relatórios (ReportFactory)
```

## 🚀 Execução Integrada

### **🚀 EXECUÇÃO COMPLETA - TODAS AS FASES (PADRÃO)**
```bash
python main.py --tickers PETR4.SA VALE3.SA --weights 0.6 0.4 \
    --start 2023-01-01 --end 2024-01-01 \
    --capital_inicial 10000 --aporte_mensal 1000 \
    --forecast_horizon 30
```

### **📊 MODO EXPLORATÓRIO - SÓ FASE 1**
```bash
python main.py --tickers PETR4.SA VALE3.SA \
    --start 2023-01-01 --end 2024-01-01 \
    --mode explore
```

### **💰 MODO SIMULAÇÃO - FASES 1 E 2**
```bash
python main.py --tickers PETR4.SA VALE3.SA --weights 0.6 0.4 \
    --start 2023-01-01 --end 2024-01-01 \
    --capital_inicial 10000 --aporte_mensal 1000 \
    --mode simulate
```

### **🔮 MODO PREVISÃO - FASES 1 E 3**
```bash
python main.py --tickers PETR4.SA VALE3.SA \
    --start 2023-01-01 --end 2024-01-01 \
    --forecast_horizon 30 \
    --mode forecast
```

## ⚡ Modos de Execução

### **Modo Completo (Padrão)**
- Gera todos os gráficos e análises
- Executa todas as 3 fases
- Salva resultados se solicitado

### **Modo Rápido (sem gráficos)**
```bash
python main.py --tickers PETR4.SA VALE3.SA --weights 0.6 0.4 \
    --start 2023-01-01 --end 2024-01-01 \
    --capital_inicial 10000 --aporte_mensal 1000 \
    --forecast_horizon 30 --no-plots
```

### **Modo com Salvamento de Dados**
```bash
python main.py --tickers PETR4.SA VALE3.SA --weights 0.6 0.4 \
    --start 2023-01-01 --end 2024-01-01 \
    --capital_inicial 10000 --aporte_mensal 1000 \
    --forecast_horizon 30 --save-data
```

## 📊 Funcionalidades do Sistema

### **1. Coleta de Dados (DataCollectorFactory)**
- **Fonte**: Yahoo Finance (yfinance)
- **Cache**: Sistema de cache para evitar downloads desnecessários
- **Validação**: Verificação automática de tickers válidos
- **Tratamento**: Limpeza automática de dados ausentes

### **2. Simulação de Investimentos (InvestmentSimulatorFactory)**
- **Juros Compostos**: Simulação com aportes mensais
- **Carteira de Ações**: Simulação baseada em dados reais
- **Cenários Comparativos**: Juros fixos vs. renda variável
- **Aportes Mensais**: Simulação realista de investimentos

### **3. Métricas de Performance (MetricsCalculatorFactory)**
- **Retorno**: CAGR, retorno total, retorno mensal
- **Risco**: Volatilidade, máximo drawdown, VaR
- **Risco-Retorno**: Sharpe ratio, Sortino ratio
- **Comparação**: Benchmark vs. carteira

### **4. Previsões (ForecastFactory)**
- **Modelo ARIMA**: Previsões estatísticas avançadas
- **Backtesting**: Validação das previsões
- **Horizonte Configurável**: De 30 a 365 dias
- **Métricas de Erro**: MAPE, RMSE

### **5. Análise de Portfólio (PortfolioFactory)**
- **Alocação de Ativos**: Pesos configuráveis
- **Diversificação**: Análise de correlação
- **Rebalanceamento**: Estratégias automáticas
- **Otimização**: Busca da melhor alocação

### **6. Relatórios (ReportFactory)**
- **Executivo**: Resumo das 3 fases
- **Gráficos**: Visualizações interativas
- **Tabelas**: Métricas consolidadas
- **Exportação**: CSV, PDF, Excel

## 🎯 Casos de Uso

### **1. Análise Completa de Investimento**
```bash
# Executar todas as análises de uma vez
python main.py --tickers PETR4.SA VALE3.SA ITUB4.SA --weights 0.4 0.3 0.3 \
    --start 2020-01-01 --end 2024-01-01 \
    --capital_inicial 50000 --aporte_mensal 2000 \
    --forecast_horizon 90 --save-data
```

### **2. Análise Rápida para Decisão**
```bash
# Modo rápido para análise rápida
python main.py --tickers PETR4.SA VALE3.SA --weights 0.6 0.4 \
    --start 2023-01-01 --end 2024-01-01 \
    --capital_inicial 10000 --aporte_mensal 1000 \
    --forecast_horizon 30 --no-plots
```

### **3. Análise de Portfólio Existente**
```bash
# Só análise exploratória
python main.py --tickers PETR4.SA VALE3.SA ITUB4.SA \
    --start 2022-01-01 --end 2024-01-01 \
    --mode explore
```

## 🔧 Configurações Avançadas

### **Parâmetros de Previsão**
- `--forecast_horizon`: Horizonte de previsão em dias (padrão: 30)

### **Parâmetros de Simulação**
- `--taxa_juros_mensal`: Taxa de juros mensal (padrão: 1%)
- `--capital_inicial`: Capital inicial para simulação
- `--aporte_mensal`: Aporte mensal para simulação

### **Parâmetros de Saída**
- `--no-plots`: Pular geração de gráficos
- `--save-data`: Salvar dados em CSV
- `--verbose`: Saída detalhada para debug

### **Modos de Execução**
- `--mode integrated`: Todas as fases (padrão)
- `--mode explore`: Só análise exploratória
- `--mode simulate`: Análise + simulação
- `--mode forecast`: Análise + previsões

## 🧪 Testes Integrados

### **Executar Todos os Testes**
```bash
python -m pytest tests/ -v
```

### **Testes Individuais**
```bash
# Testes de métricas
python -m pytest tests/test_metrics.py -v

# Testes de simuladores
python -m pytest tests/test_simulators.py -v

# Testes de previsões
python -m pytest tests/test_forecast.py -v
```

### **Testes com Cobertura**
```bash
python -m pytest tests/ --cov=modules --cov-report=term-missing
```

## 📁 Estrutura do Projeto

```
c2p/
├── main.py                          # Script principal integrado
├── modules/                         # Módulos do sistema (Factory Pattern)
│   ├── __init__.py                 # Factory principal
│   ├── data.py                     # Coleta de dados
│   ├── interests.py                # Simuladores de investimento
│   ├── metrics.py                  # Cálculo de métricas
│   ├── forecast.py                 # Modelos de previsão
│   ├── portfolio.py                # Análise de portfólio
│   └── report.py                   # Geração de relatórios
├── tests/                          # Testes unitários
│   ├── test_metrics.py            # Testes de métricas
│   ├── test_simulators.py         # Testes de simuladores
│   └── test_forecast.py           # Testes de previsões
├── outputs/                        # Resultados gerados
├── requirements.txt                # Dependências
└── README.md                       # Documentação
```

### **Padrões de Código**
- **Factory Pattern**: Implementado em todos os módulos
- **PEP 8**: Padrões de código Python
- **Type Hints**: Tipagem estática
- **Documentação**: Docstrings em todas as funções
- **Testes**: Cobertura > 90%
- **Logging**: Sistema de logs estruturado

## 📦 Dependências

### **Principais**
- `pandas>=1.5.0`: Manipulação de dados
- `numpy>=1.21.0`: Computação numérica
- `yfinance>=0.2.0`: Dados financeiros
- `statsmodels>=0.14.0`: Modelos ARIMA
- `matplotlib>=3.5.0`: Gráficos
- `seaborn>=0.11.0`: Visualizações avançadas

### **Desenvolvimento**
- `pytest>=7.0.0`: Framework de testes
- `pytest-cov>=4.0.0`: Cobertura de testes
- `black>=22.0.0`: Formatação de código
- `flake8>=5.0.0`: Linting

**🎉 Sistema 100% Integrado com Factory Pattern!**

Execute `python main.py --help` para ver todas as opções disponíveis! 🚀
