
# 🚀 Investment Simulator

**A comprehensive investment analysis and simulation tool built with Python**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Phase%202%20Complete-brightgreen.svg)](README.md)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Development Phases](#development-phases)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### 🎯 **Core Functionality**
- **Real-time data collection** from Yahoo Finance
- **Compound interest simulation** with monthly contributions
- **Stock portfolio simulation** using real market data or fixed returns
- **Advanced financial metrics** calculation (CAGR, Sharpe ratio, drawdown)
- **Portfolio analysis** with rebalancing recommendations
- **Comprehensive reporting** with executive summaries

### 🔧 **Technical Features**
- **Factory Pattern** implementation for modular design
- **Robust error handling** and data validation
- **Configurable parameters** for flexible simulations
- **Professional logging** and debugging support
- **Cross-platform compatibility** (Windows, macOS, Linux)

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/investment-simulator.git
cd investment-simulator

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Usage

### Phase 1: Basic Stock Analysis
```bash
python main.py \
  --tickers PETR4.SA VALE3.SA ITUB4.SA \
  --start-date 2023-01-01 \
  --end-date 2023-12-31
```

### Phase 2: Investment Simulation
```bash
python main.py \
  --tickers PETR4.SA VALE3.SA ITUB4.SA \
  --weights 0.4 0.3 0.3 \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --capital-inicial 10000 \
  --aporte-mensal 1000 \
  --taxa-juros 0.01 \
  --retorno-fixo 0.015
```

### Command Line Arguments

| Argument | Description | Required | Default |
|----------|-------------|----------|---------|
| `--tickers` | Stock symbols to analyze | Yes | - |
| `--weights` | Portfolio weights | No | Equal weights |
| `--start-date` | Analysis start date (YYYY-MM-DD) | Yes | - |
| `--end-date` | Analysis end date (YYYY-MM-DD) | Yes | - |
| `--capital-inicial` | Initial capital amount | Phase 2 | - |
| `--aporte-mensal` | Monthly contribution amount | Phase 2 | - |
| `--taxa-juros` | Monthly interest rate | No | 1% |
| `--retorno-fixo` | Fixed monthly return rate | No | 1% |

## 🏗️ Architecture

### **Modular Design**
```
modules/
├── data.py          # Data collection and validation
├── interests.py     # Investment simulators
├── metrics.py       # Financial metrics calculation
├── portfolio.py     # Portfolio analysis
└── report.py        # Reporting and visualization
```

### **Design Patterns**
- **Factory Pattern**: Object creation and management
- **Strategy Pattern**: Different simulation approaches
- **Observer Pattern**: Event logging and monitoring

### **Key Classes**
- `InvestmentSimulator`: Main simulation orchestrator
- `CompoundInterestSimulator`: Fixed interest simulation
- `StockPortfolioSimulator`: Stock portfolio simulation
- `PerformanceMetrics`: Financial metrics calculation
- `PortfolioManager`: Portfolio analysis and optimization

## 📈 Development Phases

### **Phase 1** ✅ Complete
- `yfinance`: Financial data library
- `pandas`: Data manipulation
- `matplotlib`: Basic plotting
- Adjusted vs. normal prices
- Returns and normalization

### **Phase 2** ✅ Complete
- Compound interest and future value
- Regular contribution simulation
- Pandas loops and iterations
- Cumulative return calculations
- **Advanced financial metrics**
- **Portfolio analysis**
- **Executive reporting**

### **Phase 3** 🚧 In Progress
- Advanced financial metrics
- Descriptive statistics
- Risk vs. return analysis
- Professional visualizations

### **Phase 4** 📋 Planned
- Forecasting models (Holt-Winters)
- Backtesting and validation
- Error metrics
- Time series analysis

### **Phase 5** 📋 Planned
- CLI with `argparse` ✅ (Already implemented)
- Unit testing with `pytest`
- Code structuring ✅ (Already implemented)
- Documentation ✅ (This README)

## 📊 Example Output

### Investment Simulation Results
```
🏆 RESULTADO FINAL COMPARATIVO:
--------------------------------------------------
💰 CAPITAL FINAL:
  🏦 Juros Fixos:     R$     22,018.61
  📈 Carteira Ações:  R$     29,497.85
  💰 Diferença:       R$      7,479.24

🏆 Carteira de Ações venceu por 34.0%
```

## 🙏 Acknowledgments

- **Yahoo Finance** for financial data
- **Pandas** community for data manipulation tools
- **Matplotlib** team for visualization capabilities
- **Python** community for the amazing ecosystem
