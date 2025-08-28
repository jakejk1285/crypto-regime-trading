# Cryptocurrency Trading Strategy System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue)](https://isocpp.org/)
[![Build Status](https://img.shields.io/badge/Build-Passing-green)](https://github.com/jakejk1285/crypto-trading-strategy)

## 🚀 Overview

A sophisticated cryptocurrency trading system that combines **machine learning-driven regime detection** with **quantitative trading strategies**. Built for educational and research purposes with **1-year historical data** from CoinGecko's free API. The system consists of two main components:

1. **Python Research & Backtesting Engine** - Advanced regime analysis and strategy validation
2. **C++ Paper Trading System** - Real-time execution and portfolio management

## 📊 Key Features

### Research & Analysis (Python)
- **Regime Detection**: 7-regime market classification using PCA and K-means clustering
- **EV Optimization**: Expected Value analysis with regime-specific allocation optimization
- **Enhanced Performance**: 30.53% annual returns with data-driven position sizing
- **PC Factor Integration**: Dynamic position sizing based on Principal Component strength
- **Decision Explanations**: Comprehensive logging of all trading decision rationale
- **Feature Engineering**: 988+ technical indicators and market features
- **Performance Analytics**: Sharpe ratio, drawdown analysis, regime attribution

### Trading Execution (C++)
- **Real-time Trading**: Paper trading with Alpaca Markets integration
- **Dynamic Position Sizing**: Regime-based allocation with volatility adjustment
- **Risk Management**: Trailing stops, dynamic stop-loss/take-profit levels
- **Portfolio Rebalancing**: Automatic regime transition handling
- **Performance Tracking**: Real-time P&L and trade analytics

## 🏗️ System Architecture

```
CryptoTrading/
├── crypto_cluster_pca/           # Python Research System
│   ├── backtest_system/          # Backtesting engine
│   ├── research/                 # Jupyter notebooks for analysis
│   ├── src/                      # Core regime detection logic
│   └── data/                     # Historical analysis results
├── paper_trading/                # C++ Trading System
│   ├── src/                      # Core trading implementation
│   ├── include/                  # Header files
│   └── CMakeLists.txt           # Build configuration
└── shared_regime_data/           # Regime data exchange
    └── regime_output/            # JSON regime files
```

## 📈 Trading Strategy

### Market Regime Classification

The system identifies 7 distinct market regimes with **performance-optimized allocation**:

| Regime ID | Name | Strategy | Actual Performance | Allocation | Status |
|-----------|------|----------|-------------------|------------|---------|
| 0 | BULL_MOMENTUM | Aggressive | 55.6% WR, $839 avg P&L | 35% | 🚀 **Top Performer** |
| 1 | SHARP_CORRECTION | Defensive | 44.0% WR, -$59 avg P&L | 5% | ⚠️ Minimal |
| 2 | SIDEWAYS_MARKET | Balanced | 40.0% WR, -$264 avg P&L | 5% | ❌ Avoided |
| 3 | WAIT_AND_SEE | None | N/A | 0% | 🚫 Eliminated |
| 4 | CONSERVATIVE | Moderate | 52.9% WR, $540 avg P&L | 25% | ✅ Strong |
| 5 | VOLATILE_REBOUND | Selective | 57.7% WR, $87 avg P&L | 12% | ⚡ Limited |
| 6 | EXTREME_VOLATILITY | None | N/A | 0% | 🚫 Eliminated |

### Performance-Based Optimization

The strategy **dynamically allocates** based on actual backtest performance:
- **BULL_MOMENTUM (Regime 0)**: 35% allocation - Highest average P&L per trade
- **CONSERVATIVE (Regime 4)**: 25% allocation - Strong consistent performer
- **VOLATILE_REBOUND (Regime 5)**: 12% allocation - Good win rate but low average P&L
- **Losing regimes**: 5% or 0% allocation - Capital protection

### Enhanced Position Sizing Algorithm

```python
# EV-optimized position sizing with PC factor strength
position_size = ev_base_allocation × pc_strength_multiplier × persistence_bonus × coin_score × risk_limits

# Where:
# - ev_base_allocation: Expected Value optimized allocation per regime:
#   * Extreme Outlier (Regime 3): 40% (2.230R EV, 75% WR)
#   * Breakout Momentum (Regime 5): 30% (0.761R EV, 60% WR) 
#   * Stable Growth (Regime 0): 25% (0.494R EV, 54% WR)
#   * Others: 15%-1% based on negative EV performance
# - pc_strength_multiplier: PC1/PC2/PC3 factor strength (0.5x - 1.6x range)
# - persistence_bonus: Regime stability multiplier (0.7x - 1.3x)
# - coin_score: Enhanced PC-factor cryptocurrency attractiveness
# - risk_limits: Portfolio and correlation constraints (max 50% correlation exposure)
```

## ⚡ Enhanced Strategy Features

### Sharpe Ratio Optimization (NEW)
- **Risk-Adjusted Performance**: 28.5% improvement in Sharpe ratio (0.621 → 0.798)
- **Smart Regime Selectivity**: Enhanced focus on high-EV regimes, aggressive filtering of negative-EV regimes
- **Volatility Control**: 19.2% reduction in portfolio volatility through intelligent position sizing
- **Balanced Risk-Reward**: Moderate risk controls that preserve profitable trades while reducing volatility
- **Improved Drawdown Management**: 12.2% reduction in maximum drawdown

### Expected Value (EV) Optimization
- **Data-Driven Allocation**: Position sizes based on historical Expected Value analysis
- **Regime EV Ranking**: Regimes ranked by actual performance (2.230R to -0.901R range)
- **Dynamic Thresholds**: Entry thresholds adjusted per regime based on EV data
- **Risk-Adjusted Sizing**: Positions sized according to regime-specific win rates and R multiples

### PC Factor Strength Integration  
- **PC1 Market Direction**: 0.75x - 1.3x multiplier based on directional signal strength
- **PC2 Volatility Adjustment**: 0.6x - 1.2x adjustment for market stress conditions
- **PC3 Sector/Style Bonus**: Up to 1.1x bonus for strong sector rotation signals
- **Combined Strength**: Multi-factor approach with 0.7x - 1.4x total range (optimized for Sharpe)

### Decision Explanations System
- **Comprehensive Logging**: Every trade decision explained with detailed reasoning
- **Regime Analysis**: EV ranking, win rates, persistence, and market stress factors
- **PC Factor Breakdown**: Individual PC1/PC2/PC3 contributions to position sizing
- **Risk Management**: Stop loss, take profit, and portfolio risk calculations
- **Performance Tracking**: Decision success rates and rejection reason analysis

## 🛠️ Installation & Setup

### Prerequisites

**Python Environment:**
```bash
Python 3.8+
pandas, numpy, scikit-learn
matplotlib, seaborn
requests (for CoinGecko API - free tier, 1-year data limit)
```

**C++ Environment:**
```bash
CMake 3.16+
C++17 compatible compiler
jsoncpp library
libcurl
```

**API Requirements:**
- **CoinGecko**: Free tier (365-day historical data limit)
- **Alpaca Markets**: Paper trading account (free)

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/jakejk1285/crypto-trading-strategy
cd crypto-trading-strategy
```

2. **Install Python dependencies:**
```bash
cd crypto_cluster_pca
pip install -r requirements.txt
```

3. **Install C++ dependencies (macOS):**
```bash
brew install jsoncpp curl cmake
```

4. **Run regime analysis:**
```bash
cd crypto_cluster_pca/src
python crypto_regime_analysis.py
```

5. **Build C++ trading system:**
```bash
cd paper_trading
mkdir build && cd build
cmake ..
make
```

6. **Run paper trading:**
```bash
./crypto_paper_trading
```

## 📊 Backtesting Results

### Performance Summary (2024-2025, 1-Year Limited Dataset)

**🎯 Sharpe-Optimized Strategy (Recommended)**
- **Total Return**: 28.90% 
- **Sharpe Ratio**: 0.798 ⚡ (+28.5% improvement)
- **Annualized Volatility**: 23.85% (-19.2% reduction)
- **Maximum Drawdown**: 15.12%
- **Win Rate**: 53.2%
- **Total Trades**: 106
- **Final Portfolio Value**: $128,901

**📊 Original Enhanced Strategy**
- **Total Return**: 30.53% (improved from 21.12%)
- **Sharpe Ratio**: 0.621 
- **Maximum Drawdown**: 17.23%
- **Win Rate**: 51.7%
- **Total Trades**: 118
- **Final Portfolio Value**: $130,535

### Risk Metrics
- **Sharpe-Optimized Annualized Return**: 21.58%
- **Sharpe-Optimized Volatility**: 23.85% (vs 29.52% original)
- **Alpha vs Bitcoin**: -58.84% (Bitcoin buy & hold: 87.74%)
- **Risk-Free Rate**: 3.95% (10Y Treasury Jan 2, 2024, [FRED DGS10](https://fred.stlouisfed.org/series/DGS10))

### Strategy Performance Comparison
| Metric | Original Strategy | Sharpe-Optimized | Improvement |
|--------|------------------|------------------|-------------|
| **Sharpe Ratio** | 0.621 | 0.798 | **+28.5%** |
| **Volatility** | 29.52% | 23.85% | **-19.2%** |
| **Return** | 30.53% | 28.90% | -5.3% |
| **Max Drawdown** | 17.23% | 15.12% | **-12.2%** |
| **Risk-Adj Return** | 1.03 | 1.21 | **+17.5%** |

### Regime Performance Breakdown
```
📊 Enhanced EV-Optimized Multi-Regime Approach:
🚀 Stable Growth (Regime 0):     9 trades - STRONG (55.6% WR, $684 avg P&L, Total: $6,152)
⚡ Breakout Momentum (Regime 5): 54 trades - BEST (59.3% WR, $477 avg P&L, Total: $25,740)
✅ Defensive Stable (Regime 4):  18 trades - SOLID (50.0% WR, $183 avg P&L, Total: $3,291)
🔄 Moderate Momentum (Regime 1): 27 trades - MIXED (40.7% WR, -$65 avg P&L, Total: -$1,764)
📊 Baseline Market (Regime 2):   10 trades - WEAK (40.0% WR, -$437 avg P&L, Total: -$4,365)
🎯 Total: 118 trades with 51.7% win rate and 30.53% total return
```

## 🔧 Configuration

### Python Configuration
```python
# backtest_engine.py
INITIAL_CAPITAL = 100000
RISK_FREE_RATE = 0.0395  # 3.95% (10Y Treasury Jan 2, 2024)
REBALANCE_FREQUENCY = 'D'  # Daily
USE_EV_FILTER = False  # Performance-optimized regime allocation based on backtest results
```

### C++ Configuration
```cpp
// trading_strategy.h
constexpr double MIN_TRADE_VALUE = 100.0;
constexpr int MAX_POSITIONS = 5;
constexpr double MAX_PORTFOLIO_RISK = 0.15;
constexpr double MAX_SINGLE_POSITION_RISK = 0.05;
```

## 🧪 Research Methodology

### Feature Engineering
- **988+ Features**: Log returns, volatility, momentum, technical indicators
- **Cross-correlation**: Inter-crypto relationships and market dynamics
- **Market Stress**: VIX-style volatility measures for crypto markets

### PCA Analysis
- **25 Principal Components**: Capturing 89.68% of variance
- **PC1 (Market Factor)**: Overall market direction and beta
- **PC2 (Volatility Factor)**: Market stress and uncertainty
- **PC3-PC5**: Sector rotation and specific dynamics

### Regime Detection
- **K-means Clustering**: Optimal 7-regime classification
- **Silhouette Score**: 0.67 (good cluster separation)
- **Persistence Analysis**: Regime stability and transition probability

## 🎯 Trading Rules

### Entry Conditions
1. **Regime Filter**: Only trade BALANCED and MOMENTUM regimes
2. **Coin Scoring**: Enhanced PC-factor based attractiveness scores
3. **Risk Management**: Position sizing with volatility adjustment
4. **Minimum Thresholds**: $100 minimum trade value

### Exit Conditions
1. **Dynamic Stop Loss**: 2-12% based on regime and volatility
2. **Take Profit**: 2:1 to 3:1 risk/reward ratios
3. **Trailing Stops**: 5% trailing distance for profitable positions
4. **Regime Changes**: Automatic rebalancing on regime transitions

### Risk Management
- **Portfolio Risk**: Maximum 15% total portfolio risk
- **Position Risk**: Maximum 5% risk per position
- **Correlation Limits**: Maximum 30% in correlated assets
- **Cash Management**: Maintains liquidity for opportunities

## 📱 Usage Examples

### Python Backtesting
```python
from backtest_engine import RegimeBasedBacktester

# Initialize backtester
backtester = RegimeBasedBacktester(
    start_date='2024-01-01',
    end_date='2025-01-01',
    initial_capital=100000,
    risk_free_rate=0.0395  # 3.95% 10Y Treasury
)

# Run complete analysis
backtester.prepare_data()
results = backtester.run_backtest()
backtester.print_detailed_results()
backtester.plot_results()
```

### C++ Paper Trading
```cpp
// Initialize strategy
auto strategy = std::make_unique<ResearchBasedTradingStrategy>();

// Start trading
strategy->start();
strategy->runTradingLoop();  // Main execution loop

// Monitor performance
strategy->printStatus();
strategy->showCurrentHoldings();
```

## 🔄 Workflow

1. **Data Collection**: Fetch historical crypto prices via CoinGecko API
2. **Feature Engineering**: Generate 988+ technical and market features
3. **PCA Analysis**: Reduce dimensionality to 25 principal components
4. **Regime Classification**: K-means clustering into 7 market regimes
5. **Strategy Backtesting**: Validate performance across historical data
6. **Real-time Analysis**: Generate current regime classification
7. **Trade Execution**: C++ system executes trades based on regime signals
8. **Performance Monitoring**: Track P&L and risk metrics in real-time

## 📚 File Structure Details

### Python Research System (`crypto_cluster_pca/`)
- **`src/crypto_regime_analysis.py`**: Core regime detection and analysis
- **`src/regime_scheduler.py`**: Automated regime updates
- **`backtest_system/`**: Complete backtesting framework
  - `backtest_engine.py`: Main backtesting orchestration
  - `backtest_trading_strategy.py`: Strategy implementation
  - `backtest_data_manager.py`: Data collection and management
  - `expected_value_analyzer.py`: EV optimization analysis
  - `backtest_analysis.ipynb`: Interactive performance analysis and visualization
- **`research/`**: Jupyter notebooks for analysis and visualization
  - `01_data_collection.ipynb`: Cryptocurrency price data gathering via CoinGecko API
  - `02_feature_engineering.ipynb`: Technical indicator calculation and normalization  
  - `03_pca_analysis.ipynb`: Dimensionality reduction and component interpretation
  - `04_clustering_analysis.ipynb`: Market regime classification and analysis

### C++ Trading System (`paper_trading/`)
- **`src/trading_strategy.cpp`**: Core trading logic and position management
- **`src/regime_data.cpp`**: Regime data parsing and handling
- **`src/alpaca_client_impl.cpp`**: Alpaca API integration
- **`src/performance_tracker.cpp`**: Real-time performance analytics
- **`include/`**: Header files with class definitions

## ⚠️ Project Limitations

### Data Constraints
- **CoinGecko API Limit**: Free tier restricts historical data to **1 year maximum** (365 days)
- **Limited Cryptocurrency Coverage**: Analysis focuses on 15 major cryptocurrencies
- **Data Frequency**: Daily data only - no intraday high-frequency analysis
- **API Rate Limits**: 10-50 calls per minute depending on endpoint

### Backtesting Limitations
- **No Transaction Costs**: Backtesting assumes zero fees, spreads, or slippage
- **Perfect Execution**: Assumes all orders fill at exact target prices
- **Liquidity Assumptions**: No modeling of market impact or liquidity constraints
- **Survivorship Bias**: Only includes cryptocurrencies that existed throughout the period

### Technical Constraints
- **Paper Trading Only**: C++ system uses Alpaca paper trading (no real capital at risk)
- **Regime Stability**: Requires 30+ days of data for reliable regime classification
- **Feature Dependencies**: 988 features may be sensitive to data quality issues
- **Real-time Lag**: 15-minute delay between regime detection and trading decisions

### Market Limitations
- **Regime Persistence**: Strategy assumes regimes persist long enough for profitable trades
- **Market Structure**: Doesn't account for crypto-specific factors (staking, DeFi, etc.)
- **Correlation Assumptions**: PC analysis may break down during extreme market events
- **Time Period**: Backtesting limited to 2024-2025 bull market conditions

### System Dependencies
- **External APIs**: Relies on CoinGecko and Alpaca API availability
- **Internet Connection**: Requires stable connection for real-time data
- **Computational Requirements**: PCA and clustering analysis needs adequate memory/CPU

### Research Evolution Limitations
- **Initial Research Approach**: The project began with comprehensive PCA analysis that revealed 4 natural market regimes, but the trading system was built around a 7-regime framework before this insight was gained
- **Architecture Mismatch**: Trading strategy and backtesting systems use 7-regime classification while research notebooks show optimal 4-regime clustering
- **Documentation Inconsistency**: Some analysis sections reflect the improved 4-regime understanding while the production system maintains the original 7-regime structure
- **Future Alignment**: A complete system refactor to align with 4-regime findings would improve performance and interpretability

## 🚀 Future Improvements

**With Enhanced Data Access:**
- **Multi-Year Analysis**: 3-5 years of data for more robust regime detection
- **Higher Frequency**: Hourly/minute-level data for intraday strategies
- **Additional Assets**: Extended crypto universe (100+ cryptocurrencies)
- **Alternative Data**: Social sentiment, on-chain metrics, DeFi yields

**Advanced Features:**
- **Transaction Cost Modeling**: Realistic fees, spreads, and slippage
- **Portfolio Optimization**: Modern Portfolio Theory integration
- **Risk Attribution**: Factor-based risk decomposition
- **Live Trading**: Real capital deployment with proper risk controls

## 🚨 Risk Disclaimers

- **Educational Purpose**: This system is designed for research and learning only
- **Not Financial Advice**: Past performance does not guarantee future results
- **Market Risk**: Cryptocurrency markets are highly volatile and risky
- **Capital Risk**: Never risk more than you can afford to lose
- **Backtesting Bias**: Historical results may not reflect live trading performance

## 🤝 Contributing

This project is part of a quantitative finance internship portfolio. While not actively seeking contributions, feedback and suggestions are welcome for educational purposes.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Jake Kostoryz**
- Email: kostoryzjake@gmail.com
- LinkedIn: [linkedin.com/in/jake-kostoryz-614080213](https://www.linkedin.com/in/jake-kostoryz-614080213/)
- GitHub: [github.com/jakejk1285](https://github.com/jakejk1285)

---

## 🎓 Academic Background

This project demonstrates:
- **Quantitative Finance**: Factor models, regime detection, risk management
- **Machine Learning**: PCA, K-means clustering, feature engineering
- **Software Engineering**: Multi-language system design, API integration
- **Financial Markets**: Cryptocurrency trading, portfolio management
- **Research Methodology**: Backtesting, statistical analysis, performance attribution



