# Changelog

## [1.2.0] - 2025-01-XX - Sharpe Ratio Optimization

### Added
- **Sharpe-Optimized Trading Strategy** (`sharpe_optimized_strategy.py`)
  - 28.5% improvement in Sharpe ratio (0.621 → 0.798)
  - 19.2% reduction in portfolio volatility (29.52% → 23.85%)
  - Smart regime selectivity based on Expected Value analysis
  - Balanced risk controls preserving profitable trades
  - Enhanced position sizing with volatility awareness

- **Strategy Comparison Framework** (`strategy_comparison.py`)
  - Comprehensive performance comparison between strategies
  - Detailed risk-adjusted metrics analysis
  - Trade-off evaluation and recommendations
  - Risk reduction analysis and volatility breakdown

- **Enhanced Testing Suite** (`test_enhanced_strategy.py`)
  - Quick testing framework for strategy validation
  - Performance benchmarking capabilities
  - Minimal logging for focused testing

### Enhanced
- **README.md**
  - Added Sharpe-optimized strategy results and comparison table
  - Updated Enhanced Strategy Features with optimization details
  - Improved performance metrics documentation
  - Added risk-adjusted return calculations

- **Trading Strategy** (`backtest_trading_strategy.py`)
  - Enhanced EV-based position sizing algorithms
  - Improved PC factor strength integration
  - Better volatility penalty calculations
  - Refined regime-specific trading rules

### Performance Improvements
- **Sharpe Ratio**: 0.621 → 0.798 (+28.5%)
- **Volatility**: 29.52% → 23.85% (-19.2%)
- **Max Drawdown**: 17.23% → 15.12% (-12.2%)
- **Win Rate**: 51.7% → 53.2% (+2.9%)
- **Risk-Adjusted Return**: 1.03 → 1.21 (+17.5%)

### Technical Details
- Enhanced regime filtering for negative-EV regimes
- Moderate position sizing caps (45% → 35% max position)
- Improved stop-loss and take-profit ratios
- Better portfolio diversification controls
- Volatility-aware PC factor multipliers (1.5x penalty multiplier)

---

## [1.1.0] - 2025-01-XX - Enhanced Strategy Implementation

### Added
- Expected Value (EV) optimization with regime-specific allocation
- PC factor strength integration for dynamic position sizing
- Decision explanation system with comprehensive logging
- Enhanced performance tracking and regime attribution

### Performance Results
- **Total Return**: 30.53% (improved from 21.12%)
- **Sharpe Ratio**: 0.621
- **Win Rate**: 51.7%
- **Total Trades**: 118

---

## [1.0.0] - 2024-XX-XX - Initial Release

### Added
- Regime-based cryptocurrency trading strategy
- 7-regime market classification using PCA and K-means
- Python backtesting engine with historical data analysis
- C++ paper trading system with Alpaca integration
- Comprehensive documentation and analysis notebooks