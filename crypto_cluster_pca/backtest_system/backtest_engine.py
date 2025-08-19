#!/usr/bin/env python3
"""
Main Backtesting Engine
Orchestrates the complete backtesting process using your exact trading strategy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

from backtest_data_manager import CryptoDataManager
from backtest_trading_strategy import RegimeBasedTradingStrategy


class RegimeBasedBacktester:
    """
    Complete backtesting engine for your regime-based cryptocurrency trading strategy
    """

    def __init__(self, start_date: str = '2024-01-01', end_date: str = '2025-01-01',
                 initial_capital: float = 100000, risk_free_rate: float = 0.0427):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate  # Annual risk-free rate (default 4.27% as of Aug 12, 2025)

        # Initialize components
        self.data_manager = CryptoDataManager(start_date, end_date)
        self.trading_strategy = RegimeBasedTradingStrategy(initial_capital)

        # Data storage
        self.crypto_data = {}
        self.regime_data = None
        self.backtest_results = {}

    def prepare_data(self, force_refresh: bool = False) -> None:
        """
        Prepare all data needed for backtesting
        """
        print("🔧 PREPARING BACKTEST DATA")
        print("=" * 50)

        # Collect cryptocurrency data
        print("1. Collecting cryptocurrency price data...")
        self.crypto_data = self.data_manager.collect_historical_crypto_data(force_refresh)
        self.data_manager.crypto_data_cache = self.crypto_data  # Cache for price lookups

        # Create regime features
        print("2. Creating regime features...")
        features_df = self.data_manager.create_historical_regime_features(self.crypto_data)

        # Generate historical regime assignments
        print("3. Generating historical regime assignments...")
        self.regime_data = self.data_manager.generate_historical_regimes(features_df, force_refresh)

        print("✅ Data preparation complete!")

    def run_backtest(self, rebalance_frequency: str = 'D') -> Dict:
        """
        Run the complete backtesting simulation
        """
        print(f"\n🚀 STARTING REGIME-BASED BACKTEST")
        print("=" * 60)
        print(f"📅 Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"💰 Initial Capital: ${self.initial_capital:,.2f}")
        print(f"🔄 Rebalance Frequency: {rebalance_frequency}")
        print("=" * 60)

        if self.regime_data is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        # Get trading timestamps based on rebalance frequency
        all_timestamps = self.regime_data.index
        
        # Filter timestamps based on rebalance frequency
        if rebalance_frequency == 'D' or rebalance_frequency == '1D':
            # Daily frequency - use all available daily timestamps
            trading_timestamps = all_timestamps
        elif rebalance_frequency == 'W' or rebalance_frequency == '1W':
            # Weekly frequency - trade once per week (Mondays)
            trading_timestamps = all_timestamps[all_timestamps.dayofweek == 0]
        elif rebalance_frequency == 'M' or rebalance_frequency == '1M':
            # Monthly frequency - trade on first day of each month
            trading_timestamps = all_timestamps[all_timestamps.to_series().dt.is_month_start]
        else:
            # Default to daily for any unrecognized frequency
            print(f"⚠️ Warning: Unrecognized frequency '{rebalance_frequency}', defaulting to daily")
            trading_timestamps = all_timestamps
            
        total_timestamps = len(trading_timestamps)
        
        print(f"📊 Total trading timestamps: {total_timestamps:,} ({rebalance_frequency} frequency)")
        print(f"📊 Original data points: {len(all_timestamps):,}")

        # Track progress
        progress_interval = max(1000, total_timestamps // 50)  # Show progress every 2%

        # Run backtesting simulation
        for i, timestamp in enumerate(trading_timestamps):

            # Show progress
            if i % progress_interval == 0 or i == total_timestamps - 1:
                progress_pct = (i + 1) / total_timestamps * 100
                print(f"⏳ Progress: {progress_pct:.1f}% ({i + 1:,}/{total_timestamps:,})")

            # Get regime data for current timestamp
            regime_row = self.regime_data.loc[timestamp]
            regime_data = {
                'regime_id': int(regime_row['regime_id']),
                'strategy': regime_row['strategy'],
                'pc1_market_factor': regime_row['pc1_market_factor'],
                'pc2_volatility_factor': regime_row['pc2_volatility_factor'],
                'pc3_factor': regime_row['pc3_factor'],
                'pc4_factor': regime_row['pc4_factor'],
                'pc5_factor': regime_row['pc5_factor'],
                'persistence': regime_row['persistence'],
                'avg_duration': regime_row['avg_duration'],
                'frequency_percentage': regime_row['frequency_percentage'],
                'risk_multiplier': regime_row['risk_multiplier'],
                'market_stress': regime_row['market_stress'],
                'should_trade': regime_row['should_trade']
            }

            # Get current prices for all cryptocurrencies
            current_prices = {}
            for symbol in self.trading_strategy.crypto_coins.keys():
                if symbol in self.crypto_data:
                    price_data = self.data_manager.get_crypto_prices(symbol, timestamp)
                    current_prices[symbol] = price_data['close']

            # Execute trading cycle
            if current_prices:  # Only trade if we have price data
                self.trading_strategy.execute_trading_cycle(regime_data, current_prices, timestamp)
                self.trading_strategy.update_equity_curve(timestamp, current_prices)

        # Generate final results
        self.backtest_results = self.trading_strategy.get_performance_summary()
        self.backtest_results['start_date'] = self.start_date
        self.backtest_results['end_date'] = self.end_date
        self.backtest_results['duration_days'] = (self.end_date - self.start_date).days
        
        # Calculate and add Sharpe ratio to results
        sharpe_metrics = self.calculate_sharpe_ratio()
        self.backtest_results.update(sharpe_metrics)

        print("\n✅ BACKTESTING COMPLETE!")
        return self.backtest_results

    def calculate_benchmark_performance(self) -> Dict:
        """
        Calculate buy-and-hold benchmark performance for comparison
        """
        print("📈 Calculating benchmark (Bitcoin Buy & Hold) performance...")

        if 'BTCUSD' not in self.crypto_data:
            return {'benchmark_return': 0.0}

        btc_data = self.crypto_data['BTCUSD']
        start_price = btc_data.iloc[0]['close']
        end_price = btc_data.iloc[-1]['close']

        benchmark_return = (end_price - start_price) / start_price * 100

        print(f"📊 Bitcoin Buy & Hold: {benchmark_return:.2f}%")

        return {
            'benchmark_return': benchmark_return,
            'benchmark_start_price': start_price,
            'benchmark_end_price': end_price
        }

    def calculate_sharpe_ratio(self) -> Dict:
        """
        Calculate Sharpe ratio using industry standard methodology (252 trading days)
        """
        if not self.backtest_results or not self.backtest_results.get('equity_curve'):
            return {'sharpe_ratio': 0.0, 'annualized_volatility': 0.0, 'annualized_return': 0.0}
        
        equity_curve = self.backtest_results['equity_curve']
        if len(equity_curve) < 2:
            return {'sharpe_ratio': 0.0, 'annualized_volatility': 0.0, 'annualized_return': 0.0}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(equity_curve)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate returns as decimals (not percentages for volatility calculation)
        returns = df['portfolio_value'].pct_change().dropna()
        
        if returns.empty or returns.std() == 0:
            return {'sharpe_ratio': 0.0, 'annualized_volatility': 0.0, 'annualized_return': 0.0}
        
        # Industry standard: Calculate compound annualized return
        initial_value = df['portfolio_value'].iloc[0]
        final_value = df['portfolio_value'].iloc[-1]
        annualized_return = (final_value / initial_value) ** (252 / len(returns)) - 1
        
        # Industry standard: Annualized volatility using 252 trading days (convert to percentage)
        annualized_volatility = returns.std() * np.sqrt(252) * 100
        
        # Calculate Sharpe ratio (risk-free rate already in percentage)
        sharpe_ratio = (annualized_return * 100 - self.risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0.0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'risk_free_rate': self.risk_free_rate,
            'data_frequency': "252 trading days (industry standard)"
        }

    def print_detailed_results(self) -> None:
        """
        Print comprehensive backtesting results
        """
        if not self.backtest_results:
            print("❌ No backtest results available. Run backtest first.")
            return

        # Strategy performance
        self.trading_strategy.print_performance_summary()

        # Benchmark comparison
        benchmark = self.calculate_benchmark_performance()
        strategy_return = self.backtest_results['total_return']

        print(f"\n📊 STRATEGY vs BENCHMARK COMPARISON")
        print("-" * 50)
        print(f"🤖 Strategy Return: {strategy_return:.2f}%")
        print(f"₿  Bitcoin Buy & Hold: {benchmark['benchmark_return']:.2f}%")

        if benchmark['benchmark_return'] != 0:
            alpha = strategy_return - benchmark['benchmark_return']
            print(f"⚡ Alpha (Excess Return): {alpha:.2f}%")

        # Additional metrics
        print(f"\n⏰ TIMING METRICS")
        print("-" * 30)
        print(f"📅 Duration: {self.backtest_results['duration_days']} days")
        print(
            f"🔄 Trades per Month: {self.backtest_results['num_trades'] / (self.backtest_results['duration_days'] / 30):.1f}")

        # Risk metrics using dedicated Sharpe ratio calculation
        if self.backtest_results.get('sharpe_ratio') is not None:
            print(f"\n📊 RISK METRICS")
            print("-" * 25)
            print(f"📊 Annualized Return: {self.backtest_results['annualized_return'] * 100:.2f}%")
            print(f"📊 Annualized Volatility: {self.backtest_results['annualized_volatility'] * 100:.2f}%")
            print(f"⚡ Sharpe Ratio: {self.backtest_results['sharpe_ratio']:.3f}")
            print(f"🏦 Risk-Free Rate: {self.backtest_results['risk_free_rate'] * 100:.2f}%")
            if 'data_frequency' in self.backtest_results:
                print(f"⏰ Data Frequency: {self.backtest_results['data_frequency']}")

    def plot_results(self, save_plots: bool = True) -> None:
        """
        Create comprehensive visualization of backtest results
        """
        if not self.backtest_results or not self.backtest_results['equity_curve']:
            print("❌ No data available for plotting")
            return

        print("📊 Creating backtest visualization...")

        # Prepare data for plotting
        equity_df = pd.DataFrame(self.backtest_results['equity_curve'])
        equity_df.set_index('timestamp', inplace=True)

        # Create subplot layout
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Regime-Based Cryptocurrency Trading Strategy - Backtest Results', fontsize=16, fontweight='bold')

        # 1. Equity Curve
        ax1 = axes[0, 0]
        ax1.plot(equity_df.index, equity_df['portfolio_value'], linewidth=2, color='blue', label='Strategy')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')

        # Add benchmark (Bitcoin)
        if 'BTCUSD' in self.crypto_data:
            btc_data = self.crypto_data['BTCUSD']
            btc_normalized = btc_data['close'] / btc_data.iloc[0]['close'] * self.initial_capital
            btc_aligned = btc_normalized.reindex(equity_df.index, method='ffill')
            ax1.plot(btc_aligned.index, btc_aligned.values, linewidth=1.5, color='orange', alpha=0.8,
                     label='Bitcoin Buy & Hold')

        ax1.set_title('Portfolio Equity Curve', fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Regime Distribution Over Time
        ax2 = axes[0, 1]
        regime_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink']
        for regime_id in range(7):
            regime_mask = equity_df['regime_id'] == regime_id
            if regime_mask.any():
                ax2.scatter(equity_df.index[regime_mask],
                            equity_df['portfolio_value'][regime_mask],
                            c=regime_colors[regime_id], alpha=0.6, s=10, label=f'Regime {regime_id}')

        ax2.set_title('Regime Distribution Over Time', fontweight='bold')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        # 3. Monthly Returns Heatmap
        ax3 = axes[1, 0]
        equity_df_monthly = equity_df.resample('M').last()
        monthly_returns = equity_df_monthly['portfolio_value'].pct_change() * 100

        if len(monthly_returns) > 1:
            monthly_returns_pivot = monthly_returns.groupby(
                [monthly_returns.index.year, monthly_returns.index.month]).first().unstack(fill_value=0)

            if not monthly_returns_pivot.empty:
                sns.heatmap(monthly_returns_pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax3)
                ax3.set_title('Monthly Returns Heatmap (%)', fontweight='bold')
                ax3.set_ylabel('Year')
                ax3.set_xlabel('Month')

        # 4. Drawdown Analysis
        ax4 = axes[1, 1]
        equity_values = equity_df['portfolio_value'].values
        peak = equity_values[0]
        drawdowns = []

        for value in equity_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            drawdowns.append(drawdown)

        ax4.fill_between(equity_df.index, 0, drawdowns, color='red', alpha=0.3)
        ax4.plot(equity_df.index, drawdowns, color='red', linewidth=1)
        ax4.set_title('Drawdown Analysis', fontweight='bold')
        ax4.set_ylabel('Drawdown (%)')
        ax4.invert_yaxis()
        ax4.grid(True, alpha=0.3)

        # 5. Trade Distribution by Regime
        ax5 = axes[2, 0]
        regime_names = ['STABLE_GROWTH', 'MODERATE_MOMENTUM', 'BASELINE_MARKET',
                        'EXTREME_OUTLIER', 'DEFENSIVE_STABLE', 'BREAKOUT_MOMENTUM', 'EXTREME_VOLATILITY']

        regime_trade_counts = []
        regime_labels = []

        for regime_id in range(7):
            count = self.trading_strategy.regime_trades[regime_id]
            if count > 0:
                regime_trade_counts.append(count)
                regime_labels.append(f'R{regime_id}\n{regime_names[regime_id][:8]}')

        if regime_trade_counts:
            bars = ax5.bar(regime_labels, regime_trade_counts, color=regime_colors[:len(regime_trade_counts)])
            ax5.set_title('Trades by Regime', fontweight='bold')
            ax5.set_ylabel('Number of Trades')
            plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax5.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        # 6. Performance Metrics Summary
        ax6 = axes[2, 1]
        ax6.axis('off')

        # Create performance summary text
        sharpe_text = f"Sharpe Ratio: {self.backtest_results.get('sharpe_ratio', 0):.3f}" if 'sharpe_ratio' in self.backtest_results else "Sharpe Ratio: N/A"
        
        # Calculate alpha vs benchmark
        benchmark = self.calculate_benchmark_performance()
        strategy_return = self.backtest_results['total_return']
        alpha = strategy_return - benchmark.get('benchmark_return', 0)
        alpha_text = f"Alpha vs BTC: {alpha:.2f}%"
        
        summary_text = f"""
        PERFORMANCE SUMMARY

        Total Return: {self.backtest_results['total_return']:.2f}%
        Total P&L: ${self.backtest_results['total_pnl']:,.2f}
        {alpha_text}

        Total Trades: {self.backtest_results['num_trades']:,}
        Win Rate: {self.backtest_results['win_rate']:.1f}%

        Profit Factor: {self.backtest_results['profit_factor']:.2f}
        Max Drawdown: {self.backtest_results['max_drawdown']:.2f}%
        {sharpe_text}

        Avg Win: ${self.backtest_results['avg_win']:.2f}
        Avg Loss: ${self.backtest_results['avg_loss']:.2f}

        Duration: {self.backtest_results['duration_days']} days
        Trades/Month: {self.backtest_results['num_trades'] / (self.backtest_results['duration_days'] / 30):.1f}
        """

        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        plt.tight_layout()

        if save_plots:
            filename = f"regime_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📁 Plot saved as: {filename}")

        plt.show()


def main():
    """
    Main execution function - run complete backtesting analysis
    """
    print("🔬 REGIME-BASED CRYPTOCURRENCY BACKTESTING SYSTEM")
    print("=" * 70)
    print("🎯 Using your exact trading strategy from crypto_regime_analysis.py")
    print("📊 Backtesting period: 2024-2025 (1 year, 366 data points)")
    print("💰 Initial capital: $100,000")
    print("🏦 Risk-free rate: 4.27% (as of Aug 12, 2025)")
    print("📊 Data source: CoinGecko API (1-year limit)")
    print("=" * 70)

    # Initialize backtester
    backtester = RegimeBasedBacktester(
        start_date='2024-01-01',
        end_date='2025-01-01',
        initial_capital=100000
    )

    print("\n🔧 Step 1: Preparing historical data...")
    backtester.prepare_data(force_refresh=False)  # Set to True to refresh cached data

    print("\n🚀 Step 2: Running backtest simulation...")
    results = backtester.run_backtest(rebalance_frequency='D')

    print("\n📊 Step 3: Analyzing results...")
    backtester.print_detailed_results()

    print("\n📈 Step 4: Creating visualizations...")
    backtester.plot_results(save_plots=True)

    print("\n🎉 BACKTESTING ANALYSIS COMPLETE!")
    print("=" * 50)
    print("💡 Key Insights:")
    print(f"   📈 Strategy delivered {results['total_return']:.2f}% return")
    print(f"   🔄 Executed {results['num_trades']} trades with {results['win_rate']:.1f}% win rate")
    print(f"   📉 Maximum drawdown was {results['max_drawdown']:.2f}%")
    print(f"   ⚡ Sharpe ratio: {results['sharpe_ratio']:.3f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
