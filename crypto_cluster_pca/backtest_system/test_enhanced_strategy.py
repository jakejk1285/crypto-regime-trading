#!/usr/bin/env python3
"""
Quick test of enhanced strategy performance with minimal logging
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_engine import RegimeBasedBacktester

def test_enhanced_strategy():
    """Test enhanced strategy with minimal decision logging"""
    print("🚀 TESTING ENHANCED STRATEGY PERFORMANCE")
    print("=" * 60)
    
    # Initialize backtester with decision explanations disabled during run
    backtester = RegimeBasedBacktester(
        start_date='2024-01-01',
        end_date='2025-01-01',
        initial_capital=100000,
        risk_free_rate=0.0395,
        enable_decision_explanations=False  # Disable verbose logging for speed
    )
    
    print("🔧 Preparing data...")
    backtester.prepare_data()
    
    print("🚀 Running enhanced backtest...")
    backtester.run_backtest()
    
    print("📊 Enhanced Strategy Results:")
    print("=" * 60)
    
    # Print comprehensive results
    backtester.print_detailed_results()
    
    # Get performance summary
    summary = backtester.strategy.get_performance_summary()
    
    print("\n⚡ KEY PERFORMANCE METRICS:")
    print("-" * 40)
    print(f"📈 Total Return: {summary['total_return']:.2f}%")
    print(f"📊 Sharpe Ratio: {summary['sharpe_ratio']:.3f}")
    print(f"📉 Max Drawdown: {summary['max_drawdown']:.2f}%")
    print(f"🎯 Win Rate: {summary['win_rate']:.1f}%")
    print(f"💵 Total P&L: ${summary['total_pnl']:,.2f}")
    print(f"🔄 Total Trades: {summary['num_trades']}")
    
    return summary

if __name__ == "__main__":
    try:
        results = test_enhanced_strategy()
        print("\n✅ Enhanced strategy test complete!")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()