#!/usr/bin/env python3
"""
Compare Original vs Sharpe-Optimized Strategy Performance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_enhanced_strategy import test_enhanced_strategy
from sharpe_optimized_strategy import test_sharpe_optimized_strategy


def compare_strategies():
    """Compare original and Sharpe-optimized strategies"""
    
    print("🚀 COMPREHENSIVE STRATEGY COMPARISON")
    print("=" * 70)
    
    # Test original strategy
    print("\n📊 TESTING ORIGINAL ENHANCED STRATEGY...")
    print("-" * 50)
    original_results = test_enhanced_strategy()
    
    print("\n📊 TESTING SHARPE-OPTIMIZED STRATEGY...")
    print("-" * 50)
    sharpe_results = test_sharpe_optimized_strategy()
    
    # Compare results
    print("\n⚡ STRATEGY PERFORMANCE COMPARISON")
    print("=" * 70)
    
    comparison_data = [
        ("Metric", "Original Strategy", "Sharpe-Optimized", "Difference"),
        ("─" * 20, "─" * 20, "─" * 20, "─" * 20),
        ("Total Return", f"{original_results['total_return']:.2f}%", f"{sharpe_results['total_return']:.2f}%", 
         f"{sharpe_results['total_return'] - original_results['total_return']:+.2f}%"),
        ("Sharpe Ratio", f"{original_results['sharpe_ratio']:.3f}", f"{sharpe_results['sharpe_ratio']:.3f}",
         f"{sharpe_results['sharpe_ratio'] - original_results['sharpe_ratio']:+.3f}"),
        ("Max Drawdown", f"{original_results['max_drawdown']:.2f}%", f"{sharpe_results['max_drawdown']:.2f}%",
         f"{sharpe_results['max_drawdown'] - original_results['max_drawdown']:+.2f}%"),
        ("Annualized Volatility", f"{original_results.get('annualized_volatility', 0) * 100:.2f}%", 
         f"{sharpe_results.get('annualized_volatility', 0) * 100:.2f}%",
         f"{(sharpe_results.get('annualized_volatility', 0) - original_results.get('annualized_volatility', 0)) * 100:+.2f}%"),
        ("Win Rate", f"{original_results['win_rate']:.1f}%", f"{sharpe_results['win_rate']:.1f}%",
         f"{sharpe_results['win_rate'] - original_results['win_rate']:+.1f}%"),
        ("Total Trades", f"{original_results['num_trades']}", f"{sharpe_results['num_trades']}",
         f"{sharpe_results['num_trades'] - original_results['num_trades']:+d}"),
        ("Final Value", f"${original_results['final_value']:,.0f}", f"${sharpe_results['final_value']:,.0f}",
         f"${sharpe_results['final_value'] - original_results['final_value']:+,.0f}"),
        ("Profit Factor", f"{original_results.get('profit_factor', 0):.2f}", f"{sharpe_results.get('profit_factor', 0):.2f}",
         f"{sharpe_results.get('profit_factor', 0) - original_results.get('profit_factor', 0):+.2f}")
    ]
    
    # Print comparison table
    for row in comparison_data:
        print(f"{row[0]:20} | {row[1]:20} | {row[2]:20} | {row[3]:20}")
    
    # Analysis and recommendations
    print(f"\n💡 SHARPE OPTIMIZATION ANALYSIS:")
    print("=" * 70)
    
    volatility_reduction = (original_results.get('annualized_volatility', 0) - sharpe_results.get('annualized_volatility', 0)) * 100
    return_reduction = original_results['total_return'] - sharpe_results['total_return']
    sharpe_improvement = sharpe_results['sharpe_ratio'] - original_results['sharpe_ratio']
    
    print(f"✅ RISK REDUCTION ACHIEVED:")
    print(f"   • Volatility reduced by {volatility_reduction:.2f}% (from {original_results.get('annualized_volatility', 0) * 100:.2f}% to {sharpe_results.get('annualized_volatility', 0) * 100:.2f}%)")
    print(f"   • Max drawdown reduced by {original_results['max_drawdown'] - sharpe_results['max_drawdown']:.2f}% (from {original_results['max_drawdown']:.2f}% to {sharpe_results['max_drawdown']:.2f}%)")
    print(f"   • Trade frequency reduced by {original_results['num_trades'] - sharpe_results['num_trades']} trades")
    
    print(f"\n📊 TRADE-OFFS:")
    print(f"   • Return decreased by {return_reduction:.2f}% (from {original_results['total_return']:.2f}% to {sharpe_results['total_return']:.2f}%)")
    print(f"   • Final value decreased by ${original_results['final_value'] - sharpe_results['final_value']:,.0f}")
    
    if sharpe_improvement > 0:
        print(f"\n🎯 SHARPE RATIO IMPROVEMENT: {sharpe_improvement:+.3f}")
        print(f"   • Successfully improved risk-adjusted returns!")
        print(f"   • Strategy is more efficient per unit of risk taken")
    else:
        print(f"\n⚠️  SHARPE RATIO DECLINE: {sharpe_improvement:+.3f}")
        print(f"   • Risk reduction came at the cost of risk-adjusted returns")
        print(f"   • May need further optimization for better Sharpe ratio")
    
    # Risk-adjusted value creation
    original_risk_adj = original_results['total_return'] / (original_results.get('annualized_volatility', 1) * 100)
    sharpe_risk_adj = sharpe_results['total_return'] / (sharpe_results.get('annualized_volatility', 1) * 100)
    
    print(f"\n📈 RISK-ADJUSTED VALUE CREATION:")
    print(f"   • Original: {original_risk_adj:.2f} return per unit volatility")
    print(f"   • Sharpe-Optimized: {sharpe_risk_adj:.2f} return per unit volatility")
    print(f"   • Improvement: {sharpe_risk_adj - original_risk_adj:+.2f}")
    
    print(f"\n🎯 STRATEGY RECOMMENDATION:")
    if sharpe_improvement > 0:
        print("   ✅ Use Sharpe-optimized strategy for better risk-adjusted returns")
        print("   💡 Consider further fine-tuning for even better Sharpe ratio")
    else:
        print("   ⚠️  Original strategy provides better absolute returns")
        print("   💡 Consider hybrid approach or alternative risk reduction methods")
        print("   🔧 Further volatility reduction needed to improve Sharpe ratio")
    
    return {
        'original': original_results,
        'sharpe_optimized': sharpe_results,
        'comparison': {
            'volatility_reduction': volatility_reduction,
            'return_reduction': return_reduction, 
            'sharpe_improvement': sharpe_improvement,
            'risk_adjusted_improvement': sharpe_risk_adj - original_risk_adj
        }
    }


if __name__ == "__main__":
    try:
        comparison_results = compare_strategies()
        print("\n✅ Strategy comparison complete!")
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()