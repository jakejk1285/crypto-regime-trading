#!/usr/bin/env python3
"""
Sharpe-Optimized Trading Strategy
Focuses on improving Sharpe ratio by reducing volatility while maintaining reasonable returns
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_trading_strategy import RegimeBasedTradingStrategy
import numpy as np
from typing import Dict, Tuple


class SharpeOptimizedStrategy(RegimeBasedTradingStrategy):
    """
    Improved Sharpe-optimized version that focuses on reducing volatility while maintaining profitable trades:
    1. Enhanced regime selectivity (avoid low-EV regimes more aggressively)
    2. Dynamic position sizing based on regime EV and volatility
    3. Improved stop-loss and take-profit ratios
    4. Better trade timing with volatility-adjusted entries
    5. Portfolio concentration limits with correlation awareness
    6. Risk-adjusted position sizing that preserves good trades
    """
    
    def __init__(self, initial_capital: float = 100000, enable_decision_explanations: bool = False):
        super().__init__(initial_capital, enable_decision_explanations=enable_decision_explanations)
        
        # IMPROVED SHARPE-OPTIMIZED PARAMETERS
        self.max_single_position_risk = 0.04  # Slightly reduce from 5% to 4%
        self.max_portfolio_risk = 0.12        # Slightly reduce from 15% to 12%
        self.max_correlation_exposure = 0.40  # Slightly reduce from 50% to 40%
        
        # Moderate volatility sensitivity (not too aggressive)
        self.volatility_penalty_multiplier = 1.5  # Moderate penalty increase
        
        # Moderate position sizing caps (preserve profitability)
        self.max_position_percent = 0.35      # Reduce from 45% to 35% (less aggressive than before)
        self.min_position_percent = 0.008     # Slight increase from 1% to 0.8%
        
        print("🎯 SHARPE-OPTIMIZED STRATEGY INITIALIZED")
        print("=" * 50)
        print("🔧 Risk Reduction Features:")
        print(f"   • Max Single Position Risk: {self.max_single_position_risk:.1%}")
        print(f"   • Max Portfolio Risk: {self.max_portfolio_risk:.1%}")
        print(f"   • Max Correlation Exposure: {self.max_correlation_exposure:.1%}")
        print(f"   • Max Position Size: {self.max_position_percent:.1%}")
        print(f"   • Volatility Penalty Multiplier: {self.volatility_penalty_multiplier:.1f}x")
    
    def calculate_position_size(self, regime_data: Dict, symbol: str, coin_score: float, current_price: float) -> float:
        """
        IMPROVED SHARPE-OPTIMIZED position sizing with smarter risk controls
        """
        regime_id = regime_data['regime_id']
        strategy = regime_data['strategy']
        
        # SMARTER EV-BASED ALLOCATION
        # Focus on the most profitable regimes, moderately reduce others
        if regime_id == 3:  # EXTREME_OUTLIER - High EV (2.230R), keep substantial allocation
            base_percent = 0.35  # Reduce slightly from 40% but keep high
        elif regime_id == 5:  # BREAKOUT_MOMENTUM - Good EV (0.761R), maintain good allocation
            base_percent = 0.25  # Reduce slightly from 30% 
        elif regime_id == 0:  # STABLE_GROWTH - Positive EV (0.494R), moderate allocation
            base_percent = 0.20  # Reduce slightly from 25%
        elif regime_id == 2:  # BASELINE_MARKET - Negative EV, reduce significantly
            base_percent = 0.08  # Reduce from 15%
        elif regime_id == 1:  # MODERATE_MOMENTUM - Negative EV, reduce significantly
            base_percent = 0.05  # Reduce from 8%
        elif regime_id == 4:  # DEFENSIVE_STABLE - Very negative EV, minimal allocation
            base_percent = 0.02  # Keep minimal from 3%
        elif regime_id == 6:  # EXTREME_VOLATILITY - Very negative EV, avoid mostly
            base_percent = 0.005  # Keep minimal from 1%
        else:
            base_percent = 0.08  # Default reduced
        
        # Skip eliminated strategies
        if strategy == "WAIT_AND_SEE":
            base_percent = 0.0
        
        # BALANCED PC FACTOR STRENGTH ADJUSTMENTS
        pc1_factor = regime_data['pc1_market_factor']
        pc2_factor = regime_data['pc2_volatility_factor']
        pc3_factor = regime_data.get('pc3_factor', 0)
        
        # Moderate PC1 multipliers (preserve signal strength)
        pc1_strength = abs(pc1_factor)
        if pc1_strength > 2.0:
            pc1_multiplier = min(1.3, 1.0 + (pc1_strength - 2.0) * 0.15)  # Moderate reduction from original
        elif pc1_strength > 1.5:
            pc1_multiplier = min(1.15, 1.0 + (pc1_strength - 1.5) * 0.3)  # Preserve some strength
        else:
            pc1_multiplier = max(0.75, 1.0 - (1.5 - pc1_strength) * 0.15)  # Moderate penalty
        
        # Balanced PC2 volatility penalties (not too aggressive)
        pc2_strength = abs(pc2_factor)
        if pc2_strength > 3.0:  # Very high volatility - significant penalty
            pc2_adjustment = max(0.6, 1.0 - (pc2_strength - 3.0) * 0.2)
        elif pc2_strength > 2.0:  # High volatility - moderate penalty  
            pc2_adjustment = max(0.8, 1.0 - (pc2_strength - 2.0) * 0.1)
        elif pc2_strength < 0.8:  # Low volatility - moderate bonus
            pc2_adjustment = min(1.2, 1.0 + (0.8 - pc2_strength) * 0.3)  
        else:
            pc2_adjustment = 1.0
        
        # Moderate PC3 bonus (preserve diversification benefits)
        pc3_strength = abs(pc3_factor)
        if pc3_strength > 1.5:
            pc3_bonus = min(1.1, 1.0 + (pc3_strength - 1.5) * 0.2)  # Moderate bonus
        else:
            pc3_bonus = 1.0
        
        # Combined PC Factor Strength with reasonable bounds
        pc_strength_multiplier = pc1_multiplier * pc2_adjustment * pc3_bonus
        pc_strength_multiplier = max(0.7, min(1.4, pc_strength_multiplier))  # Reasonable range: 70%-140%
        
        # Moderate persistence bonus (preserve stability rewards)
        persistence = regime_data['persistence']
        if persistence > 0.85:  # Very stable - moderate bonus
            persistence_bonus = 1.2  # Moderate reduction from 1.3
        elif persistence > 0.65:  # Stable - small bonus  
            persistence_bonus = 1.08  # Slight reduction from 1.1
        elif persistence < 0.3:  # Very unstable - penalty
            persistence_bonus = 0.7   # Moderate penalty, not too harsh
        else:
            persistence_bonus = 1.0
        
        # BALANCED VOLATILITY PENALTIES
        market_stress = regime_data['market_stress']
        volatility_penalty = 1.0
        
        # Apply moderate volatility penalties (not too aggressive)
        if market_stress > 0.8:  # Very high stress - significant penalty
            volatility_penalty = 0.75  # Moderate penalty
        elif market_stress > 0.6:  # High stress - moderate penalty
            volatility_penalty = 0.9   # Small penalty
        elif pc2_strength > 2.5:  # Very high PC2 volatility - additional penalty
            volatility_penalty = min(volatility_penalty, 0.85)
        
        # MODERATE PORTFOLIO CONCENTRATION LIMITS
        portfolio_value = self.get_portfolio_value()
        current_positions_value = sum(pos.quantity * current_price 
                                    for pos in self.positions.values())
        current_exposure = current_positions_value / portfolio_value if portfolio_value > 0 else 0
        
        # Moderate concentration limit (not too restrictive)
        concentration_limit = max(0.6, 0.85 - current_exposure)  # Less restrictive than before
        
        # DIVERSIFICATION BONUS
        # Reward having multiple positions for better diversification
        num_positions = len(self.positions)
        diversification_bonus = 1.0
        if num_positions >= 3:
            diversification_bonus = 1.1  # Small bonus for diversification
        elif num_positions >= 5:
            diversification_bonus = 1.15  # Larger bonus for good diversification
        
        # FINAL SHARPE-OPTIMIZED POSITION SIZE CALCULATION
        adjusted_percent = (base_percent * pc_strength_multiplier * persistence_bonus * 
                           volatility_penalty * coin_score * concentration_limit * diversification_bonus)
        
        # Apply STRICT safety caps for volatility control
        adjusted_percent = max(self.min_position_percent, min(self.max_position_percent, adjusted_percent))
        
        position_value = portfolio_value * adjusted_percent
        return position_value / current_price
    
    def regime_specific_trading_rules(self, regime_data: Dict, coin_scores: Dict[str, float]) -> Dict[str, bool]:
        """
        IMPROVED SHARPE-OPTIMIZED trading rules with smarter selectivity
        """
        strategy = regime_data['strategy']
        regime_id = regime_data['regime_id']
        persistence = regime_data['persistence']
        market_stress = regime_data['market_stress']
        pc2_factor = abs(regime_data['pc2_volatility_factor'])
        
        trading_decisions = {}
        
        # SMARTER EV-BASED THRESHOLDS
        # Focus on regimes with positive EV, be more selective on negative EV regimes
        if regime_id == 3:  # EXTREME_OUTLIER - High EV (2.230R), keep low threshold
            threshold = 0.08  # Slightly higher than original 0.02 but still accessible
        elif regime_id == 5:  # BREAKOUT_MOMENTUM - Good EV (0.761R), moderate threshold
            threshold = 0.12  # Slightly higher than original 0.06
        elif regime_id == 0:  # STABLE_GROWTH - Positive EV (0.494R), moderate threshold
            threshold = 0.15  # Slightly higher than original 0.08
        elif regime_id == 2:  # BASELINE_MARKET - Negative EV, much higher threshold  
            threshold = 0.40  # Much higher than original 0.25
        elif regime_id == 1:  # MODERATE_MOMENTUM - Negative EV, high threshold
            threshold = 0.55  # Higher than original 0.45
        elif regime_id == 4:  # DEFENSIVE_STABLE - Very negative EV, very high threshold
            threshold = 0.70  # Higher than original 0.65
        elif regime_id == 6:  # EXTREME_VOLATILITY - Very negative EV, nearly eliminate
            threshold = 0.85  # Same as original
        else:
            threshold = 0.45  # Higher than original 0.35
        
        # MODERATE VOLATILITY FILTERS
        
        # Market stress penalty (moderate)
        if market_stress > 0.7:  # High stress - moderate penalty
            threshold += 0.12
        elif market_stress > 0.5:  # Moderate stress - small penalty
            threshold += 0.06
        
        # PC2 volatility penalty (moderate)
        if pc2_factor > 3.5:  # Extreme volatility
            threshold += 0.15
        elif pc2_factor > 2.5:  # Very high volatility
            threshold += 0.08
        elif pc2_factor > 2.0:  # High volatility
            threshold += 0.04
        
        # Persistence filter (moderate)
        if persistence < 0.25:  # Very unstable regime
            threshold += 0.15
        elif persistence < 0.4:  # Moderately unstable
            threshold += 0.08
        
        # Apply filters with higher selectivity
        qualified_coins = [(symbol, score) for symbol, score in coin_scores.items() 
                          if score >= threshold]
        
        # Sort by score and limit to fewer positions for better risk control
        qualified_coins.sort(key=lambda x: x[1], reverse=True)
        max_positions = self.get_sharpe_optimized_max_positions(regime_data, market_stress)
        qualified_coins = qualified_coins[:max_positions]
        
        return {symbol: True for symbol, _ in qualified_coins}
    
    def get_sharpe_optimized_max_positions(self, regime_data: Dict, market_stress: float) -> int:
        """
        Moderately conservative maximum positions for better risk control while preserving diversification
        """
        strategy = regime_data['strategy']
        
        # Moderate reduction in max positions (not too restrictive)
        if strategy == "CRISIS":
            return 3  # Keep original 3
        elif strategy == "WAIT_AND_SEE":
            return 1  # Keep reduced to 1
        elif strategy in ["MOMENTUM", "BREAKOUT"]:
            # Stress-based limits but less restrictive
            if market_stress > 0.8:  # Very high stress
                return 4  # Moderately limited
            elif market_stress > 0.6:  # High stress
                return 5  # Slightly limited
            else:
                return 6  # Moderate reduction from 8
        else:
            # Moderate stress-based limits for other strategies
            if market_stress > 0.7:  # High stress
                return 3
            elif market_stress > 0.5:  # Moderate stress
                return 4
            else:
                return 5  # Moderate reduction from 6
    
    def calculate_stop_loss_take_profit(self, regime_data: Dict, entry_price: float) -> Tuple[float, float]:
        """
        IMPROVED SHARPE-OPTIMIZED risk management with balanced risk-reward
        """
        pc2_vol = abs(regime_data['pc2_volatility_factor'])
        strategy = regime_data['strategy']
        market_stress = regime_data['market_stress']
        
        # Moderately tighter stop losses (not too aggressive)
        base_stop_loss = {
            "CRISIS": 0.05,           # Slight reduction from 0.06
            "BASELINE": 0.035,        # Moderate reduction from 0.04
            "STABLE_GROWTH": 0.035,   # Moderate reduction from 0.04
            "MOMENTUM": 0.045,        # Slight reduction from 0.05
            "BREAKOUT": 0.055,        # Slight reduction from 0.06
            "DEFENSIVE": 0.07,        # Moderate reduction from 0.08
            "EXTREME_VOLATILITY": 0.09  # Slight reduction from 0.10
        }
        
        base_stop = base_stop_loss.get(strategy, 0.045)  # Moderate reduction from 0.05
        
        # Moderate volatility adjustment (not too restrictive)
        vol_adjustment = max(0.85, min(1.4, 1.0 + (pc2_vol - 1.0) * 0.18))  # Less restrictive range
        
        # Moderate stress adjustment
        stress_adjustment = max(1.0, min(1.25, 1.0 + market_stress * 0.25))  # Moderate range
        
        # Final stop loss calculation with reasonable range
        stop_loss_pct = base_stop * vol_adjustment * stress_adjustment
        stop_loss_pct = max(0.025, min(0.10, stop_loss_pct))  # Reasonable range: 2.5-10%
        
        # Balanced take profit ratios (preserve some upside)
        risk_reward_ratios = {
            "BASELINE": 2.8,          # Slight reduction from 3.0
            "BREAKOUT": 2.3,          # Slight reduction from 2.5
            "STABLE_GROWTH": 1.9,     # Slight reduction from 2.0
            "MOMENTUM": 1.9,          # Slight reduction from 2.0
            "CRISIS": 1.5,            # Same
            "DEFENSIVE": 1.2,         # Slight improvement from 1.0
            "EXTREME_VOLATILITY": 1.0 # Same
        }
        
        risk_reward = risk_reward_ratios.get(strategy, 1.9)  # Slight reduction from 2.0
        take_profit_pct = stop_loss_pct * risk_reward
        
        stop_loss = entry_price * (1 - stop_loss_pct)
        take_profit = entry_price * (1 + take_profit_pct)
        
        return stop_loss, take_profit
    
    def should_trade_regime(self, regime_data: Dict) -> bool:
        """
        IMPROVED SHARPE-OPTIMIZED regime filter with smart volatility controls
        """
        strategy = regime_data['strategy']
        regime_id = regime_data['regime_id']
        market_stress = regime_data['market_stress']
        pc2_vol = abs(regime_data['pc2_volatility_factor'])
        
        # Always avoid uncertainty
        if strategy == "WAIT_AND_SEE":
            return False
        
        # Smart volatility filtering (focus on extreme cases)
        if pc2_vol > 4.0:  # Extreme volatility - avoid completely
            return False
        elif pc2_vol > 3.0 and market_stress > 0.7:  # Very high vol + high stress - avoid
            return False
        
        # Moderate stress limits
        if market_stress > 0.85:  # Only avoid extreme stress
            return False
        
        # Reasonable persistence requirements
        if regime_data['persistence'] < 0.35:  # Moderate requirement
            return False
        
        # Avoid very negative EV regimes in stress conditions
        if regime_id in [4, 6] and market_stress > 0.6:  # Avoid defensive/extreme vol in high stress
            return False
        elif regime_id == 6 and pc2_vol > 2.5:  # Avoid extreme volatility regime in high volatility
            return False
        
        return regime_data.get('should_trade', True)


def test_sharpe_optimized_strategy():
    """Test the Sharpe-optimized strategy and compare with original"""
    from backtest_engine import RegimeBasedBacktester
    
    print("🎯 TESTING SHARPE-OPTIMIZED STRATEGY")
    print("=" * 60)
    
    # Test Sharpe-optimized strategy
    backtester = RegimeBasedBacktester(
        start_date='2024-01-01',
        end_date='2025-01-01',
        initial_capital=100000,
        risk_free_rate=0.0395,
        enable_decision_explanations=False
    )
    
    # Replace strategy with Sharpe-optimized version
    backtester.strategy = SharpeOptimizedStrategy(100000, enable_decision_explanations=False)
    backtester.trading_strategy = backtester.strategy  # Backward compatibility
    
    print("🔧 Preparing data...")
    backtester.prepare_data()
    
    print("🚀 Running Sharpe-optimized backtest...")
    backtester.run_backtest()
    
    print("📊 Sharpe-Optimized Results:")
    print("=" * 60)
    
    backtester.print_detailed_results()
    
    # Get performance summary
    summary = backtester.strategy.get_performance_summary()
    
    return summary


if __name__ == "__main__":
    try:
        results = test_sharpe_optimized_strategy()
        print("\n✅ Sharpe-optimized strategy test complete!")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()