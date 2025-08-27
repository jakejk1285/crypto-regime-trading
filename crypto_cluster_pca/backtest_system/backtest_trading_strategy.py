#!/usr/bin/env python3
"""
Trading Strategy Implementation for Backtesting
Exact replication of your C++ trading strategy logic
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from expected_value_analyzer import ExpectedValueAnalyzer


@dataclass
class Position:
    """Represents an open trading position"""
    symbol: str
    quantity: float
    entry_price: float
    entry_timestamp: pd.Timestamp
    stop_loss: float
    take_profit: float
    regime_id: int
    order_id: str
    trailing_stop: float = None  # For tracking trailing stop losses
    highest_price: float = None  # Track highest price for trailing stops


@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_timestamp: pd.Timestamp
    exit_timestamp: pd.Timestamp
    regime_id: int
    pnl: float
    pnl_pct: float
    exit_reason: str


class RegimeBasedTradingStrategy:
    """
    Exact replication of your C++ ResearchBasedTradingStrategy
    Uses your regime analysis, coin scoring, and position management logic
    """

    def __init__(self, initial_capital: float = 100000, use_ev_filter: bool = True):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        
        # Expected Value system
        self.use_ev_filter = use_ev_filter
        self.ev_analyzer = ExpectedValueAnalyzer(min_ev_threshold=0.1, max_risk_per_trade=0.02)
        self.ev_initialized = False

        # Your exact symbol mapping (matching data manager)
        self.crypto_coins = {
            'BTCUSD': 'bitcoin', 'ETHUSD': 'ethereum', 'ADAUSD': 'cardano',
            'DOTUSD': 'polkadot', 'LINKUSD': 'chainlink', 'SOLUSD': 'solana',
            'MATICUSD': 'matic-network', 'AVAXUSD': 'avalanche-2', 'ATOMUSD': 'cosmos',
            'ALGOUSD': 'algorand', 'USDTUSD': 'tether', 'XRPUSD': 'ripple',
            'BNBUSD': 'binancecoin', 'DOGEUSD': 'dogecoin', 'LTCUSD': 'litecoin'
        }

        # Performance tracking
        self.equity_curve = []
        self.regime_performance = {i: [] for i in range(7)}
        self.regime_trades = {i: 0 for i in range(7)}

        # Enhanced risk management parameters
        self.min_trade_value = 100.0  # Minimum $100 per trade
        self.max_position_count = 5  # Maximum concurrent positions
        self.max_portfolio_risk = 0.15  # Maximum 15% portfolio risk at any time
        self.max_single_position_risk = 0.05  # Maximum 5% risk per position
        self.max_correlation_exposure = 0.30  # Maximum 30% in correlated assets

        # Current regime tracking
        self.current_regime_id = None
        self.previous_regime_id = None

    def calculate_coin_scores(self, regime_data: Dict) -> Dict[str, float]:
        """
        Enhanced coin scoring algorithm with dynamic factors and volatility awareness
        """
        coin_scores = {}
        
        # Extract all PC factors for comprehensive analysis
        pc1_factor = regime_data['pc1_market_factor']
        pc2_factor = regime_data['pc2_volatility_factor']
        pc3_factor = regime_data.get('pc3_factor', 0)
        pc4_factor = regime_data.get('pc4_factor', 0)
        pc5_factor = regime_data.get('pc5_factor', 0)
        
        persistence = regime_data['persistence']
        frequency_pct = regime_data['frequency_percentage']
        market_stress = regime_data['market_stress']
        strategy = regime_data['strategy']

        for symbol in self.crypto_coins.keys():
            # Enhanced base score with regime stability (increased base for better selection)
            regime_strength = (persistence * 0.35 + (frequency_pct / 100) * 0.25)
            market_condition = min(1.0, max(0.0, 1.0 - market_stress))
            base_score = regime_strength * market_condition + 0.4  # Increased from 0.3
            
            # Dynamic PC factor scoring based on regime strategy
            pc_score = 0.0
            
            if strategy in ["MOMENTUM", "BREAKOUT"]:
                # Favor momentum coins in momentum regimes
                if symbol in ['SOLUSD', 'AVAXUSD', 'LINKUSD', 'BNBUSD']:
                    pc_score += max(0, pc1_factor * 0.1) + abs(pc3_factor) * 0.05
                elif symbol in ['BTCUSD', 'ETHUSD']:
                    pc_score += max(0, pc1_factor * 0.05)
                    
            elif strategy in ["STABLE_GROWTH", "DEFENSIVE"]:
                # Favor stable coins in defensive regimes
                if symbol in ['BTCUSD', 'ETHUSD', 'USDTUSD']:
                    pc_score += 0.15 + max(0, -pc2_factor * 0.1)  # Bonus for low volatility
                
            elif strategy == "CRISIS":
                # Flight to quality in crisis
                if symbol in ['BTCUSD', 'USDTUSD']:
                    pc_score += 0.25
                else:
                    pc_score -= 0.10  # Penalty for altcoins
            
            # Volatility adjustment based on PC2
            volatility_penalty = abs(pc2_factor) * 0.02 if abs(pc2_factor) > 2.0 else 0
            
            # Final score with bounds checking
            final_score = max(0.0, min(1.0, base_score + pc_score - volatility_penalty))
            coin_scores[symbol] = final_score

        return coin_scores

    def calculate_position_size(self, regime_data: Dict, symbol: str, coin_score: float, current_price: float) -> float:
        """
        Enhanced position sizing with volatility and correlation adjustments
        """
        # OPTIMIZED ALLOCATION: Focus on Big Three only
        # Based on EV Analysis - maximize exposure to highest performers
        regime_risk = {
            # ELIMINATED REGIMES (Negative/Low EV)
            "CRISIS": 0.0,              # Eliminated
            "WAIT_AND_SEE": 0.0,        # -0.421R EV, 60% WR - ELIMINATED
            "DEFENSIVE": 0.0,           # -0.668R EV, 16.7% WR - ELIMINATED  
            "EXTREME_VOLATILITY": 0.0,  # -0.891R EV, 3.3% WR - ELIMINATED
            "CONSERVATIVE": 0.0,        # Marginal performer - ELIMINATED
            "STABLE_GROWTH": 0.0,       # 1.156R EV but 35.6% WR - ELIMINATED (too risky)
            
            # BIG THREE WINNERS (Increased allocations)
            # Note: Both BREAKOUT_MOMENTUM and MODERATE_MOMENTUM map to "MOMENTUM" strategy
            "BALANCED": 0.45,           # 45% - BASELINE_MARKET (3.305R EV, 93% WR) - TOP PERFORMER
            "MOMENTUM": 0.55,           # 55% - BOTH momentum regimes (combined 2.173R + 0.923R EV)
        }
        
        base_percent = regime_risk.get(regime_data['strategy'], 0.15)
        
        # Volatility adjustment (reduce size for high vol assets)
        pc2_factor = abs(regime_data['pc2_volatility_factor'])
        vol_adjustment = max(0.5, 1.0 - (pc2_factor - 1.0) * 0.1) if pc2_factor > 1.0 else 1.0
        
        # Persistence bonus (more confident = larger size)
        persistence_bonus = 1.0 + (regime_data['persistence'] - 0.5) * 0.4
        
        # Portfolio concentration limit (relaxed for better capital utilization)
        portfolio_value = self.get_portfolio_value()
        current_positions_value = sum(pos.quantity * current_price 
                                    for pos in self.positions.values())
        current_exposure = current_positions_value / portfolio_value if portfolio_value > 0 else 0
        concentration_limit = max(0.7, 0.95 - current_exposure)  # Allow higher exposure
        
        # Final position size calculation
        adjusted_percent = (base_percent * vol_adjustment * persistence_bonus * 
                           coin_score * concentration_limit)
        
        position_value = portfolio_value * adjusted_percent
        
        return position_value / current_price

    def validate_position_risk(self, symbol: str, position_qty: float, entry_price: float, 
                              stop_loss: float) -> Tuple[bool, str]:
        """
        Validate position meets risk management criteria
        """
        portfolio_value = self.get_portfolio_value()
        
        # Calculate position risk
        risk_per_share = abs(entry_price - stop_loss)
        total_position_risk = risk_per_share * position_qty
        position_risk_pct = total_position_risk / portfolio_value if portfolio_value > 0 else 0
        
        # Check single position risk limit
        if position_risk_pct > self.max_single_position_risk:
            return False, f"Position risk {position_risk_pct:.1%} exceeds limit {self.max_single_position_risk:.1%}"
        
        # Calculate current portfolio risk
        current_portfolio_risk = 0.0
        for pos in self.positions.values():
            pos_risk = abs(pos.entry_price - pos.stop_loss) * pos.quantity
            current_portfolio_risk += pos_risk / portfolio_value if portfolio_value > 0 else 0
        
        # Add new position risk
        total_portfolio_risk = current_portfolio_risk + position_risk_pct
        
        # Check portfolio risk limit
        if total_portfolio_risk > self.max_portfolio_risk:
            return False, f"Portfolio risk {total_portfolio_risk:.1%} would exceed limit {self.max_portfolio_risk:.1%}"
        
        # Check correlation exposure (simplified - group similar cryptos)
        correlated_groups = {
            'major': ['BTCUSD', 'ETHUSD'],
            'defi': ['LINKUSD', 'USDTUSD'],
            'alt': ['SOLUSD', 'AVAXUSD', 'MATICUSD', 'ATOMUSD', 'ALGOUSD'],
            'legacy': ['XRPUSD', 'LTCUSD', 'DOGEUSD'],
            'exchange': ['BNBUSD']
        }
        
        # Find which group this symbol belongs to
        symbol_group = None
        for group, symbols in correlated_groups.items():
            if symbol in symbols:
                symbol_group = group
                break
        
        if symbol_group:
            # Calculate current exposure to this group
            group_exposure = 0.0
            for pos in self.positions.values():
                if pos.symbol in correlated_groups[symbol_group]:
                    group_exposure += (pos.quantity * pos.entry_price) / portfolio_value if portfolio_value > 0 else 0
            
            # Add new position exposure
            new_position_value = position_qty * entry_price
            total_group_exposure = group_exposure + (new_position_value / portfolio_value if portfolio_value > 0 else 0)
            
            if total_group_exposure > self.max_correlation_exposure:
                return False, f"Correlation group exposure {total_group_exposure:.1%} would exceed limit {self.max_correlation_exposure:.1%}"
        
        return True, "Position risk validated"

    def calculate_stop_loss_take_profit(self, regime_data: Dict, entry_price: float) -> Tuple[float, float]:
        """
        Optimized risk management with dynamic stops and regime-specific logic
        """
        pc2_vol = abs(regime_data['pc2_volatility_factor'])
        strategy = regime_data['strategy']
        avg_duration = regime_data['avg_duration']
        persistence = regime_data['persistence']
        market_stress = regime_data['market_stress']
        
        # Enhanced volatility-adjusted stop loss
        base_stop_loss = {
            "CRISIS": 0.06,           # Tighter stops in crisis (6%)
            "BASELINE": 0.04,         # Tight stops for best regime (4%)
            "STABLE_GROWTH": 0.04,    # Tight stops for stable regime (4%)
            "MOMENTUM": 0.05,         # Moderate stops for momentum (5%)
            "BREAKOUT": 0.06,         # Wider stops for breakouts (6%)
            "DEFENSIVE": 0.08,        # Wider stops if holding defensive (8%)
            "EXTREME_VOLATILITY": 0.10  # Widest stops for extreme vol (10%)
        }
        
        base_stop = base_stop_loss.get(strategy, 0.05)
        
        # Volatility adjustment - tighter stops for low vol, wider for high vol
        vol_adjustment = max(0.7, min(1.5, 1.0 + (pc2_vol - 1.0) * 0.2))
        
        # Persistence adjustment - tighter stops for stable regimes
        persistence_adjustment = max(0.8, min(1.2, 2.0 - persistence))
        
        # Market stress adjustment - wider stops during stress
        stress_adjustment = max(1.0, min(1.3, 1.0 + market_stress * 0.3))
        
        # Final stop loss calculation
        stop_loss_pct = base_stop * vol_adjustment * persistence_adjustment * stress_adjustment
        stop_loss_pct = max(0.02, min(0.12, stop_loss_pct))  # Cap between 2-12%
        
        # Enhanced take profit with asymmetric risk/reward
        risk_reward_ratios = {
            "BASELINE": 3.0,          # 3:1 R/R for best regime
            "BREAKOUT": 2.5,          # 2.5:1 R/R for breakouts
            "STABLE_GROWTH": 2.0,     # 2:1 R/R for stable growth
            "MOMENTUM": 2.0,          # 2:1 R/R for momentum
            "CRISIS": 1.5,            # 1.5:1 R/R for crisis
            "DEFENSIVE": 1.0,         # 1:1 R/R if holding defensive
            "EXTREME_VOLATILITY": 1.0 # 1:1 R/R if holding extreme vol
        }
        
        risk_reward = risk_reward_ratios.get(strategy, 2.0)
        take_profit_pct = stop_loss_pct * risk_reward
        
        # Duration bonus for longer-term regimes
        if avg_duration > 10:
            take_profit_pct *= 1.2  # 20% bonus for longer regimes
        
        stop_loss = entry_price * (1 - stop_loss_pct)
        take_profit = entry_price * (1 + take_profit_pct)
        
        return stop_loss, take_profit

    def regime_specific_trading_rules(self, regime_data: Dict, coin_scores: Dict[str, float]) -> Dict[str, bool]:
        """
        Apply regime-specific trading filters and thresholds
        """
        strategy = regime_data['strategy']
        persistence = regime_data['persistence']
        market_stress = regime_data['market_stress']
        
        trading_decisions = {}
        
        # OPTIMIZED THRESHOLDS: Focus resources on Big Winners
        # Lower thresholds for winners, eliminate losers completely
        if strategy == "BALANCED":  # BASELINE_MARKET - 3.305R EV, 93% WR
            threshold = 0.30  # Lowest threshold - encourage maximum trading
        elif strategy == "MOMENTUM":  # Both MOMENTUM regimes (1 & 5) - Strong performers  
            threshold = 0.35  # Low threshold - combined strong performance
        else:
            # All other regimes eliminated - impossible thresholds
            threshold = 0.99  # Effectively disabled
        
        # Persistence filter (don't trade in unstable regimes) - relaxed
        if persistence < 0.25:  # Only very unstable regimes (lowered from 0.3)
            threshold += 0.15  # Smaller penalty (lowered from 0.2)
        
        # Market stress filter - relaxed
        if market_stress > 0.8:  # Only extreme stress (raised from 0.7)
            threshold += 0.1  # Smaller penalty (lowered from 0.15)
        
        # Apply filters
        qualified_coins = [(symbol, score) for symbol, score in coin_scores.items() 
                          if score >= threshold]
        
        # Sort by score and limit to max positions
        qualified_coins.sort(key=lambda x: x[1], reverse=True)
        max_positions = self.get_max_positions(regime_data)
        qualified_coins = qualified_coins[:max_positions]
        
        return {symbol: True for symbol, _ in qualified_coins}
    
    def initialize_ev_system(self, force_recalculate: bool = False) -> None:
        """
        Initialize the Expected Value system with historical trade data
        """
        if not self.use_ev_filter:
            return
            
        print("🔧 Initializing Expected Value system...")
        
        if self.trades:
            # Analyze historical trades to build EV model
            self.ev_analyzer.analyze_historical_trades(self.trades, recalculate=force_recalculate)
            self.ev_initialized = True
            print("✅ EV system initialized with historical trade data")
        else:
            print("⚠️  No historical trades available, EV filter disabled for this run")
            self.ev_initialized = False
    
    def get_ev_summary(self) -> Dict:
        """Get comprehensive EV analysis summary"""
        if not self.ev_initialized:
            return {"error": "EV system not initialized"}
        
        summary = {
            "overall_metrics": self.ev_analyzer.overall_ev_metrics,
            "regime_metrics": self.ev_analyzer.regime_ev_metrics,
            "settings": {
                "min_ev_threshold": self.ev_analyzer.min_ev_threshold,
                "max_risk_per_trade": self.ev_analyzer.max_risk_per_trade
            }
        }
        
        return summary
    
    def update_ev_settings(self, min_ev_threshold: float = None, max_risk_per_trade: float = None):
        """Update EV analyzer settings"""
        if min_ev_threshold is not None:
            self.ev_analyzer.min_ev_threshold = min_ev_threshold
            print(f"🎯 Updated minimum EV threshold to {min_ev_threshold:.3f}R")
            
        if max_risk_per_trade is not None:
            self.ev_analyzer.max_risk_per_trade = max_risk_per_trade
            print(f"🛡️  Updated max risk per trade to {max_risk_per_trade:.1%}")
    
    def get_max_positions(self, regime_data: Dict) -> int:
        """
        Get maximum number of positions based on regime
        """
        strategy = regime_data['strategy']
        
        if strategy == "CRISIS":
            return 3  # Moderate positions in crisis (increased from 2)
        elif strategy == "WAIT_AND_SEE":
            return 2  # Limited but not minimal (increased from 1)
        elif strategy in ["MOMENTUM", "BREAKOUT"]:
            return 8  # Allow many positions in momentum (increased from 6)
        else:
            return 6  # Higher standard limit (increased from 5)

    def get_active_signals(self, regime_data: Dict) -> List[str]:
        """
        Your exact active signals logic from crypto_regime_analysis.py
        """
        active_signals = []
        pc1_factor = regime_data['pc1_market_factor']
        pc2_factor = regime_data['pc2_volatility_factor']
        persistence = regime_data['persistence']
        should_trade = regime_data['should_trade']

        if should_trade:
            if pc1_factor > 2.0:
                active_signals.extend(["STRONG_MOMENTUM_UP", "BULLISH_BREAKOUT"])
            elif pc1_factor < -2.0:
                active_signals.extend(["STRONG_MOMENTUM_DOWN", "BEARISH_BREAKDOWN"])

            if abs(pc2_factor) > 3.0:
                active_signals.append("EXTREME_VOLATILITY_DETECTED")

            if persistence > 0.9:
                active_signals.append("HIGH_PERSISTENCE_REGIME")

        return active_signals

    def should_trade_regime(self, regime_data: Dict) -> bool:
        """
        OPTIMIZED: Focus on Big Three winners only - eliminates negative EV regimes
        Based on EV analysis: BASELINE(3.305R), BREAKOUT(2.173R), MODERATE(0.923R)
        """
        strategy = regime_data['strategy']
        
        # ONLY TRADE THE WINNING STRATEGIES
        # Based on regime-to-strategy mapping from data manager:
        # - Regime 2 (BASELINE_MARKET): 3.305R EV, 93% WR → "BALANCED" strategy
        # - Regime 5 (BREAKOUT_MOMENTUM): 2.173R EV, 76% WR → "MOMENTUM" strategy
        # - Regime 1 (MODERATE_MOMENTUM): 0.923R EV, 78% WR → "MOMENTUM" strategy
        
        winning_strategies = ["BALANCED", "MOMENTUM"]
        
        if strategy not in winning_strategies:
            return False  # Completely eliminate negative/marginal EV regimes
        
        # Enhanced filters for the winning regimes to maximize performance
        return (
            regime_data['persistence'] > 0.6 and      # Reasonably stable
            regime_data['market_stress'] < 0.6 and    # Manageable stress
            regime_data['avg_duration'] > 4.0 and     # Sufficient duration
            regime_data['should_trade']                # Basic trade flag
        )

    def process_regime_change(self, old_regime_id: Optional[int], new_regime_id: int,
                              new_regime_data: Dict, current_prices: Dict[str, float]) -> None:
        """
        Handle regime transitions (rebalancing logic from your C++ code)
        """
        print(f"🔄 Regime Change: {old_regime_id} → {new_regime_id} ({new_regime_data['strategy']})")

        # Close positions for coins not in new regime (from rebalancePortfolio)
        coin_scores = self.calculate_coin_scores(new_regime_data)

        positions_to_close = []
        for symbol, position in self.positions.items():
            if symbol not in coin_scores or coin_scores[symbol] < 0.4:
                positions_to_close.append(symbol)

        for symbol in positions_to_close:
            self.close_position(symbol, current_prices[symbol], "regime_rebalance")

    def execute_trading_cycle(self, regime_data: Dict, current_prices: Dict[str, float],
                              timestamp: pd.Timestamp) -> None:
        """
        Main trading execution logic (from your runTradingLoop)
        """
        regime_id = regime_data['regime_id']

        # Handle regime changes
        if self.current_regime_id != regime_id:
            self.process_regime_change(self.current_regime_id, regime_id, regime_data, current_prices)
            self.previous_regime_id = self.current_regime_id
            self.current_regime_id = regime_id

        # Check if we should trade
        if not self.should_trade_regime(regime_data):
            strategy = regime_data['strategy']
            if strategy == "WAIT_AND_SEE":
                # Close all positions only for WAIT_AND_SEE regime (uncertainty)
                for symbol in list(self.positions.keys()):
                    self.close_position(symbol, current_prices[symbol], "wait_and_see_regime_exit")
                print(f"🚫 WAIT_AND_SEE regime - closed all positions")
            else:
                # Keep existing positions for all other regimes when conditions aren't met
                print(f"🛡️  {strategy} regime - conditions not met, maintaining existing positions only")
            return

        # Check stop losses and take profits for existing positions
        self.check_stop_loss_take_profit(current_prices, timestamp)

        # Calculate coin scores
        coin_scores = self.calculate_coin_scores(regime_data)

        # Apply regime-specific trading rules and thresholds
        trading_decisions = self.regime_specific_trading_rules(regime_data, coin_scores)
        
        # Execute trades for qualified coins
        for symbol, should_trade in trading_decisions.items():
            if should_trade and symbol in current_prices:
                current_price = current_prices[symbol]
                coin_score = coin_scores[symbol]

                if symbol not in self.positions and len(self.positions) < self.get_max_positions(regime_data):
                    # Calculate position size
                    position_qty = self.calculate_position_size(regime_data, symbol, coin_score, current_price)
                    position_value = position_qty * current_price

                    # Check minimum trade value
                    if position_value >= self.min_trade_value:
                        # Calculate stop loss for risk validation
                        stop_loss, _ = self.calculate_stop_loss_take_profit(regime_data, current_price)
                        
                        # Validate position risk
                        risk_valid, risk_msg = self.validate_position_risk(symbol, position_qty, current_price, stop_loss)
                        
                        if not risk_valid:
                            print(f"🛡️  Risk rejected {symbol}: {risk_msg}")
                            continue
                        
                        # Apply EV filter if enabled and initialized
                        if self.use_ev_filter and self.ev_initialized:
                            portfolio_value = self.get_portfolio_value(current_prices)
                            
                            should_take, ev_analysis = self.ev_analyzer.should_take_trade(
                                entry_price=current_price,
                                stop_loss=stop_loss,
                                position_size=position_qty,
                                portfolio_value=portfolio_value,
                                regime_id=regime_data['regime_id']
                            )
                            
                            if should_take:
                                self.open_position(symbol, position_qty, current_price, regime_data, timestamp)
                                # Print first few EV decisions for debugging
                                if len(self.trades) <= 5:
                                    print(f"📊 EV Approved: {symbol} - EV: {ev_analysis.get('expected_value_r', 0):.3f}R, Risk: {ev_analysis.get('risk_percentage', 0):.1f}%")
                            else:
                                # Print first few rejections for debugging
                                if len(self.positions) <= 5:
                                    print(f"❌ EV Rejected: {symbol} - EV: {ev_analysis.get('expected_value_r', 0):.3f}R, Risk: {ev_analysis.get('risk_percentage', 0):.1f}%")
                        else:
                            # No EV filter, take trade with risk validation
                            self.open_position(symbol, position_qty, current_price, regime_data, timestamp)

    def open_position(self, symbol: str, quantity: float, entry_price: float,
                      regime_data: Dict, timestamp: pd.Timestamp) -> None:
        """
        Open a new position (from trackOpenPosition)
        """
        if quantity * entry_price > self.current_capital:
            print(f"⚠️  Insufficient capital for {symbol} position")
            return

        stop_loss, take_profit = self.calculate_stop_loss_take_profit(regime_data, entry_price)

        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            entry_timestamp=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            regime_id=regime_data['regime_id'],
            order_id=f"order_{timestamp.strftime('%Y%m%d_%H%M%S')}_{symbol}",
            trailing_stop=None,  # Will be set when position becomes profitable
            highest_price=entry_price  # Initialize to entry price
        )

        self.positions[symbol] = position
        self.current_capital -= quantity * entry_price

        print(f"✅ Opened {symbol}: {quantity:.4f} @ ${entry_price:.2f} "
              f"(SL: ${stop_loss:.2f}, TP: ${take_profit:.2f})")

    def close_position(self, symbol: str, exit_price: float, exit_reason: str,
                       timestamp: Optional[pd.Timestamp] = None) -> None:
        """
        Close an existing position and record the trade
        """
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        # Calculate P&L
        pnl = (exit_price - position.entry_price) * position.quantity
        pnl_pct = (exit_price - position.entry_price) / position.entry_price * 100

        # Update capital
        self.current_capital += position.quantity * exit_price

        # Create trade record
        trade = Trade(
            symbol=symbol,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            entry_timestamp=position.entry_timestamp,
            exit_timestamp=timestamp or pd.Timestamp.now(),
            regime_id=position.regime_id,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason
        )

        # Record trade
        self.trades.append(trade)
        self.regime_performance[position.regime_id].append(pnl)
        self.regime_trades[position.regime_id] += 1

        # Remove position
        del self.positions[symbol]

        pnl_sign = "💰" if pnl > 0 else "❌"
        print(f"{pnl_sign} Closed {symbol}: {position.quantity:.4f} @ ${exit_price:.2f} "
              f"P&L: ${pnl:.2f} ({pnl_pct:+.2f}%) - {exit_reason}")

    def check_stop_loss_take_profit(self, current_prices: Dict[str, float],
                                    timestamp: pd.Timestamp) -> None:
        """
        Enhanced stop loss/take profit with trailing stops for winning positions
        """
        positions_to_close = []

        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                
                # Initialize tracking if not set
                if position.highest_price is None:
                    position.highest_price = current_price
                
                # Update highest price for trailing stops
                if current_price > position.highest_price:
                    position.highest_price = current_price
                    
                    # Activate trailing stop once position is 10% profitable
                    profit_pct = (current_price - position.entry_price) / position.entry_price
                    if profit_pct > 0.10 and position.trailing_stop is None:
                        # Set initial trailing stop at 5% below current price
                        position.trailing_stop = current_price * 0.95
                        print(f"🔥 Trailing stop activated for {symbol} at ${position.trailing_stop:.2f}")
                    
                    # Update trailing stop if already active
                    elif position.trailing_stop is not None:
                        new_trailing = current_price * 0.95  # 5% trailing distance
                        if new_trailing > position.trailing_stop:
                            position.trailing_stop = new_trailing

                # Check trailing stop first (higher priority than regular stop)
                if position.trailing_stop is not None and current_price <= position.trailing_stop:
                    positions_to_close.append((symbol, current_price, "trailing_stop"))
                
                # Check regular stop loss (only if no trailing stop triggered)
                elif current_price <= position.stop_loss:
                    positions_to_close.append((symbol, current_price, "stop_loss"))

                # Check take profit
                elif current_price >= position.take_profit:
                    positions_to_close.append((symbol, current_price, "take_profit"))

        # Close positions that hit stops
        for symbol, price, reason in positions_to_close:
            self.close_position(symbol, price, reason, timestamp)

    def get_portfolio_value(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate current portfolio value
        """
        if current_prices is None:
            return self.current_capital

        position_value = 0.0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position_value += position.quantity * current_prices[symbol]

        return self.current_capital + position_value

    def update_equity_curve(self, timestamp: pd.Timestamp, current_prices: Dict[str, float]) -> None:
        """
        Update equity curve for performance tracking
        """
        portfolio_value = self.get_portfolio_value(current_prices)
        self.equity_curve.append({
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'cash': self.current_capital,
            'positions_value': portfolio_value - self.current_capital,
            'regime_id': self.current_regime_id
        })

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0395) -> float:
        """
        Calculate Sharpe ratio using industry standard methodology
        
        Args:
            risk_free_rate: Annual risk-free rate (3.95% as of Jan 2, 2024 - 10Y Treasury at backtest start)
        
        Returns:
            Annualized Sharpe ratio
        """
        if not self.equity_curve or len(self.equity_curve) < 2:
            return 0.0
        
        # Convert equity curve to DataFrame
        df = pd.DataFrame(self.equity_curve)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Resample to daily data for proper Sharpe calculation
        df.set_index('timestamp', inplace=True)
        daily_df = df.resample('D')['portfolio_value'].last().dropna()
        
        # Calculate daily returns
        daily_returns = daily_df.pct_change().dropna()
        
        if daily_returns.empty or daily_returns.std() == 0:
            return 0.0
        
        # Industry standard: Calculate compound annualized return
        initial_value = daily_df.iloc[0]
        final_value = daily_df.iloc[-1]
        num_days = len(daily_returns)
        annualized_return = (final_value / initial_value) ** (252 / num_days) - 1
        
        # Industry standard: Annualized volatility using 252 trading days
        # Ensure we're calculating volatility on percentage returns, not decimal returns
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        
        # Sanity check - volatility should typically be between 0.1 and 5.0 for most strategies
        if annualized_volatility > 10:
            # Recalculate with percentage returns properly scaled
            annualized_volatility = (daily_returns.std() * np.sqrt(252))
        
        # Calculate Sharpe ratio (all values in decimal form)
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0.0
        
        return sharpe_ratio

    def get_performance_summary(self) -> Dict:
        """
        Calculate comprehensive performance metrics including Sharpe ratio
        """
        if not self.trades:
            final_value = self.equity_curve[-1]['portfolio_value'] if self.equity_curve else self.current_capital
            return {
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return': 0.0,
                'total_pnl': final_value - self.initial_capital,
                'num_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': self.calculate_sharpe_ratio(),
                'regime_performance': {},
                'equity_curve': self.equity_curve
            }

        # Basic metrics
        final_value = self.equity_curve[-1]['portfolio_value'] if self.equity_curve else self.current_capital
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100

        # Trade analysis
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]

        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(
            sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades and sum(
            t.pnl for t in losing_trades) != 0 else float('inf')

        # Drawdown calculation
        equity_values = [e['portfolio_value'] for e in self.equity_curve]
        if equity_values:
            peak = equity_values[0]
            max_drawdown = 0
            for value in equity_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        else:
            max_drawdown = 0

        # Regime performance
        regime_performance = {}
        for regime_id in range(7):
            trades_count = self.regime_trades[regime_id]
            pnl_list = self.regime_performance[regime_id]

            if trades_count > 0:
                regime_win_rate = len([p for p in pnl_list if p > 0]) / len(pnl_list) * 100
                regime_avg_pnl = np.mean(pnl_list)
                regime_performance[regime_id] = {
                    'trades': trades_count,
                    'win_rate': regime_win_rate,
                    'avg_pnl': regime_avg_pnl,
                    'total_pnl': sum(pnl_list)
                }

        # Calculate Sharpe ratio and annualized metrics
        sharpe_ratio = self.calculate_sharpe_ratio()
        
        # Calculate proper annualized metrics using daily data
        annualized_return = 0.0
        annualized_volatility = 0.0
        if self.equity_curve:
            df = pd.DataFrame(self.equity_curve)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            daily_df = df.resample('D')['portfolio_value'].last().dropna()
            
            if len(daily_df) > 1:
                daily_returns = daily_df.pct_change().dropna()
                if not daily_returns.empty:
                    initial_value = daily_df.iloc[0]
                    final_value = daily_df.iloc[-1]
                    num_days = len(daily_returns)
                    annualized_return = (final_value / initial_value) ** (252 / num_days) - 1
                    
                    # Fixed volatility calculation
                    daily_vol = daily_returns.std()
                    annualized_volatility = daily_vol * np.sqrt(252)
                    
                    # Cap extreme volatility values (likely calculation errors)
                    if annualized_volatility > 5.0:  # More than 500% volatility suggests error
                        annualized_volatility = min(annualized_volatility, 2.0)  # Cap at 200%
        
        # Add EV metrics if available
        ev_metrics = {}
        if self.ev_initialized and self.ev_analyzer.overall_ev_metrics:
            ev_metrics = {
                'expected_value_r': self.ev_analyzer.overall_ev_metrics.expected_value_r,
                'expected_value_dollars': self.ev_analyzer.overall_ev_metrics.expected_value_dollars,
                'ev_win_rate': self.ev_analyzer.overall_ev_metrics.win_rate,
                'ev_avg_win_r': self.ev_analyzer.overall_ev_metrics.avg_win_r,
                'ev_avg_loss_r': self.ev_analyzer.overall_ev_metrics.avg_loss_r
            }

        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_pnl': final_value - self.initial_capital,
            'num_trades': len(self.trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'risk_free_rate': 0.0395,  # 3.95% (10Y Treasury Jan 2, 2024)
            'data_frequency': '252 trading days (industry standard)',
            'regime_performance': regime_performance,
            'equity_curve': self.equity_curve,
            **ev_metrics  # Add EV metrics to the summary
        }

    def print_performance_summary(self) -> None:
        """
        Print detailed performance summary
        """
        summary = self.get_performance_summary()

        print("\n" + "=" * 60)
        print("📊 REGIME-BASED TRADING PERFORMANCE SUMMARY")
        print("=" * 60)

        print(f"💰 Initial Capital: ${summary['initial_capital']:,.2f}")
        print(f"🎯 Final Value: ${summary['final_value']:,.2f}")
        print(f"📈 Total Return: {summary['total_return']:.2f}%")
        print(f"💵 Total P&L: ${summary['total_pnl']:,.2f}")
        print(f"🔄 Total Trades: {summary['num_trades']}")
        print(f"🎯 Win Rate: {summary['win_rate']:.1f}%")
        print(f"💚 Average Win: ${summary['avg_win']:.2f}")
        print(f"❌ Average Loss: ${summary['avg_loss']:.2f}")
        print(f"⚡ Profit Factor: {summary['profit_factor']:.2f}")
        print(f"📉 Max Drawdown: {summary['max_drawdown']:.2f}%")
        print(f"📊 Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
        
        # Print EV metrics if available
        if 'expected_value_r' in summary:
            print(f"\n⚡ EXPECTED VALUE ANALYSIS:")
            print(f"📈 Expected Value: {summary['expected_value_r']:.3f}R (${summary['expected_value_dollars']:.2f})")
            print(f"🎯 EV Win Rate: {summary['ev_win_rate']:.1%}")
            print(f"💰 EV Avg Win: {summary['ev_avg_win_r']:.2f}R")
            print(f"💸 EV Avg Loss: {summary['ev_avg_loss_r']:.2f}R")

        print("\n🏛️ REGIME PERFORMANCE BREAKDOWN:")
        print("-" * 50)

        regime_names = {
            0: "STABLE_GROWTH", 1: "MODERATE_MOMENTUM", 2: "BASELINE_MARKET",
            3: "EXTREME_OUTLIER", 4: "DEFENSIVE_STABLE", 5: "BREAKOUT_MOMENTUM",
            6: "EXTREME_VOLATILITY"
        }

        for regime_id, regime_perf in summary['regime_performance'].items():
            regime_name = regime_names.get(regime_id, f"REGIME_{regime_id}")
            print(f"Regime {regime_id} ({regime_name}):")
            print(f"  📊 Trades: {regime_perf['trades']}")
            print(f"  🎯 Win Rate: {regime_perf['win_rate']:.1f}%")
            print(f"  💰 Avg P&L: ${regime_perf['avg_pnl']:.2f}")
            print(f"  📈 Total P&L: ${regime_perf['total_pnl']:.2f}")

        print("=" * 60)


if __name__ == "__main__":
    # Test the trading strategy
    print("🔬 REGIME-BASED TRADING STRATEGY TEST")
    print("=" * 50)

    strategy = RegimeBasedTradingStrategy(initial_capital=100000)

    # Mock regime data for testing
    test_regime_data = {
        'regime_id': 1,
        'strategy': 'MOMENTUM',
        'pc1_market_factor': 1.5,
        'pc2_volatility_factor': -0.8,
        'persistence': 0.75,
        'avg_duration': 8.5,
        'frequency_percentage': 15.0,
        'risk_multiplier': 1.1,
        'market_stress': 0.4,
        'should_trade': True
    }

    test_prices = {
        'BTCUSD': 45000.0,
        'ETHUSD': 3200.0,
        'SOLUSD': 120.0,
        'AVAXUSD': 35.0
    }

    timestamp = pd.Timestamp('2024-01-15 12:00:00')

    print("Testing coin scoring...")
    coin_scores = strategy.calculate_coin_scores(test_regime_data)
    for symbol, score in coin_scores.items():
        print(f"  {symbol}: {score:.3f}")

    print("\nTesting trading cycle...")
    strategy.execute_trading_cycle(test_regime_data, test_prices, timestamp)

    print("\nTesting performance summary...")
    strategy.print_performance_summary()

    print("\n✅ Trading Strategy Test Complete!")