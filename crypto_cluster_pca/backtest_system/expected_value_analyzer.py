#!/usr/bin/env python3
"""
Expected Value (EV) Analysis System for Trading Strategy
Calculates EV based on historical trades and implements EV-based filtering
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import pickle
import os


@dataclass
class TradeAnalysis:
    """Individual trade analysis with EV metrics"""
    entry_price: float
    exit_price: float
    stop_loss: float
    risk_amount: float  # Entry - Stop Loss
    reward_amount: float  # Exit - Entry (can be negative)
    r_multiple: float  # Reward/Risk ratio
    is_winner: bool
    regime_id: int
    symbol: str


@dataclass
class EVMetrics:
    """Expected Value metrics for strategy analysis"""
    win_rate: float
    loss_rate: float
    avg_win_r: float  # Average win in R multiples
    avg_loss_r: float  # Average loss in R multiples
    avg_win_dollars: float
    avg_loss_dollars: float
    expected_value_r: float  # EV in R multiples
    expected_value_dollars: float  # EV in dollars
    total_trades: int
    winning_trades: int
    losing_trades: int
    profit_factor: float
    largest_win: float
    largest_loss: float


class ExpectedValueAnalyzer:
    """
    Comprehensive Expected Value analysis system for trading strategies
    """
    
    def __init__(self, min_ev_threshold: float = 0.0, max_risk_per_trade: float = 0.02):
        """
        Initialize EV analyzer
        
        Args:
            min_ev_threshold: Minimum EV required to take a trade (default: 0.0)
            max_risk_per_trade: Maximum risk per trade as % of equity (default: 2%)
        """
        self.min_ev_threshold = min_ev_threshold
        self.max_risk_per_trade = max_risk_per_trade
        
        # Historical data storage
        self.trade_analyses: List[TradeAnalysis] = []
        self.regime_ev_metrics: Dict[int, EVMetrics] = {}
        self.overall_ev_metrics: Optional[EVMetrics] = None
        
        # EV calculation cache
        self.ev_cache_file = "ev_metrics_cache.pkl"
        
    def analyze_historical_trades(self, trades: List, recalculate: bool = False) -> EVMetrics:
        """
        Analyze historical trades to calculate EV metrics
        
        Args:
            trades: List of Trade objects from backtest
            recalculate: Force recalculation even if cache exists
            
        Returns:
            EVMetrics object with comprehensive analysis
        """
        print("📊 EXPECTED VALUE ANALYSIS")
        print("=" * 50)
        
        # Check cache first
        if not recalculate and os.path.exists(self.ev_cache_file):
            try:
                with open(self.ev_cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.overall_ev_metrics = cached_data['overall']
                    self.regime_ev_metrics = cached_data['regime']
                    print("📁 Loaded cached EV metrics")
                    return self.overall_ev_metrics
            except:
                print("⚠️  Cache loading failed, recalculating...")
        
        if not trades:
            print("❌ No trades to analyze")
            return self._create_empty_metrics()
        
        print(f"🔍 Analyzing {len(trades)} historical trades...")
        
        # Convert trades to trade analyses
        self.trade_analyses = []
        for trade in trades:
            analysis = self._analyze_single_trade(trade)
            self.trade_analyses.append(analysis)
        
        # Calculate overall metrics
        self.overall_ev_metrics = self._calculate_ev_metrics(self.trade_analyses)
        
        # Calculate regime-specific metrics
        self.regime_ev_metrics = {}
        for regime_id in range(7):  # 0-6 regimes
            regime_trades = [t for t in self.trade_analyses if t.regime_id == regime_id]
            if regime_trades:
                self.regime_ev_metrics[regime_id] = self._calculate_ev_metrics(regime_trades)
        
        # Cache results
        self._cache_metrics()
        
        # Print analysis
        self._print_ev_analysis()
        
        return self.overall_ev_metrics
    
    def load_from_backtest_cache(self, cache_dir: str = "backtest_data_cache", force_recalculate: bool = False) -> Optional[EVMetrics]:
        """
        Load EV analysis from existing backtest cache data
        
        Args:
            cache_dir: Directory containing backtest cache files
            force_recalculate: Force recalculation from backtest cache even if EV cache exists
            
        Returns:
            EVMetrics object if successful, None otherwise
        """
        print("📊 LOADING EV ANALYSIS FROM BACKTEST CACHE")
        print("=" * 50)
        
        # Check if EV cache already exists and is recent (only if not forcing recalculation)
        if not force_recalculate and os.path.exists(self.ev_cache_file):
            try:
                with open(self.ev_cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    
                    # Check if the cache has real data (more than 10 trades) indicating backtest data
                    if (cached_data['overall'] and 
                        cached_data['overall'].total_trades > 10):
                        self.overall_ev_metrics = cached_data['overall']
                        self.regime_ev_metrics = cached_data['regime']
                        print("📁 Loaded existing comprehensive EV cache from backtest data")
                        self._print_ev_analysis()
                        return self.overall_ev_metrics
                    else:
                        print("⚠️  Existing cache contains sample data, regenerating from backtest cache...")
            except Exception as e:
                print(f"⚠️  Failed to load EV cache: {e}")
        elif force_recalculate:
            print("🔄 Force recalculation requested, generating from backtest cache...")
        
        # Try to load from backtest data cache
        try:
            # Look for the most recent backtest cache files
            import glob
            
            crypto_files = glob.glob(os.path.join(cache_dir, "crypto_data_*.pkl"))
            regime_files = glob.glob(os.path.join(cache_dir, "historical_regimes_*.pkl"))
            
            if not crypto_files or not regime_files:
                print(f"❌ No backtest cache files found in {cache_dir}")
                return None
            
            # Use the most recent files
            crypto_file = max(crypto_files, key=os.path.getmtime)
            regime_file = max(regime_files, key=os.path.getmtime)
            
            print(f"📊 Loading crypto data from: {os.path.basename(crypto_file)}")
            print(f"🎯 Loading regime data from: {os.path.basename(regime_file)}")
            
            # Load cached data
            with open(crypto_file, 'rb') as f:
                crypto_data = pickle.load(f)
            
            with open(regime_file, 'rb') as f:
                regime_data = pickle.load(f)
            
            print(f"✅ Loaded {len(crypto_data)} crypto datasets")
            print(f"✅ Loaded {len(regime_data)} regime timestamps")
            
            # Simulate trades from cache data for EV analysis
            simulated_trades = self._simulate_trades_from_cache(crypto_data, regime_data)
            
            if simulated_trades:
                print(f"🔍 Generated {len(simulated_trades)} simulated trades for EV analysis")
                return self.analyze_historical_trades(simulated_trades, recalculate=True)
            else:
                print("❌ Could not generate trades from cache data")
                return None
                
        except Exception as e:
            print(f"❌ Error loading from backtest cache: {e}")
            return None
    
    def _simulate_trades_from_cache(self, crypto_data: Dict, regime_data: pd.DataFrame) -> List:
        """
        Simulate trades from cached backtest data for EV analysis
        
        Args:
            crypto_data: Dictionary of cryptocurrency price data
            regime_data: DataFrame with regime classifications
            
        Returns:
            List of simulated Trade objects
        """
        from backtest_trading_strategy import Trade
        import pandas as pd
        
        simulated_trades = []
        regime_trade_counts = {i: 0 for i in range(7)}  # Track trades per regime
        max_trades_per_regime = 100  # Ensure balanced representation
        
        # Simple simulation: create trades based on regime changes and price movements
        for idx, (timestamp, row) in enumerate(regime_data.iterrows()):
            if idx == 0 or idx >= len(regime_data) - 10:  # Skip first and last few rows
                continue
                
            regime_id = row['regime_id']
            strategy = row['strategy']
            
            # Skip if this regime already has enough trades
            if regime_trade_counts[regime_id] >= max_trades_per_regime:
                continue
            
            # Skip non-trading regimes (but allow some crisis/wait-and-see trades for analysis)
            if strategy in ['WAIT_AND_SEE'] and regime_trade_counts[regime_id] >= 20:
                continue
            
            # Look for a cryptocurrency with good data at this timestamp
            for symbol_key, symbol_data in crypto_data.items():
                    
                try:
                    # Find price at this timestamp
                    if timestamp not in symbol_data.index:
                        continue
                        
                    entry_price = symbol_data.loc[timestamp, 'close']
                    
                    # Look ahead 3-7 days for exit
                    exit_idx = min(idx + 5, len(regime_data) - 1)
                    exit_timestamp = regime_data.index[exit_idx]
                    
                    if exit_timestamp not in symbol_data.index:
                        continue
                        
                    exit_price = symbol_data.loc[exit_timestamp, 'close']
                    
                    # Calculate stop loss based on regime (simple approximation)
                    if strategy in ['MOMENTUM', 'BREAKOUT']:
                        stop_loss_pct = 0.05  # 5% stop
                    elif strategy in ['STABLE_GROWTH', 'BASELINE']:
                        stop_loss_pct = 0.03  # 3% stop  
                    else:
                        stop_loss_pct = 0.04  # 4% stop
                    
                    stop_loss = entry_price * (1 - stop_loss_pct)
                    
                    # Calculate P&L
                    pnl = exit_price - entry_price
                    pnl_pct = (exit_price / entry_price - 1) * 100
                    
                    # Determine exit reason
                    if exit_price <= stop_loss:
                        exit_reason = "stop_loss"
                        exit_price = stop_loss
                        pnl = stop_loss - entry_price
                        pnl_pct = (stop_loss / entry_price - 1) * 100
                    elif pnl > 0:
                        exit_reason = "take_profit"
                    else:
                        exit_reason = "regime_exit"
                    
                    # Create trade
                    trade = Trade(
                        symbol=symbol_key,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        quantity=1.0,  # Normalized quantity
                        entry_timestamp=pd.to_datetime(timestamp),
                        exit_timestamp=pd.to_datetime(exit_timestamp),
                        regime_id=regime_id,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason
                    )
                    
                    # Add stop loss to trade object
                    trade.stop_loss = stop_loss
                    
                    simulated_trades.append(trade)
                    regime_trade_counts[regime_id] += 1
                    
                    # Break if this regime has enough trades
                    if regime_trade_counts[regime_id] >= max_trades_per_regime:
                        break
                        
                except Exception as e:
                    # Skip problematic data points
                    continue
            
            # Stop when all regimes have adequate representation or we have enough total trades
            if (all(count >= 20 for count in regime_trade_counts.values()) or 
                len(simulated_trades) >= 700):
                break
        
        return simulated_trades
    
    @classmethod
    def quick_cache_analysis(cls, cache_dir: str = "backtest_data_cache", force_recalculate: bool = True) -> 'ExpectedValueAnalyzer':
        """
        Quick method to create analyzer and load from cache
        
        Args:
            cache_dir: Directory containing backtest cache files
            force_recalculate: Force recalculation from backtest cache (default True)
            
        Returns:
            Initialized ExpectedValueAnalyzer with loaded metrics
        """
        analyzer = cls(min_ev_threshold=0.1, max_risk_per_trade=0.02)
        analyzer.load_from_backtest_cache(cache_dir, force_recalculate=force_recalculate)
        return analyzer
    
    def _analyze_single_trade(self, trade) -> TradeAnalysis:
        """Analyze a single trade for EV calculation"""
        # Calculate risk (entry to stop loss)
        risk_amount = abs(trade.entry_price - trade.stop_loss)
        
        # Calculate reward (actual exit - entry)
        reward_amount = trade.exit_price - trade.entry_price
        
        # Calculate R multiple (reward/risk ratio)
        r_multiple = reward_amount / risk_amount if risk_amount > 0 else 0
        
        # Determine if winner
        is_winner = reward_amount > 0
        
        return TradeAnalysis(
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            stop_loss=trade.stop_loss,
            risk_amount=risk_amount,
            reward_amount=reward_amount,
            r_multiple=r_multiple,
            is_winner=is_winner,
            regime_id=trade.regime_id,
            symbol=trade.symbol
        )
    
    def _calculate_ev_metrics(self, trade_analyses: List[TradeAnalysis]) -> EVMetrics:
        """Calculate EV metrics from trade analyses"""
        if not trade_analyses:
            return self._create_empty_metrics()
        
        # Separate winners and losers
        winners = [t for t in trade_analyses if t.is_winner]
        losers = [t for t in trade_analyses if not t.is_winner]
        
        total_trades = len(trade_analyses)
        winning_trades = len(winners)
        losing_trades = len(losers)
        
        # Calculate win/loss rates
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        loss_rate = losing_trades / total_trades if total_trades > 0 else 0
        
        # Calculate average wins and losses in R multiples
        avg_win_r = np.mean([t.r_multiple for t in winners]) if winners else 0
        avg_loss_r = np.mean([t.r_multiple for t in losers]) if losers else 0
        
        # Calculate average wins and losses in dollars
        avg_win_dollars = np.mean([t.reward_amount for t in winners]) if winners else 0
        avg_loss_dollars = np.mean([t.reward_amount for t in losers]) if losers else 0
        
        # Calculate Expected Value
        # EV = (P_win × Avg_Win) - (P_loss × Avg_Loss)
        expected_value_r = (win_rate * avg_win_r) - (loss_rate * abs(avg_loss_r))
        expected_value_dollars = (win_rate * avg_win_dollars) - (loss_rate * abs(avg_loss_dollars))
        
        # Calculate profit factor
        total_wins = sum(t.reward_amount for t in winners) if winners else 0
        total_losses = abs(sum(t.reward_amount for t in losers)) if losers else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Calculate largest win/loss
        largest_win = max([t.reward_amount for t in winners]) if winners else 0
        largest_loss = min([t.reward_amount for t in losers]) if losers else 0
        
        return EVMetrics(
            win_rate=win_rate,
            loss_rate=loss_rate,
            avg_win_r=avg_win_r,
            avg_loss_r=avg_loss_r,
            avg_win_dollars=avg_win_dollars,
            avg_loss_dollars=avg_loss_dollars,
            expected_value_r=expected_value_r,
            expected_value_dollars=expected_value_dollars,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            profit_factor=profit_factor,
            largest_win=largest_win,
            largest_loss=largest_loss
        )
    
    def _create_empty_metrics(self) -> EVMetrics:
        """Create empty metrics for when no trades available"""
        return EVMetrics(
            win_rate=0.0, loss_rate=0.0, avg_win_r=0.0, avg_loss_r=0.0,
            avg_win_dollars=0.0, avg_loss_dollars=0.0, expected_value_r=0.0,
            expected_value_dollars=0.0, total_trades=0, winning_trades=0,
            losing_trades=0, profit_factor=0.0, largest_win=0.0, largest_loss=0.0
        )
    
    def should_take_trade(self, entry_price: float, stop_loss: float, 
                         position_size: float, portfolio_value: float,
                         regime_id: Optional[int] = None) -> Tuple[bool, Dict]:
        """
        Determine if a trade should be taken based on EV and risk criteria
        
        Args:
            entry_price: Proposed entry price
            stop_loss: Proposed stop loss price
            position_size: Proposed position size (in units)
            portfolio_value: Current portfolio value
            regime_id: Current regime ID for regime-specific EV
            
        Returns:
            Tuple of (should_take_trade, analysis_dict)
        """
        analysis = {}
        
        # Calculate trade risk
        risk_per_unit = abs(entry_price - stop_loss)
        total_risk_dollars = risk_per_unit * position_size
        risk_percentage = total_risk_dollars / portfolio_value if portfolio_value > 0 else 0
        
        analysis['risk_per_unit'] = risk_per_unit
        analysis['total_risk_dollars'] = total_risk_dollars
        analysis['risk_percentage'] = risk_percentage * 100
        
        # Check risk limit
        risk_ok = risk_percentage <= self.max_risk_per_trade
        analysis['risk_ok'] = risk_ok
        analysis['max_risk_allowed'] = self.max_risk_per_trade * 100
        
        # Get appropriate EV metrics
        ev_metrics = self.overall_ev_metrics
        if regime_id is not None and regime_id in self.regime_ev_metrics:
            ev_metrics = self.regime_ev_metrics[regime_id]
            analysis['using_regime_ev'] = True
            analysis['regime_id'] = regime_id
        else:
            analysis['using_regime_ev'] = False
        
        if ev_metrics is None:
            analysis['ev_available'] = False
            analysis['should_take'] = risk_ok  # Fall back to risk check only
            return analysis['should_take'], analysis
        
        analysis['ev_available'] = True
        analysis['expected_value_r'] = ev_metrics.expected_value_r
        analysis['expected_value_dollars'] = ev_metrics.expected_value_dollars
        analysis['win_rate'] = ev_metrics.win_rate * 100
        
        # Calculate expected value for this specific trade
        expected_dollar_ev = ev_metrics.expected_value_dollars * position_size
        analysis['trade_expected_ev'] = expected_dollar_ev
        
        # Check EV threshold
        ev_ok = ev_metrics.expected_value_r >= self.min_ev_threshold
        analysis['ev_ok'] = ev_ok
        analysis['min_ev_required'] = self.min_ev_threshold
        
        # Final decision
        should_take = risk_ok and ev_ok
        analysis['should_take'] = should_take
        
        return should_take, analysis
    
    def get_regime_ev_summary(self) -> pd.DataFrame:
        """Get summary of EV metrics by regime"""
        if not self.regime_ev_metrics:
            return pd.DataFrame()
        
        regime_names = {
            0: "STABLE_GROWTH", 1: "MODERATE_MOMENTUM", 2: "BASELINE_MARKET",
            3: "EXTREME_OUTLIER", 4: "DEFENSIVE_STABLE", 5: "BREAKOUT_MOMENTUM",
            6: "EXTREME_VOLATILITY"
        }
        
        data = []
        for regime_id, metrics in self.regime_ev_metrics.items():
            data.append({
                'Regime_ID': regime_id,
                'Regime_Name': regime_names.get(regime_id, f"REGIME_{regime_id}"),
                'Total_Trades': metrics.total_trades,
                'Win_Rate_%': metrics.win_rate * 100,
                'Avg_Win_R': metrics.avg_win_r,
                'Avg_Loss_R': metrics.avg_loss_r,
                'Expected_Value_R': metrics.expected_value_r,
                'Expected_Value_$': metrics.expected_value_dollars,
                'Profit_Factor': metrics.profit_factor
            })
        
        return pd.DataFrame(data).sort_values('Expected_Value_R', ascending=False)
    
    def _cache_metrics(self):
        """Cache EV metrics to file"""
        try:
            cache_data = {
                'overall': self.overall_ev_metrics,
                'regime': self.regime_ev_metrics,
                'timestamp': datetime.now()
            }
            with open(self.ev_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"💾 Cached EV metrics to {self.ev_cache_file}")
        except Exception as e:
            print(f"⚠️  Failed to cache metrics: {e}")
    
    def _print_ev_analysis(self):
        """Print comprehensive EV analysis"""
        if not self.overall_ev_metrics:
            print("❌ No EV metrics available")
            return
        
        metrics = self.overall_ev_metrics
        
        print(f"\n📈 OVERALL EXPECTED VALUE ANALYSIS")
        print("-" * 45)
        print(f"📊 Total Trades: {metrics.total_trades:,}")
        print(f"✅ Winning Trades: {metrics.winning_trades:,} ({metrics.win_rate:.1%})")
        print(f"❌ Losing Trades: {metrics.losing_trades:,} ({metrics.loss_rate:.1%})")
        print(f"")
        print(f"💰 Average Win: {metrics.avg_win_r:.2f}R (${metrics.avg_win_dollars:.2f})")
        print(f"💸 Average Loss: {metrics.avg_loss_r:.2f}R (${metrics.avg_loss_dollars:.2f})")
        print(f"")
        print(f"⚡ Expected Value: {metrics.expected_value_r:.3f}R (${metrics.expected_value_dollars:.2f})")
        print(f"🔥 Profit Factor: {metrics.profit_factor:.2f}")
        print(f"🎯 Largest Win: ${metrics.largest_win:.2f}")
        print(f"📉 Largest Loss: ${metrics.largest_loss:.2f}")
        
        # EV interpretation
        if metrics.expected_value_r > 0:
            print(f"✅ POSITIVE EV: Strategy has positive expected value!")
        elif metrics.expected_value_r == 0:
            print(f"⚖️  NEUTRAL EV: Strategy breaks even on average")
        else:
            print(f"❌ NEGATIVE EV: Strategy has negative expected value")
        
        # Risk assessment
        print(f"\n🛡️  RISK MANAGEMENT SETTINGS")
        print(f"📊 Minimum EV Required: {self.min_ev_threshold:.3f}R")
        print(f"💼 Max Risk Per Trade: {self.max_risk_per_trade:.1%} of portfolio")
        
        # Regime breakdown
        if self.regime_ev_metrics:
            print(f"\n🏛️  REGIME-SPECIFIC EV ANALYSIS")
            print("-" * 45)
            regime_df = self.get_regime_ev_summary()
            for _, row in regime_df.iterrows():
                regime_name = row['Regime_Name'][:15]  # Truncate for display
                ev_r = row['Expected_Value_R']
                win_rate = row['Win_Rate_%']
                trades = row['Total_Trades']
                print(f"{regime_name:15} | EV: {ev_r:6.3f}R | WR: {win_rate:5.1f}% | Trades: {trades:3d}")


def test_ev_analyzer():
    """Test the EV analyzer with backtest cache data only"""
    print("🧪 TESTING EXPECTED VALUE ANALYZER")
    print("=" * 50)
    
    # Load from backtest cache
    analyzer = ExpectedValueAnalyzer.quick_cache_analysis(force_recalculate=True)
    
    if analyzer.overall_ev_metrics:
        print("✅ Successfully loaded EV metrics from backtest cache!")
        print(f"📊 Total Trades: {analyzer.overall_ev_metrics.total_trades}")
        print(f"⚡ Expected Value: {analyzer.overall_ev_metrics.expected_value_r:.3f}R")
        print(f"🎯 Win Rate: {analyzer.overall_ev_metrics.win_rate:.1%}")
        
        # Test trade decision with cache data
        should_take, analysis = analyzer.should_take_trade(
            entry_price=45000,
            stop_loss=44000,
            position_size=0.1,
            portfolio_value=100000,
            regime_id=0
        )
        
        print(f"\n🎯 TRADE DECISION TEST")
        print(f"Should take trade: {should_take}")
        print(f"Expected Value: {analysis.get('expected_value_r', 'N/A'):.3f}R")
        
        # Show regime breakdown
        if analyzer.regime_ev_metrics:
            print(f"\n🏛️  REGIMES WITH TRADES:")
            for regime_id, metrics in analyzer.regime_ev_metrics.items():
                print(f"   Regime {regime_id}: {metrics.total_trades} trades, {metrics.expected_value_r:.3f}R EV")
        
    else:
        print("❌ Failed to load EV metrics from backtest cache")


if __name__ == "__main__":
    test_ev_analyzer()