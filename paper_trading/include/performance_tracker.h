/**
 * @file performance_tracker.h
 * @brief Performance tracking and analytics for trading system
 * 
 * @author Jake Kostoryz
 * @date 2025
 * @version 4.0
 */

#pragma once

#include "common.h"
#include "data_structures.h"

// ============================================================================
// PERFORMANCE TRACKING
// ============================================================================

/**
 * @brief Comprehensive performance tracking and analytics
 * 
 * Tracks trading performance across multiple dimensions including
 * regime-based performance, symbol-specific results, drawdown metrics,
 * and overall portfolio statistics.
 */
class PerformanceTracker {
public:
    /// @brief Constructor to initialize tracking metrics
    PerformanceTracker();
    
    /// @brief Record the opening of a new trade
    void recordTradeOpen(const std::string& symbol, int regime_id);
    
    /// @brief Record the closing of a trade with results
    void recordTradeClose(const TradeResult& result);
    
    /// @brief Update unrealized P&L for all open positions
    void updateUnrealizedPnL(const std::map<std::string, PositionInfo>& positions,
                            const std::map<std::string, double>& current_prices);
    
    /// @brief Update drawdown metrics based on current portfolio value
    void updateDrawdown(double current_portfolio_value);
    
    /// @brief Print comprehensive performance statistics
    void printPerformance();
    
    /// @brief Calculate overall win rate percentage
    double getWinRate() const;
    
    /// @brief Calculate average winning trade amount
    double getAvgWin() const;
    
    /// @brief Calculate average losing trade amount
    double getAvgLoss() const;
    
    /// @brief Calculate profit factor (total wins / total losses)
    double getProfitFactor() const;

public: // Made public for direct access by strategy
    std::map<int, double> regime_returns;     ///< Cumulative returns by regime
    std::map<int, int> regime_trade_counts;   ///< Number of trades per regime
    std::map<int, double> regime_win_rates;   ///< Win rate percentage by regime
    std::map<int, int> regime_wins;           ///< Winning trades by regime
    std::map<int, int> regime_losses;         ///< Losing trades by regime
    std::map<std::string, double> symbol_performance; ///< Performance by symbol
    std::map<std::string, int> symbol_wins;   ///< Winning trades by symbol
    std::map<std::string, int> symbol_losses; ///< Losing trades by symbol

    double total_return;          ///< Overall portfolio return percentage
    double realized_pnl;          ///< Realized profit/loss from closed positions
    double unrealized_pnl;        ///< Unrealized profit/loss from open positions
    double max_drawdown;          ///< Maximum drawdown experienced
    double current_drawdown;      ///< Current drawdown from peak
    double peak_portfolio_value;  ///< Highest portfolio value achieved
    int total_trades;             ///< Total number of trades executed
    int winning_trades;           ///< Number of profitable trades
    int losing_trades;            ///< Number of losing trades
    int open_positions;           ///< Current number of open positions

    std::vector<TradeResult> trade_history; ///< Complete history of all trades
    std::chrono::time_point<std::chrono::system_clock> start_time; ///< Strategy start time
};