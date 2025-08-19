/**
 * @file trading_strategy.h
 * @brief Main trading strategy class definition
 * 
 * @author Jake Kostoryz
 * @date 2025
 * @version 4.0
 */

#pragma once

#include "common.h"
#include "data_structures.h"
#include "performance_tracker.h"
#include "regime_data.h"
#include "alpaca_client.h"

// ============================================================================
// MAIN TRADING STRATEGY
// ============================================================================

/**
 * @brief Enhanced research-based trading strategy implementation
 * 
 * Main trading strategy class that orchestrates regime-based trading decisions,
 * portfolio rebalancing, position management, and performance tracking.
 * Uses research-backed regime switching models for optimal trade execution.
 */
class ResearchBasedTradingStrategy {
private:
    // Core Components
    std::unique_ptr<AlpacaClient> m_alpaca_client;         ///< Alpaca API client
    std::unique_ptr<RegimeDataReader> m_regime_reader;     ///< Regime data reader
    std::unique_ptr<PerformanceTracker> m_performance_tracker; ///< Performance tracking

    // Position Management
    std::map<std::string, PositionInfo> m_open_positions;  ///< Currently open positions
    std::map<std::string, std::string> m_position_orders;  ///< Symbol to order ID mapping
    std::map<std::string, double> m_target_positions;      ///< Target position sizes
    
    // Configuration
    std::map<std::string, std::string> m_symbol_mapping;   ///< Python to Alpaca symbol mapping
    
    // State
    double m_total_portfolio_value;                        ///< Current total portfolio value
    bool m_strategy_active;                                ///< Whether strategy is currently running

    // Position Sizing and Management
    /// @brief Calculate optimal position size based on regime and coin score
    double calculateOptimalPositionSize(const RegimeData& regime, const std::string& python_symbol,
                                       double coin_score, double current_price, double portfolio_value);
    
    /// @brief Calculate target positions for all symbols based on current regime
    void calculateTargetPositions(const RegimeData& regime, double portfolio_value);
    
    /// @brief Convert Python symbol format to Alpaca symbol format
    std::string convertToAlpacaSymbol(const std::string& python_symbol);
    
    /// @brief Convert Alpaca symbol format to Python symbol format
    std::string convertToPythonSymbol(const std::string& alpaca_symbol);
    
    /// @brief Close positions that are too risky in current regime
    void closeRiskyPositions();
    
    /// @brief Calculate adaptive wait time based on regime and market stress
    int calculateAdaptiveWaitTime(const RegimeData& regime, double market_stress);

    // Enhanced Trading Methods (Python Backtest Alignment)
    /// @brief Calculate enhanced coin scores using PC factor analysis
    std::map<std::string, double> calculateEnhancedCoinScores(const RegimeData& regime);
    
    /// @brief Determine if regime should be traded based on EV optimization
    bool shouldTradeRegime(const RegimeData& regime);
    
    /// @brief Get current portfolio exposure percentage
    double getCurrentPortfolioExposure(double portfolio_value);
    
    /// @brief Calculate dynamic stop loss and take profit levels
    std::pair<double, double> calculateDynamicStopLossTakeProfit(const RegimeData& regime, double entry_price);

    // Portfolio Management
    /// @brief Get current holdings across all positions
    std::map<std::string, double> getCurrentHoldings();
    
    /// @brief Calculate adjustment needed for a specific position
    double calculatePositionAdjustment(const RegimeData& regime, const std::string& python_symbol,
                                     double coin_score, double current_price, double portfolio_value,
                                     double current_quantity);
    
    /// @brief Get minimum trade threshold for a symbol to avoid micro-trades
    double getMinimumTradeThreshold(const std::string& python_symbol);
    
    /// @brief Execute position adjustment (buy/sell)
    void executePositionAdjustment(const std::string& alpaca_symbol, double adjustment,
                                  const RegimeData& regime, double current_price);

    // Position Tracking
    /// @brief Track a newly opened position
    void trackOpenPosition(const std::string& symbol, double entry_price, double quantity,
                          double stop_loss, double take_profit, int regime_id, const std::string& order_id);
    
    /// @brief Check for positions that have been closed externally
    void checkClosedPositions();
    
    /// @brief Record a position that has been closed
    void recordClosedPosition(const std::string& symbol, double exit_price, const std::string& exit_reason);
    
    /// @brief Update position tracking with latest market data
    void updatePositionTracking();

public:
    /// @brief Constructor to initialize trading strategy
    ResearchBasedTradingStrategy();
    
    /// @brief Rebalance portfolio based on new regime data
    void rebalancePortfolio(const RegimeData& new_regime);
    
    /// @brief Print current strategy status and performance
    void printStatus();
    
    /// @brief Main trading loop that runs continuously
    void runTradingLoop();
    
    /// @brief Start the trading strategy
    void start();
    
    /// @brief Stop the trading strategy
    void stop();
    
    /// @brief Check if strategy is currently active
    bool isActive() const;
    
    /// @brief Display current holdings and positions
    void showCurrentHoldings();
};