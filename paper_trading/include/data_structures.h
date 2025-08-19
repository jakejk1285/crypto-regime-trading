/**
 * @file data_structures.h
 * @brief Data structures for position tracking and trade results
 * 
 * @author Jake Kostoryz
 * @date 2025
 * @version 4.0
 */

#pragma once

#include "common.h"

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/**
 * @brief Structure to track open trading positions
 * 
 * Contains all necessary information to manage and monitor
 * an active trading position including entry details, risk
 * management parameters, and timing information.
 */
struct PositionInfo {
    std::string symbol;          ///< Trading symbol (e.g., "BTC/USD")
    double entry_price;          ///< Price at which position was opened
    double quantity;             ///< Number of shares/units held
    double stop_loss;            ///< Stop loss price level
    double take_profit;          ///< Take profit price level
    int regime_id;               ///< Market regime when position was opened
    std::chrono::time_point<std::chrono::system_clock> entry_time; ///< Position entry timestamp
    std::string order_id;        ///< Alpaca order ID for tracking
    
    /// @brief Default constructor
    PositionInfo() = default;
    
    /// @brief Parameterized constructor
    PositionInfo(const std::string& sym, double entry, double qty, double sl, double tp, int regime, const std::string& order)
        : symbol(sym), entry_price(entry), quantity(qty), stop_loss(sl), take_profit(tp), 
          regime_id(regime), entry_time(std::chrono::system_clock::now()), order_id(order) {}
};

/**
 * @brief Structure to record completed trade results
 * 
 * Stores comprehensive information about a closed trade including
 * performance metrics, timing, and the reason for exit.
 */
struct TradeResult {
    std::string symbol;          ///< Trading symbol
    double entry_price;          ///< Position entry price
    double exit_price;           ///< Position exit price
    double quantity;             ///< Number of shares/units traded
    double return_pct;           ///< Return percentage for this trade
    double profit_loss;          ///< Absolute profit/loss in USD
    int regime_id;               ///< Market regime during the trade
    std::chrono::duration<double> hold_duration; ///< How long position was held
    std::string exit_reason;     ///< Exit reason: "stop_loss", "take_profit", "manual", "rebalance"
    
    /// @brief Default constructor
    TradeResult() = default;
};