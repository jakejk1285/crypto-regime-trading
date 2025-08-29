/**
 * @file trading_strategy.cpp
 * @brief Trading strategy implementation
 * 
 * @author Jake Kostoryz
 * @date 2025
 * @version 4.1
 * 
 * Updated to match current backtest implementation:
 * - Multi-regime trading approach (54.35% return, 1.437 Sharpe)
 * - WAIT_AND_SEE regime handling (close all positions)
 * - Updated allocation percentages based on performance
 * - Revised trading thresholds and stop-loss logic
 */

#include "../include/trading_strategy.h"
#include "../include/common.h"

// ============================================================================
// TRADING STRATEGY IMPLEMENTATION
// ============================================================================

ResearchBasedTradingStrategy::ResearchBasedTradingStrategy()
    : m_total_portfolio_value(0.0)
    , m_strategy_active(false)
{
    try {
        // Initialize core components
        m_alpaca_client = std::make_unique<AlpacaClient>();
        m_regime_reader = std::make_unique<RegimeDataReader>(
            "../shared_regime_data/regime_output/regime_for_cpp.json");
        m_performance_tracker = std::make_unique<PerformanceTracker>();

        // Configure symbol mappings (Python -> Alpaca format)
        m_symbol_mapping = {
            {"BTCUSD", "BTC/USD"},
            {"ETHUSD", "ETH/USD"},
            {"ADAUSD", "ADA/USD"},
            {"DOTUSD", "DOT/USD"},
            {"LINKUSD", "LINK/USD"},
            {"SOLUSD", "SOL/USD"},
            {"MATICUSD", "MATIC/USD"},
            {"AVAXUSD", "AVAX/USD"},
            {"ATOMUSD", "ATOM/USD"},
            {"ALGOUSD", "ALGO/USD"}
        };

        safeOutput("✅ Enhanced Trading Strategy initialized successfully");
        
    } catch (const std::exception& e) {
        safeOutput("❌ Failed to initialize strategy: " + std::string(e.what()));
        throw;
    }
}

void ResearchBasedTradingStrategy::start() {
    if (m_strategy_active) {
        safeOutput("⚠️  Strategy is already running");
        return;
    }

    m_strategy_active = true;
    safeOutput("\n🚀 Starting Enhanced Research-Based Trading Strategy");

    // Display initial status
    printStatus();
}

void ResearchBasedTradingStrategy::stop() {
    m_strategy_active = false;
    safeOutput("\n🛑 Stopping trading strategy...");
}

bool ResearchBasedTradingStrategy::isActive() const {
    return m_strategy_active;
}

std::map<std::string, double> ResearchBasedTradingStrategy::getCurrentHoldings() {
    std::map<std::string, double> holdings;

    try {
        Json::Value positions = m_alpaca_client->getPositions();
        if (positions.isArray()) {
            for (const auto& pos : positions) {
                const std::string symbol = pos["symbol"].asString();
                const double qty = std::stod(pos["qty"].asString());
                holdings[symbol] = qty;
            }
        }
    } catch (const std::exception& e) {
        safeOutput("❌ Error getting holdings: " + std::string(e.what()));
    }

    return holdings;
}

double ResearchBasedTradingStrategy::calculateOptimalPositionSize(const RegimeData& regime, const std::string& python_symbol,
                                                                double coin_score, double current_price, double portfolio_value) {
    (void)python_symbol; // Parameter reserved for future per-symbol adjustments
    if (current_price <= 0 || portfolio_value <= 0) return 0.0;

    // UPDATED ALLOCATION - Based on current backtest performance (54.35% return)
    // Current backtest trades multiple regimes successfully with dynamic allocation
    std::map<std::string, double> regime_risk = {
        // Non-tradeable regimes
        {"CRISIS", 0.0},              
        {"WAIT_AND_SEE", 0.0},        // Always close positions
        
        // Tradeable regimes with performance-based allocation
        {"STABLE_GROWTH", 0.20},      // 20% - Strong performers in backtest  
        {"MOMENTUM", 0.15},           // 15% - Moderate allocation
        {"BALANCED", 0.15},           // 15% - Baseline market conditions
        {"BREAKOUT", 0.15},           // 15% - Breakout momentum
        {"DEFENSIVE", 0.10},          // 10% - Conservative allocation
        {"EXTREME_VOLATILITY", 0.05}, // 5% - Minimal but present
        {"CONSERVATIVE", 0.10},       // 10% - Conservative stable
    };
    
    double base_percent = 0.15; // Default fallback
    auto it = regime_risk.find(regime.strategy);
    if (it != regime_risk.end()) {
        base_percent = it->second;
    }
    
    // Volatility adjustment (reduce size for high vol assets)
    double pc2_factor = std::abs(regime.pc2_volatility_factor);
    double vol_adjustment = (pc2_factor > 1.0) ? std::max(0.5, 1.0 - (pc2_factor - 1.0) * 0.1) : 1.0;
    
    // Persistence bonus (more confident = larger size)
    double persistence_bonus = 1.0 + (regime.persistence - 0.5) * 0.4;
    
    // Portfolio concentration limit (relaxed for better capital utilization)
    double current_exposure = getCurrentPortfolioExposure(portfolio_value);
    double concentration_limit = std::max(0.7, 0.95 - current_exposure);  // Allow higher exposure
    
    // Final position size calculation
    double adjusted_percent = base_percent * vol_adjustment * persistence_bonus * 
                             coin_score * concentration_limit;
    
    double position_value = portfolio_value * adjusted_percent;
    double position_qty = position_value / current_price;

    return position_qty;
}

double ResearchBasedTradingStrategy::calculatePositionAdjustment(const RegimeData& regime, const std::string& python_symbol,
                                                               double coin_score, double current_price, double portfolio_value,
                                                               double current_quantity) {
    double target_quantity = calculateOptimalPositionSize(regime, python_symbol, coin_score, current_price, portfolio_value);
    return target_quantity - current_quantity;
}

double ResearchBasedTradingStrategy::getMinimumTradeThreshold(const std::string& python_symbol) {
    (void)python_symbol; // Parameter reserved for future per-symbol thresholds
    // Minimum trade value of $100
    return 100.0;
}

void ResearchBasedTradingStrategy::rebalancePortfolio(const RegimeData& new_regime) {
    safeOutput("\n🔄 PORTFOLIO REBALANCING FOR NEW REGIME");
    safeOutput("=======================================");

    try {
        // Check if we should trade this regime (EV-optimized strategy)
        if (!shouldTradeRegime(new_regime)) {
            if (new_regime.strategy == "WAIT_AND_SEE") {
                // Close all positions only for WAIT_AND_SEE regime (uncertainty)
                std::map<std::string, double> current_holdings = getCurrentHoldings();
                for (const auto& [alpaca_symbol, holding_qty] : current_holdings) {
                    safeOutput("🔻 Closing position (WAIT_AND_SEE): " + alpaca_symbol);
                    m_alpaca_client->closePosition(alpaca_symbol);
                    if (m_open_positions.count(alpaca_symbol) > 0) {
                        const double current_price = m_alpaca_client->getCurrentPrice(alpaca_symbol);
                        recordClosedPosition(alpaca_symbol, current_price, "wait_and_see_regime_exit");
                    }
                }
                safeOutput("🚫 WAIT_AND_SEE regime - closed all positions");
            } else {
                safeOutput("🛡️  " + new_regime.strategy + " regime - conditions not met, maintaining existing positions only");
            }
            return;
        }

        // Calculate coin scores using enhanced algorithm
        std::map<std::string, double> enhanced_coin_scores = calculateEnhancedCoinScores(new_regime);
        
        // Close positions for coins not in new regime
        std::map<std::string, double> current_holdings = getCurrentHoldings();

        for (const auto& [alpaca_symbol, holding_qty] : current_holdings) {
            const std::string python_symbol = convertToPythonSymbol(alpaca_symbol);

            const auto score_it = enhanced_coin_scores.find(python_symbol);
            if (score_it == enhanced_coin_scores.end() || score_it->second < 0.4) {
                safeOutput("🔻 Closing position: " + alpaca_symbol);
                m_alpaca_client->closePosition(alpaca_symbol);

                // Record position closure if tracked
                if (m_open_positions.count(alpaca_symbol) > 0) {
                    const double current_price = m_alpaca_client->getCurrentPrice(alpaca_symbol);
                    recordClosedPosition(alpaca_symbol, current_price, "regime_rebalance");
                }
            }
        }

    } catch (const std::exception& e) {
        safeOutput("❌ Error in rebalancing: " + std::string(e.what()));
    }
}

std::string ResearchBasedTradingStrategy::convertToAlpacaSymbol(const std::string& python_symbol) {
    const auto it = m_symbol_mapping.find(python_symbol);
    return (it != m_symbol_mapping.end()) ? it->second : python_symbol;
}

std::string ResearchBasedTradingStrategy::convertToPythonSymbol(const std::string& alpaca_symbol) {
    for (const auto& mapping : m_symbol_mapping) {
        if (mapping.second == alpaca_symbol) {
            return mapping.first;
        }
    }
    return alpaca_symbol;
}

void ResearchBasedTradingStrategy::calculateTargetPositions(const RegimeData& regime, double portfolio_value) {
    m_target_positions.clear();

    for (const auto& [python_symbol, coin_score] : regime.coin_scores) {
        if (coin_score > 0.6) {
            const std::string alpaca_symbol = convertToAlpacaSymbol(python_symbol);
            const double current_price = m_alpaca_client->getCurrentPrice(alpaca_symbol);
            const double target_qty = calculateOptimalPositionSize(regime, python_symbol, coin_score,
                                                                  current_price, portfolio_value);
            m_target_positions[python_symbol] = target_qty;
        }
    }
}

int ResearchBasedTradingStrategy::calculateAdaptiveWaitTime(const RegimeData& regime, double market_stress) {
    int base_wait = 15;

    if (regime.avg_duration < 5.0) {
        base_wait = 10;
    } else if (regime.persistence > 0.9) {
        base_wait = 20;
    }

    if (market_stress > 0.7) {
        base_wait = 5;
    }

    return std::max(base_wait, 3);
}

// Position tracking methods
void ResearchBasedTradingStrategy::trackOpenPosition(const std::string& symbol, double entry_price,
                                                   double quantity, double stop_loss, double take_profit,
                                                   int regime_id, const std::string& order_id) {
    const PositionInfo pos(symbol, entry_price, quantity, stop_loss, take_profit, regime_id, order_id);
    
    m_open_positions[symbol] = pos;
    m_position_orders[symbol] = order_id;

    m_performance_tracker->recordTradeOpen(symbol, regime_id);

    safeOutput("📝 Tracking new position: " + symbol +
               " @ $" + std::to_string(entry_price) +
               " (Qty: " + std::to_string(quantity) + ")");
}

void ResearchBasedTradingStrategy::checkClosedPositions() {
    // Get current positions from Alpaca
    Json::Value current_positions = m_alpaca_client->getPositions();
    std::map<std::string, bool> current_symbols;

    // Mark all current positions
    if (current_positions.isArray()) {
        for (const auto& pos : current_positions) {
            const std::string symbol = pos["symbol"].asString();
            current_symbols[symbol] = true;
        }
    }

    // Check for closed positions
    std::vector<std::string> closed_positions;
    for (const auto& [symbol, tracked_pos] : m_open_positions) {
        if (current_symbols.find(symbol) == current_symbols.end()) {
            closed_positions.push_back(symbol);
        }
    }

    // Process closed positions
    for (const std::string& symbol : closed_positions) {
        // Get order history to find exit price
        Json::Value orders = m_alpaca_client->getOrders("all");
        double exit_price = 0.0;
        std::string exit_reason = "unknown";

        if (orders.isArray()) {
            for (const auto& order : orders) {
                if (order["symbol"].asString() == symbol &&
                    order["status"].asString() == "filled" &&
                    order["side"].asString() == "sell") {

                    exit_price = std::stod(order["filled_avg_price"].asString());

                    // Determine exit reason based on price
                    const auto pos_it = m_open_positions.find(symbol);
                    if (pos_it != m_open_positions.end()) {
                        const PositionInfo& pos_info = pos_it->second;
                        const double price_tolerance = 0.01;
                        
                        if (std::abs(exit_price - pos_info.stop_loss) < price_tolerance) {
                            exit_reason = "stop_loss";
                        } else if (std::abs(exit_price - pos_info.take_profit) < price_tolerance) {
                            exit_reason = "take_profit";
                        } else {
                            exit_reason = "manual";
                        }
                    }
                    break;
                }
            }
        }

        if (exit_price > 0.0) {
            recordClosedPosition(symbol, exit_price, exit_reason);
        }
    }
}

void ResearchBasedTradingStrategy::recordClosedPosition(const std::string& symbol, double exit_price,
                                                      const std::string& exit_reason) {
    const auto pos_it = m_open_positions.find(symbol);
    if (pos_it == m_open_positions.end()) return;

    const PositionInfo& pos = pos_it->second;

    // Calculate comprehensive trade result
    TradeResult result;
    result.symbol = symbol;
    result.entry_price = pos.entry_price;
    result.exit_price = exit_price;
    result.quantity = pos.quantity;
    result.return_pct = ((exit_price - pos.entry_price) / pos.entry_price) * 100.0;
    result.profit_loss = (exit_price - pos.entry_price) * pos.quantity;
    result.regime_id = pos.regime_id;
    result.hold_duration = std::chrono::system_clock::now() - pos.entry_time;
    result.exit_reason = exit_reason;

    // Archive the completed trade
    m_performance_tracker->recordTradeClose(result);

    // Clean up tracking structures
    m_open_positions.erase(symbol);
    m_position_orders.erase(symbol);

    safeOutput("📊 Position closed: " + symbol +
               " | Return: " + std::to_string(result.return_pct) + "%" +
               " | P&L: $" + std::to_string(result.profit_loss) +
               " | Reason: " + exit_reason);
}

void ResearchBasedTradingStrategy::updatePositionTracking() {
    // Collect symbols for price updates
    std::vector<std::string> symbols;
    symbols.reserve(m_open_positions.size());
    
    for (const auto& [symbol, pos] : m_open_positions) {
        symbols.push_back(symbol);
    }

    if (!symbols.empty()) {
        const auto current_prices = m_alpaca_client->getBatchPrices(symbols);
        m_performance_tracker->updateUnrealizedPnL(m_open_positions, current_prices);
    }
}

// Updated executePositionAdjustment with proper tracking
void ResearchBasedTradingStrategy::executePositionAdjustment(const std::string& alpaca_symbol, double adjustment,
                                                            const RegimeData& regime, double current_price) {
    if (std::abs(adjustment) < 0.00001) return;

    std::string side = adjustment > 0 ? "buy" : "sell";
    double quantity = std::abs(adjustment);

    // Calculate stop loss and take profit for new positions using dynamic calculation
    double stop_loss = 0.0;
    double take_profit = 0.0;

    if (side == "buy") {
        auto stop_take_pair = calculateDynamicStopLossTakeProfit(regime, current_price);
        stop_loss = stop_take_pair.first;
        take_profit = stop_take_pair.second;
    }

    safeOutput("🔄 Executing " + side + " order for " + alpaca_symbol +
              ": " + std::to_string(quantity) + " units");

    if (stop_loss > 0) {
        safeOutput("   🛑 Stop Loss: $" + std::to_string(stop_loss));
        safeOutput("   🎯 Take Profit: $" + std::to_string(take_profit));
    }

    const bool success = m_alpaca_client->placeOrder(alpaca_symbol, side, quantity, "market", stop_loss, take_profit);

    if (success) {
        safeOutput("   ✅ Order executed successfully");

        // Track new position (only for buys)
        if (side == "buy") {
            // Generate temporary order ID (in production, use actual Alpaca order ID)
            const std::string order_id = "order_" + 
                std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
            trackOpenPosition(alpaca_symbol, current_price, quantity, stop_loss, take_profit, 
                            regime.regime_id, order_id);
        }
    } else {
        safeOutput("   ❌ Order failed");
    }
}

// ============================================================================
// ENHANCED TRADING METHODS (Python Backtest Alignment)
// ============================================================================

std::map<std::string, double> ResearchBasedTradingStrategy::calculateEnhancedCoinScores(const RegimeData& regime) {
    std::map<std::string, double> coin_scores;
    
    // Extract all PC factors for comprehensive analysis
    double pc1_factor = regime.pc1_market_factor;
    double pc2_factor = regime.pc2_volatility_factor;
    double pc3_factor = regime.pc3_factor;
    // Reserve for future PC factor analysis
    (void)regime.pc4_factor;
    (void)regime.pc5_factor;
    
    double persistence = regime.persistence;
    double frequency_pct = regime.frequency_percentage;
    double market_stress = regime.market_stress_level;
    std::string strategy = regime.strategy;

    // Crypto symbol mapping
    std::vector<std::string> crypto_symbols = {
        "BTCUSD", "ETHUSD", "ADAUSD", "DOTUSD", "LINKUSD", "SOLUSD",
        "MATICUSD", "AVAXUSD", "ATOMUSD", "ALGOUSD", "USDTUSD", "XRPUSD",
        "BNBUSD", "DOGEUSD", "LTCUSD"
    };

    for (const std::string& symbol : crypto_symbols) {
        // Enhanced base score with regime stability (increased base for better selection)
        double regime_strength = (persistence * 0.35 + (frequency_pct / 100) * 0.25);
        double market_condition = std::min(1.0, std::max(0.0, 1.0 - market_stress));
        double base_score = regime_strength * market_condition + 0.4;  // Increased from 0.3
        
        // Dynamic PC factor scoring based on regime strategy
        double pc_score = 0.0;
        
        if (strategy == "MOMENTUM" || strategy == "BREAKOUT") {
            // Favor momentum coins in momentum regimes
            if (symbol == "SOLUSD" || symbol == "AVAXUSD" || symbol == "LINKUSD" || symbol == "BNBUSD") {
                pc_score += std::max(0.0, pc1_factor * 0.1) + std::abs(pc3_factor) * 0.05;
            } else if (symbol == "BTCUSD" || symbol == "ETHUSD") {
                pc_score += std::max(0.0, pc1_factor * 0.05);
            }
        } else if (strategy == "STABLE_GROWTH" || strategy == "DEFENSIVE") {
            // Favor stable coins in defensive regimes
            if (symbol == "BTCUSD" || symbol == "ETHUSD" || symbol == "USDTUSD") {
                pc_score += 0.15 + std::max(0.0, -pc2_factor * 0.1);  // Bonus for low volatility
            }
        } else if (strategy == "CRISIS") {
            // Flight to quality in crisis
            if (symbol == "BTCUSD" || symbol == "USDTUSD") {
                pc_score += 0.25;
            } else {
                pc_score -= 0.10;  // Penalty for altcoins
            }
        }
        
        // Volatility adjustment based on PC2
        double volatility_penalty = (std::abs(pc2_factor) > 2.0) ? std::abs(pc2_factor) * 0.02 : 0;
        
        // Final score with bounds checking
        double final_score = std::max(0.0, std::min(1.0, base_score + pc_score - volatility_penalty));
        coin_scores[symbol] = final_score;
    }

    return coin_scores;
}

bool ResearchBasedTradingStrategy::shouldTradeRegime(const RegimeData& regime) {
    std::string strategy = regime.strategy;
    int regime_id = regime.regime_id;
    
    // UPDATED REGIME TRADING LOGIC - Based on current backtest performance
    // Current backtest shows strong performance across multiple regimes
    
    // Always avoid WAIT_AND_SEE regime (uncertainty)
    if (strategy == "WAIT_AND_SEE") {
        return false;  
    }
    
    // Sharp correction regime - only trade if not in extreme stress
    if (regime_id == 1 && regime.market_stress_level > 0.85) {
        return false;  // Avoid severe correction phases
    }
    
    // All other regimes are tradeable with appropriate position sizing
    // This matches the current Python backtest that trades multiple regimes successfully
    return (
        regime.persistence > 0.4 &&        // Reduced threshold for more opportunities  
        regime.market_stress_level < 0.9 && // Allow higher stress levels
        regime.should_trade                 // Basic trade flag
    );
}

double ResearchBasedTradingStrategy::getCurrentPortfolioExposure(double portfolio_value) {
    if (portfolio_value <= 0) return 0.0;
    
    double total_position_value = 0.0;
    try {
        Json::Value positions = m_alpaca_client->getPositions();
        if (positions.isArray()) {
            for (const auto& pos : positions) {
                (void)std::stod(pos["qty"].asString()); // Quantity not needed for exposure calc
                double market_value = std::stod(pos["market_value"].asString());
                total_position_value += std::abs(market_value);
            }
        }
    } catch (const std::exception& e) {
        safeOutput("⚠️ Error calculating portfolio exposure: " + std::string(e.what()));
    }
    
    return total_position_value / portfolio_value;
}

std::pair<double, double> ResearchBasedTradingStrategy::calculateDynamicStopLossTakeProfit(const RegimeData& regime, double entry_price) {
    double pc2_vol = std::abs(regime.pc2_volatility_factor);
    std::string strategy = regime.strategy;
    double avg_duration = regime.avg_duration;
    double persistence = regime.persistence;
    double market_stress = regime.market_stress_level;
    
    // Enhanced volatility-adjusted stop loss - Updated to match current backtest
    std::map<std::string, double> base_stop_loss = {
        {"CRISIS", 0.06},           // Tighter stops in crisis (6%)
        {"BALANCED", 0.04},         // Tight stops for baseline regime (4%)  
        {"STABLE_GROWTH", 0.04},    // Tight stops for stable regime (4%)
        {"MOMENTUM", 0.05},         // Moderate stops for momentum (5%)
        {"BREAKOUT", 0.06},         // Wider stops for breakouts (6%)
        {"DEFENSIVE", 0.08},        // Wider stops for defensive (8%)
        {"EXTREME_VOLATILITY", 0.10}, // Widest stops for extreme vol (10%)
        {"CONSERVATIVE", 0.05}      // Moderate stops for conservative (5%)
    };
    
    double base_stop = 0.05;  // Default
    auto it = base_stop_loss.find(strategy);
    if (it != base_stop_loss.end()) {
        base_stop = it->second;
    }
    
    // Volatility adjustment - tighter stops for low vol, wider for high vol
    double vol_adjustment = std::max(0.7, std::min(1.5, 1.0 + (pc2_vol - 1.0) * 0.2));
    
    // Persistence adjustment - tighter stops for stable regimes
    double persistence_adjustment = std::max(0.8, std::min(1.2, 2.0 - persistence));
    
    // Market stress adjustment - wider stops during stress
    double stress_adjustment = std::max(1.0, std::min(1.3, 1.0 + market_stress * 0.3));
    
    // Final stop loss calculation
    double stop_loss_pct = base_stop * vol_adjustment * persistence_adjustment * stress_adjustment;
    stop_loss_pct = std::max(0.02, std::min(0.12, stop_loss_pct));  // Cap between 2-12%
    
    // Enhanced take profit with asymmetric risk/reward - Updated to match backtest
    std::map<std::string, double> risk_reward_ratios = {
        {"BALANCED", 3.0},          // 3:1 R/R for baseline regime
        {"BREAKOUT", 2.5},          // 2.5:1 R/R for breakouts
        {"STABLE_GROWTH", 2.0},     // 2:1 R/R for stable growth
        {"MOMENTUM", 2.0},          // 2:1 R/R for momentum
        {"CRISIS", 1.5},            // 1.5:1 R/R for crisis
        {"DEFENSIVE", 1.0},         // 1:1 R/R for defensive
        {"EXTREME_VOLATILITY", 1.0}, // 1:1 R/R for extreme vol
        {"CONSERVATIVE", 2.0}       // 2:1 R/R for conservative
    };
    
    double risk_reward = 2.0;  // Default
    auto rr_it = risk_reward_ratios.find(strategy);
    if (rr_it != risk_reward_ratios.end()) {
        risk_reward = rr_it->second;
    }
    
    double take_profit_pct = stop_loss_pct * risk_reward;
    
    // Duration bonus for longer-term regimes
    if (avg_duration > 10) {
        take_profit_pct *= 1.2;  // 20% bonus for longer regimes
    }
    
    double stop_loss = entry_price * (1.0 - stop_loss_pct);
    double take_profit = entry_price * (1.0 + take_profit_pct);
    
    return std::make_pair(stop_loss, take_profit);
}

// Continue with remaining methods... (runTradingLoop, printStatus, showCurrentHoldings)
// Due to length constraints, I'll create these in the next files