/**
 * @file trading_strategy_ui.cpp
 * @brief Trading strategy UI and loop methods implementation
 * 
 * @author Jake Kostoryz
 * @date 2025
 * @version 4.0
 */

#include "../include/trading_strategy.h"
#include "../include/common.h"

// ============================================================================
// TRADING LOOP AND UI METHODS
// ============================================================================

void ResearchBasedTradingStrategy::runTradingLoop() {
    safeOutput("\n🔄 Starting Enhanced Position-Aware Trading Loop");
    safeOutput("=================================================");

    while (m_strategy_active) {
        try {
            // Check for closed positions first
            checkClosedPositions();
            updatePositionTracking();

            if (!m_regime_reader->loadRegimeData()) {
                safeOutput("⚠️  Failed to load regime data, skipping cycle");
                std::this_thread::sleep_for(std::chrono::minutes(5));
                continue;
            }

            const RegimeData& regime = m_regime_reader->getCurrentRegime();

            // Handle regime changes with enhanced rebalancing
            if (m_regime_reader->hasRegimeChanged()) {
                safeOutput("\n🔄 REGIME CHANGE DETECTED - ENHANCED REBALANCING");
                rebalancePortfolio(regime);
                m_regime_reader->clearRegimeChangeFlag();
                safeOutput("✅ Enhanced portfolio rebalanced for regime: " + regime.regime_name);
            }

            // Check if we should trade
            const double market_stress = m_alpaca_client->getMarketStress(regime);
            bool should_trade = regime.should_trade && market_stress < 0.7;

            if (!should_trade) {
                safeOutput("🛑 Trading conditions not met:");
                safeOutput("   📊 Regime Trading: " + std::string(regime.should_trade ? "YES" : "NO"));
                safeOutput("   🌡️  Market Stress: " + std::to_string(market_stress));

                int wait_time = calculateAdaptiveWaitTime(regime, market_stress);
                safeOutput("⏳ Waiting " + std::to_string(wait_time) + " minutes before next check...");

                for (int i = 0; i < wait_time && m_strategy_active; i++) {
                    std::this_thread::sleep_for(std::chrono::minutes(1));
                    if (i % 5 == 0 && i > 0) {
                        checkClosedPositions();  // Check positions periodically
                        updatePositionTracking();
                    }
                }
                continue;
            }

            // Get portfolio value
            Json::Value account = m_alpaca_client->getAccount();
            double portfolio_value = std::stod(account.get("portfolio_value", "0").asString());
            m_total_portfolio_value = portfolio_value;

            // Calculate target positions
            calculateTargetPositions(regime, portfolio_value);

            // Get current holdings
            std::map<std::string, double> current_holdings = getCurrentHoldings();

            // Execute position adjustments
            safeOutput("\n📊 POSITION ADJUSTMENTS:");
            bool any_adjustments = false;

            for (const auto& coin_score : regime.coin_scores) {
                const std::string& python_symbol = coin_score.first;
                double score = coin_score.second;

                if (score < 0.4) continue;

                std::string alpaca_symbol = convertToAlpacaSymbol(python_symbol);
                double current_qty = current_holdings[alpaca_symbol];
                const double current_price = m_alpaca_client->getCurrentPrice(alpaca_symbol);

                double adjustment = calculatePositionAdjustment(regime, python_symbol, score,
                                                              current_price, portfolio_value, current_qty);

                if (std::abs(adjustment) > getMinimumTradeThreshold(python_symbol)) {
                    executePositionAdjustment(alpaca_symbol, adjustment, regime, current_price);
                    any_adjustments = true;
                }
            }

            if (!any_adjustments) {
                safeOutput("   ✅ No adjustments needed - positions aligned with targets");
            }

            // Update performance tracking
            m_performance_tracker->updateDrawdown(portfolio_value);

            // Adaptive wait time
            int wait_minutes = calculateAdaptiveWaitTime(regime, market_stress);
            safeOutput("\n⏳ Next check in " + std::to_string(wait_minutes) + " minutes...");

            // Wait with periodic position checks
            for (int i = 0; i < wait_minutes && m_strategy_active; i++) {
                std::this_thread::sleep_for(std::chrono::minutes(1));
                if (i % 3 == 0 && i > 0) {
                    checkClosedPositions();  // Check for closed positions every 3 minutes
                    updatePositionTracking();
                }
            }

        } catch (const std::exception& e) {
            safeOutput("❌ Error in trading loop: " + std::string(e.what()));
            std::this_thread::sleep_for(std::chrono::minutes(1));
        }
    }

    safeOutput("\n🛑 Trading loop stopped");
}

void ResearchBasedTradingStrategy::printStatus() {
    try {
        safeOutput("\n📊 ENHANCED TRADING STRATEGY STATUS");
        safeOutput("====================================");

        // Account info
        Json::Value account = m_alpaca_client->getAccount();
        if (!account.empty()) {
            double portfolio_value = std::stod(account.get("portfolio_value", "0").asString());
            double buying_power = std::stod(account.get("buying_power", "0").asString());

            safeOutput("\n💰 ACCOUNT INFORMATION:");
            safeOutput("   🏦 Buying Power: $" + std::to_string(buying_power));
            safeOutput("   📈 Portfolio Value: $" + std::to_string(portfolio_value));

            m_performance_tracker->updateDrawdown(portfolio_value);
            m_total_portfolio_value = portfolio_value;

            safeOutput("   📉 Current Drawdown: " + std::to_string(m_performance_tracker->current_drawdown * 100) + "%");
            safeOutput("   📊 Max Drawdown: " + std::to_string(m_performance_tracker->max_drawdown * 100) + "%");
        }

        // Current positions with real-time P&L
        safeOutput("\n📈 CURRENT POSITIONS:");
        Json::Value positions = m_alpaca_client->getPositions();
        if (positions.isArray() && positions.size() > 0) {
            safeOutput("   Symbol      |    Quantity |  Avg Price | Curr Price |     P&L    |    P&L %   | Target Qty");
            safeOutput("   ------------|-------------|------------|------------|------------|------------|------------");

            for (const auto& pos : positions) {
                std::string symbol = pos["symbol"].asString();
                double qty = std::stod(pos["qty"].asString());
                double avg_price = std::stod(pos["avg_entry_price"].asString());
                double current_price = std::stod(pos["current_price"].asString());
                double market_value = std::stod(pos["market_value"].asString());
                double cost_basis = std::stod(pos["cost_basis"].asString());
                double unrealized_pl = market_value - cost_basis;
                double pl_percentage = (unrealized_pl / cost_basis) * 100.0;

                std::string python_symbol = convertToPythonSymbol(symbol);
                double target_qty = m_target_positions.count(python_symbol) > 0 ? m_target_positions[python_symbol] : 0.0;

                // Update tracked position's current status
                std::string status_indicator = m_open_positions.count(symbol) > 0 ? " T " : "   ";

                safeOutput(status_indicator + symbol + 
                          " | " + std::to_string(qty) + 
                          " | $" + std::to_string(avg_price) + 
                          " | $" + std::to_string(current_price) + 
                          " | $" + std::to_string(unrealized_pl) + 
                          " | " + std::to_string(pl_percentage) + "%" + 
                          " | " + std::to_string(target_qty));
            }

            safeOutput("\n   T = Tracked position with stop-loss/take-profit orders");
        } else {
            safeOutput("   📭 No current positions");
        }

        // Enhanced regime analysis
        if (m_regime_reader->loadRegimeData()) {
            const RegimeData& regime = m_regime_reader->getCurrentRegime();

            safeOutput("\n🎯 ENHANCED REGIME ANALYSIS:");
            safeOutput("   📊 Regime: " + regime.regime_name + " (ID: " + std::to_string(regime.regime_id) + ")");
            safeOutput("   🎯 Confidence: " + std::to_string(regime.confidence));
            safeOutput("   📈 Strategy: " + regime.strategy);
            safeOutput("   ⚡ PC1 (Market): " + std::to_string(regime.pc1_market_factor) + " ± " + std::to_string(regime.pc1_std));
            safeOutput("   📊 PC2 (Volatility): " + std::to_string(regime.pc2_volatility_factor) + " ± " + std::to_string(regime.pc2_std));
            safeOutput("   🔒 Persistence: " + std::to_string(regime.persistence));
            safeOutput("   ⏱️  Avg Duration: " + std::to_string(regime.avg_duration) + " days");
            safeOutput("   💹 Should Trade: " + std::string(regime.should_trade ? "YES ✅" : "NO ❌"));

            // Top coins for current regime
            if (!regime.coin_scores.empty()) {
                safeOutput("\n🏆 TOP REGIME COINS:");
                std::vector<std::pair<std::string, double>> sorted_coins(regime.coin_scores.begin(), regime.coin_scores.end());
                std::sort(sorted_coins.begin(), sorted_coins.end(),
                         [](const auto& a, const auto& b) { return a.second > b.second; });

                int count = 0;
                for (const auto& coin_score : sorted_coins) {
                    if (count++ >= 5) break;
                    if (coin_score.second > 0.5) {
                        const std::string& coin = coin_score.first;
                        double score = coin_score.second;
                        std::string alpaca_symbol = convertToAlpacaSymbol(coin);
                        double target = m_target_positions.count(coin) > 0 ? m_target_positions[coin] : 0.0;
                        safeOutput("   🥇 " + coin + " (" + alpaca_symbol + "): Score " +
                                  std::to_string(score) + " | Target: " + std::to_string(target));
                    }
                }
            }

        } else {
            safeOutput("\n🎯 REGIME ANALYSIS: ❌ Error reading regime data");
        }

        // Performance summary with real metrics
        m_performance_tracker->printPerformance();

    } catch (const std::exception& e) {
        safeOutput("❌ Error in printStatus: " + std::string(e.what()));
    }
}

void ResearchBasedTradingStrategy::showCurrentHoldings() {
    try {
        safeOutput("\n📊 ENHANCED HOLDINGS & POSITION ANALYSIS");
        safeOutput("========================================");

        Json::Value account = m_alpaca_client->getAccount();
        if (!account.empty()) {
            double portfolio_value = std::stod(account.get("portfolio_value", "0").asString());
            double cash = std::stod(account.get("cash", "0").asString());
            double long_market_value = std::stod(account.get("long_market_value", "0").asString());

            safeOutput("\n💼 PORTFOLIO OVERVIEW:");
            safeOutput("   📊 Total Value: $" + std::to_string(portfolio_value));
            safeOutput("   💵 Cash: $" + std::to_string(cash));
            safeOutput("   📈 Positions Value: $" + std::to_string(long_market_value));
            safeOutput("   🎯 Cash Allocation: " + std::to_string((cash / portfolio_value) * 100) + "%");
        }

        Json::Value positions = m_alpaca_client->getPositions();
        if (positions.isArray() && positions.size() > 0) {
            safeOutput("\n📈 DETAILED POSITION ANALYSIS:");
            safeOutput("Symbol      | Quantity    | Entry    | Current  | Value      | P&L       | %Gain  | %Port | Status");
            safeOutput("------------|-------------|----------|----------|------------|-----------|--------|-------|--------");

            double total_unrealized_pl = 0.0;
            for (const auto& pos : positions) {
                std::string symbol = pos["symbol"].asString();
                double qty = std::stod(pos["qty"].asString());
                double avg_price = std::stod(pos["avg_entry_price"].asString());
                double current_price = std::stod(pos["current_price"].asString());
                double market_value = std::stod(pos["market_value"].asString());
                double cost_basis = std::stod(pos["cost_basis"].asString());
                double unrealized_pl = market_value - cost_basis;
                double pl_percentage = (unrealized_pl / cost_basis) * 100.0;
                double portfolio_percent = (market_value / m_total_portfolio_value) * 100.0;

                total_unrealized_pl += unrealized_pl;

                // Check if position is tracked
                std::string status = "Untracked";
                if (m_open_positions.count(symbol) > 0) {
                    const PositionInfo& tracked = m_open_positions[symbol];
                    if (current_price <= tracked.stop_loss) {
                        status = "Near SL";
                    } else if (current_price >= tracked.take_profit) {
                        status = "Near TP";
                    } else {
                        status = "Tracked";
                    }
                }

                safeOutput(symbol + " | " + std::to_string(qty) + " | $" + std::to_string(avg_price) + 
                          " | $" + std::to_string(current_price) + " | $" + std::to_string(market_value) + 
                          " | $" + std::to_string(unrealized_pl) + " | " + std::to_string(pl_percentage) + "%" + 
                          " | " + std::to_string(portfolio_percent) + "% | " + status);

                // Show stop-loss and take-profit levels for tracked positions
                if (m_open_positions.count(symbol) > 0) {
                    const PositionInfo& tracked = m_open_positions[symbol];
                    safeOutput("            | SL: $" + std::to_string(tracked.stop_loss) + 
                              " | TP: $" + std::to_string(tracked.take_profit));
                }
            }

            safeOutput("\n📊 POSITION SUMMARY:");
            safeOutput("   💰 Total Unrealized P&L: $" + std::to_string(total_unrealized_pl));
            safeOutput("   📊 Number of Positions: " + std::to_string(positions.size()));
            safeOutput("   🎯 Tracked Positions: " + std::to_string(m_open_positions.size()));
        } else {
            safeOutput("\n📭 No current positions");
        }

        // Show recent closed trades
        if (!m_performance_tracker->trade_history.empty()) {
            safeOutput("\n📜 RECENT CLOSED TRADES (Last 5):");
            safeOutput("Symbol      | Entry    | Exit     | Return  | P&L      | Duration | Reason");
            safeOutput("------------|----------|----------|---------|----------|----------|--------");

            int count = 0;
            for (auto it = m_performance_tracker->trade_history.rbegin();
                 it != m_performance_tracker->trade_history.rend() && count < 5; ++it, ++count) {
                const TradeResult& trade = *it;
                auto hours = std::chrono::duration_cast<std::chrono::hours>(trade.hold_duration).count();

                safeOutput(trade.symbol + " | $" + std::to_string(trade.entry_price) + 
                          " | $" + std::to_string(trade.exit_price) + " | " + std::to_string(trade.return_pct) + "%" +
                          " | $" + std::to_string(trade.profit_loss) + " | " + std::to_string(hours) + "h" +
                          " | " + trade.exit_reason);
            }
        }

    } catch (const std::exception& e) {
        safeOutput("❌ Error showing holdings: " + std::string(e.what()));
    }
}