/**
 * @file performance_tracker.cpp
 * @brief Performance tracking implementation
 * 
 * @author Jake Kostoryz
 * @date 2025
 * @version 4.0
 */

#include "../include/performance_tracker.h"

// ============================================================================
// PERFORMANCE TRACKER IMPLEMENTATION
// ============================================================================

PerformanceTracker::PerformanceTracker() 
    : total_return(0.0)
    , realized_pnl(0.0)
    , unrealized_pnl(0.0)
    , max_drawdown(0.0)
    , current_drawdown(0.0)
    , peak_portfolio_value(0.0)
    , total_trades(0)
    , winning_trades(0)
    , losing_trades(0)
    , open_positions(0)
{
    start_time = std::chrono::system_clock::now();
}

void PerformanceTracker::recordTradeOpen(const std::string& symbol, int regime_id) {
    (void)symbol; // Parameter reserved for future per-symbol tracking
    (void)regime_id; // Parameter reserved for future per-regime tracking
    ++open_positions;
}

void PerformanceTracker::recordTradeClose(const TradeResult& result) {
    ++total_trades;
    ++regime_trade_counts[result.regime_id];

    // Classify trade as winning or losing
    const bool is_winning_trade = result.return_pct > 0.0;
    if (is_winning_trade) {
        ++winning_trades;
        ++regime_wins[result.regime_id];
        ++symbol_wins[result.symbol];
    } else {
        ++losing_trades;
        ++regime_losses[result.regime_id];
        ++symbol_losses[result.symbol];
    }

    // Update performance metrics
    total_return += result.return_pct;
    realized_pnl += result.profit_loss;
    regime_returns[result.regime_id] += result.return_pct;
    symbol_performance[result.symbol] += result.return_pct;

    // Recalculate win rates
    const int regime_trades = regime_trade_counts[result.regime_id];
    if (regime_trades > 0) {
        regime_win_rates[result.regime_id] = static_cast<double>(regime_wins[result.regime_id]) / regime_trades;
    }

    // Archive trade and update position count
    trade_history.push_back(result);
    --open_positions;
}

void PerformanceTracker::updateUnrealizedPnL(const std::map<std::string, PositionInfo>& positions,
                                            const std::map<std::string, double>& current_prices) {
    unrealized_pnl = 0.0;
    for (const auto& pos_pair : positions) {
        const std::string& symbol = pos_pair.first;
        const PositionInfo& pos = pos_pair.second;

        auto price_it = current_prices.find(symbol);
        if (price_it != current_prices.end()) {
            double current_price = price_it->second;
            double position_pnl = (current_price - pos.entry_price) * pos.quantity;
            unrealized_pnl += position_pnl;
        }
    }
}

double PerformanceTracker::getWinRate() const {
    return total_trades > 0 ? (double)winning_trades / total_trades : 0.0;
}

double PerformanceTracker::getAvgWin() const {
    if (winning_trades == 0) return 0.0;

    double total_wins = 0.0;
    for (const auto& trade : trade_history) {
        if (trade.return_pct > 0) {
            total_wins += trade.return_pct;
        }
    }
    return total_wins / winning_trades;
}

double PerformanceTracker::getAvgLoss() const {
    if (losing_trades == 0) return 0.0;

    double total_losses = 0.0;
    for (const auto& trade : trade_history) {
        if (trade.return_pct < 0) {
            total_losses += std::abs(trade.return_pct);
        }
    }
    return total_losses / losing_trades;
}

double PerformanceTracker::getProfitFactor() const {
    double avg_win = getAvgWin();
    double avg_loss = getAvgLoss();
    double win_rate = getWinRate();

    if (avg_loss == 0 || win_rate == 0 || win_rate == 1) return 0.0;

    return (avg_win * win_rate) / (avg_loss * (1 - win_rate));
}

void PerformanceTracker::updateDrawdown(double current_portfolio_value) {
    if (current_portfolio_value > peak_portfolio_value) {
        peak_portfolio_value = current_portfolio_value;
        current_drawdown = 0.0;
    } else {
        current_drawdown = (peak_portfolio_value - current_portfolio_value) / peak_portfolio_value;
        max_drawdown = std::max(max_drawdown, current_drawdown);
    }
}

void PerformanceTracker::printPerformance() {
    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::hours>(now - start_time);

    std::cout << "\n📊 STRATEGY PERFORMANCE ANALYTICS" << std::endl;
    std::cout << "===================================" << std::endl;
    std::cout << "⏱️  Runtime: " << duration.count() << " hours" << std::endl;
    std::cout << "💰 Total Return: " << std::setprecision(2) << std::fixed << total_return << "%" << std::endl;
    std::cout << "💵 Realized P&L: $" << std::setprecision(2) << std::fixed << realized_pnl << std::endl;
    std::cout << "📈 Unrealized P&L: $" << std::setprecision(2) << std::fixed << unrealized_pnl << std::endl;
    std::cout << "📉 Max Drawdown: " << std::setprecision(2) << std::fixed << max_drawdown * 100 << "%" << std::endl;

    if (total_trades > 0) {
        std::cout << "\n📊 TRADE STATISTICS:" << std::endl;
        std::cout << "🎯 Win Rate: " << std::setprecision(1) << std::fixed << getWinRate() * 100 << "%" << std::endl;
        std::cout << "✅ Winning Trades: " << winning_trades << std::endl;
        std::cout << "❌ Losing Trades: " << losing_trades << std::endl;
        std::cout << "📂 Open Positions: " << open_positions << std::endl;
        std::cout << "📈 Avg Win: " << std::setprecision(2) << getAvgWin() << "%" << std::endl;
        std::cout << "📉 Avg Loss: " << std::setprecision(2) << getAvgLoss() << "%" << std::endl;
        std::cout << "💹 Profit Factor: " << std::setprecision(2) << getProfitFactor() << std::endl;
    }
    std::cout << "🔢 Total Closed Trades: " << total_trades << std::endl;

    std::cout << "\n📊 PERFORMANCE BY REGIME:" << std::endl;
    for (const auto& regime_return : regime_returns) {
        int regime = regime_return.first;
        double total_ret = regime_return.second;
        int trade_count = regime_trade_counts[regime];
        if (trade_count > 0) {
            double avg_return = total_ret / trade_count;
            double win_rate = regime_win_rates[regime] * 100;
            std::cout << "   Regime " << regime << ": " << std::setprecision(2) << avg_return
                      << "% avg, " << std::setprecision(1) << win_rate << "% win rate ("
                      << regime_wins[regime] << "W/" << regime_losses[regime] << "L)" << std::endl;
        }
    }

    std::cout << "\n🏆 TOP PERFORMING SYMBOLS:" << std::endl;
    std::vector<std::pair<std::string, double>> sorted_symbols(symbol_performance.begin(), symbol_performance.end());
    std::sort(sorted_symbols.begin(), sorted_symbols.end(),
             [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
                 return a.second > b.second;
             });

    int count = 0;
    for (const auto& symbol_perf : sorted_symbols) {
        if (count++ >= 5) break;
        int wins = symbol_wins[symbol_perf.first];
        int losses = symbol_losses[symbol_perf.first];
        double win_rate = (wins + losses) > 0 ? (double)wins / (wins + losses) * 100 : 0;
        std::cout << "   🥇 " << symbol_perf.first << ": " << std::setprecision(2) << symbol_perf.second
                  << "% (" << wins << "W/" << losses << "L, " << std::setprecision(1) << win_rate << "% WR)" << std::endl;
    }
}