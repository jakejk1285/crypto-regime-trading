/**
 * @file alpaca_client.h
 * @brief Alpaca Markets API client for trading operations
 * 
 * @author Jake Kostoryz
 * @date 2025
 * @version 4.0
 */

#pragma once

#include "common.h"
#include "regime_data.h"

// ============================================================================
// MARKET DATA AND TRADING CLIENT
// ============================================================================

/**
 * @brief Alpaca Markets API client for trading operations
 * 
 * Handles all communication with Alpaca Markets API including account management,
 * position tracking, order execution, and market data retrieval. Also integrates
 * with CoinGecko for additional pricing data.
 */
class AlpacaClient {
private:
    std::string m_api_key;        ///< Alpaca API key
    std::string m_secret_key;     ///< Alpaca secret key
    std::string m_base_url;       ///< Base URL for Alpaca API
    std::string m_trading_url;    ///< Trading-specific URL
    std::string m_data_url;       ///< Market data URL
    std::string m_coingecko_key;  ///< CoinGecko API key
    bool m_paper_trading;         ///< Whether using paper trading mode

    /// @brief Load configuration from config file
    void loadConfig();
    
    /// @brief Make HTTP request to Alpaca API
    std::string makeRequest(const std::string& endpoint, const std::string& method = "GET",
                           const std::string& data = "");
    
    /// @brief Perform HTTP request with full URL
    std::string performRequest(const std::string& url, const std::string& method = "GET",
                              const std::string& data = "");
    
    /// @brief Get CoinGecko ID for a trading symbol
    std::string getCoinGeckoId(const std::string& symbol);
    
    /// @brief Make request to CoinGecko API
    std::string makeCoinGeckoRequest(const std::string& url);
    
    /// @brief Get fallback price when primary source fails
    double getFallbackPrice(const std::string& symbol);

public:
    /// @brief Constructor to initialize Alpaca client
    AlpacaClient();
    
    /// @brief Get account information
    Json::Value getAccount();
    
    /// @brief Get all current positions
    Json::Value getPositions();
    
    /// @brief Get specific position by symbol
    Json::Value getPosition(const std::string& symbol);
    
    /// @brief Get orders with optional status filter
    Json::Value getOrders(const std::string& status = "all");
    
    /// @brief Get specific order by ID
    Json::Value getOrder(const std::string& order_id);
    
    /// @brief Place a trading order with optional stop loss and take profit
    bool placeOrder(const std::string& symbol, const std::string& side,
                   double quantity, const std::string& type = "market",
                   double stop_loss = 0.0, double take_profit = 0.0);
    
    /// @brief Cancel an existing order
    bool cancelOrder(const std::string& order_id);
    
    /// @brief Close a specific position
    bool closePosition(const std::string& symbol);
    
    /// @brief Close all open positions
    bool closeAllPositions();
    
    /// @brief Get current market price for a symbol
    double getCurrentPrice(const std::string& symbol);
    
    /// @brief Calculate market stress based on regime data
    double getMarketStress(const RegimeData& regime);
    
    /// @brief Get batch prices for multiple symbols
    std::map<std::string, double> getBatchPrices(const std::vector<std::string>& symbols);
};