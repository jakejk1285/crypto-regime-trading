/**
 * @file alpaca_client_impl.cpp
 * @brief Alpaca Markets API client implementation
 * 
 * @author Jake Kostoryz
 * @date 2025
 * @version 4.0
 */

#include "../include/alpaca_client.h"

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

namespace {
    /**
     * @brief CURL write callback function for HTTP responses
     * @param contents Response data buffer
     * @param size Size of each data element
     * @param nmemb Number of data elements
     * @param userp User data pointer (std::string*)
     * @return Number of bytes processed
     */
    size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
        static_cast<std::string*>(userp)->append(static_cast<char*>(contents), size * nmemb);
        return size * nmemb;
    }
}

// ============================================================================
// ALPACA CLIENT IMPLEMENTATION
// ============================================================================

AlpacaClient::AlpacaClient() {
    loadConfig();
    m_base_url = m_paper_trading ? "https://paper-api.alpaca.markets" : "https://api.alpaca.markets";
}

void AlpacaClient::loadConfig() {
    std::ifstream env_file(".env");
    if (!env_file.is_open()) {
        safeOutput("❌ .env file not found!");
        return;
    }

    std::string line;
    while (std::getline(env_file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        const size_t delimiter_pos = line.find('=');
        if (delimiter_pos == std::string::npos) {
            continue;
        }

        std::string key = line.substr(0, delimiter_pos);
        std::string value = line.substr(delimiter_pos + 1);

        // Remove surrounding quotes if present
        if (value.size() >= 2 && value.front() == '\'' && value.back() == '\'') {
            value = value.substr(1, value.length() - 2);
        }

        // Map configuration values
        if (key == "ALPACA_API_KEY") {
            m_api_key = std::move(value);
        } else if (key == "ALPACA_SECRET_KEY") {
            m_secret_key = std::move(value);
        } else if (key == "ALPACA_PAPER") {
            m_paper_trading = (value == "true");
        } else if (key == "COINGECKO_DEMO_KEY") {
            m_coingecko_key = std::move(value);
        }
    }
}

std::string AlpacaClient::makeRequest(const std::string& endpoint, const std::string& method, const std::string& data) {
    std::string response;
    CURL* curl = curl_easy_init();
    
    if (!curl) {
        return response;
    }

    // Configure request URL and basic options
    const std::string url = m_base_url + endpoint;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    // Set up authentication and content headers
    struct curl_slist* headers = nullptr;
    const std::string auth_header = "APCA-API-KEY-ID: " + m_api_key;
    const std::string secret_header = "APCA-API-SECRET-KEY: " + m_secret_key;

    headers = curl_slist_append(headers, auth_header.c_str());
    headers = curl_slist_append(headers, secret_header.c_str());
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    // Configure method-specific options
    if (method == "POST") {
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
    } else if (method == "DELETE") {
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");
    }

    // Execute request and cleanup
    const CURLcode result = curl_easy_perform(curl);
    static_cast<void>(result); // Suppress unused variable warning
    
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    return response;
}

std::string AlpacaClient::getCoinGeckoId(const std::string& symbol) {
    static const std::map<std::string, std::string> SYMBOL_TO_COINGECKO_MAP = {
        {"BTC/USD", "bitcoin"},       {"BTCUSD", "bitcoin"},
        {"ETH/USD", "ethereum"},      {"ETHUSD", "ethereum"},
        {"ADA/USD", "cardano"},       {"ADAUSD", "cardano"},
        {"SOL/USD", "solana"},        {"SOLUSD", "solana"},
        {"LINK/USD", "chainlink"},    {"LINKUSD", "chainlink"},
        {"DOT/USD", "polkadot"},      {"DOTUSD", "polkadot"},
        {"AVAX/USD", "avalanche-2"},  {"AVAXUSD", "avalanche-2"},
        {"MATIC/USD", "matic-network"}, {"MATICUSD", "matic-network"},
        {"ATOM/USD", "cosmos"},       {"ATOMUSD", "cosmos"},
        {"ALGO/USD", "algorand"},     {"ALGOUSD", "algorand"}
    };
    
    const auto it = SYMBOL_TO_COINGECKO_MAP.find(symbol);
    return (it != SYMBOL_TO_COINGECKO_MAP.end()) ? it->second : "";
}

std::string AlpacaClient::makeCoinGeckoRequest(const std::string& url) {
    std::string response;
    CURL* curl = curl_easy_init();
    
    if (!curl) {
        return response;
    }

    // Configure request options
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

    // Set up headers
    struct curl_slist* headers = nullptr;
    if (!m_coingecko_key.empty()) {
        const std::string api_key_header = "x-cg-demo-api-key: " + m_coingecko_key;
        headers = curl_slist_append(headers, api_key_header.c_str());
    }
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "User-Agent: CryptoTradingBot/4.0");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    // Execute request and check response
    const CURLcode result = curl_easy_perform(curl);
    long response_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
    
    if (result != CURLE_OK || response_code != 200) {
        safeOutput("⚠️  CoinGecko API request failed. CURL error: " + 
                  std::string(curl_easy_strerror(result)) + 
                  ", HTTP code: " + std::to_string(response_code));
        response.clear();
    }

    // Cleanup
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    return response;
}

double AlpacaClient::getCurrentPrice(const std::string& symbol) {
    const std::string coin_id = getCoinGeckoId(symbol);
    
    if (!coin_id.empty()) {
        try {
            const std::string url = "https://api.coingecko.com/api/v3/simple/price?ids=" + 
                                  coin_id + "&vs_currencies=usd";
            const std::string response = makeCoinGeckoRequest(url);

            if (!response.empty()) {
                Json::Value root;
                Json::CharReaderBuilder builder;
                std::string errors;
                std::istringstream iss(response);

                if (Json::parseFromStream(builder, iss, &root, &errors)) {
                    if (root.isMember(coin_id) && root[coin_id].isMember("usd")) {
                        const double price = root[coin_id]["usd"].asDouble();
                        
                        if (price > 0.0) {
                            safeOutput("💰 " + symbol + " (CoinGecko): $" + std::to_string(price));
                            return price;
                        }
                    }
                }
            }
        } catch (const std::exception&) {
            // Fall through to fallback pricing
        }
    }

    return getFallbackPrice(symbol);
}

double AlpacaClient::getFallbackPrice(const std::string& symbol) {
    safeOutput("⚠️  Using market-based estimate for " + symbol + " (APIs unavailable)");
    
    static const std::map<std::string, double> FALLBACK_PRICES = {
        {"BTC/USD", 95000.0},   {"BTCUSD", 95000.0},
        {"ETH/USD", 3400.0},    {"ETHUSD", 3400.0},
        {"SOL/USD", 220.0},     {"SOLUSD", 220.0},
        {"AVAX/USD", 42.0},     {"AVAXUSD", 42.0},
        {"LINK/USD", 24.0},     {"LINKUSD", 24.0},
        {"ADA/USD", 0.90},      {"ADAUSD", 0.90},
        {"DOT/USD", 7.50},      {"DOTUSD", 7.50},
        {"MATIC/USD", 0.55},    {"MATICUSD", 0.55},
        {"ATOM/USD", 6.80},     {"ATOMUSD", 6.80},
        {"ALGO/USD", 0.35},     {"ALGOUSD", 0.35}
    };
    
    const auto it = FALLBACK_PRICES.find(symbol);
    return (it != FALLBACK_PRICES.end()) ? it->second : 100.0;
}

double AlpacaClient::getMarketStress(const RegimeData& regime) {
    double stress_level = regime.market_stress_level;

    // Adjust stress based on RSI conditions
    if (regime.bitcoin_rsi_14 < 30.0) {
        stress_level += 0.2;  // Oversold conditions increase stress
    } else if (regime.bitcoin_rsi_14 > 70.0) {
        stress_level += 0.15; // Overbought conditions add moderate stress
    }

    return std::clamp(stress_level, 0.0, 1.0);
}

Json::Value AlpacaClient::getPositions() {
    std::string response = makeRequest("/v2/positions");
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;
    std::istringstream iss(response);
    Json::parseFromStream(builder, iss, &root, &errors);
    return root;
}

Json::Value AlpacaClient::getAccount() {
    std::string response = makeRequest("/v2/account");
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;
    std::istringstream iss(response);
    Json::parseFromStream(builder, iss, &root, &errors);
    return root;
}

bool AlpacaClient::placeOrder(const std::string& symbol, const std::string& side, double qty,
               const std::string& type, double stop_loss, double take_profit) {
    Json::Value order;
    order["symbol"] = symbol;
    order["side"] = side;
    order["type"] = type;
    order["time_in_force"] = "gtc";

    // Apply minimum quantity requirements for different cryptocurrencies
    if (symbol.find("/USD") != std::string::npos) {
        if (symbol == "BTC/USD") {
            order["qty"] = std::to_string(std::max(qty, 0.0001));
        } else if (symbol == "ETH/USD") {
            order["qty"] = std::to_string(std::max(qty, 0.001));
        } else {
            order["qty"] = std::to_string(std::max(qty, 0.01));
        }
    } else {
        order["qty"] = std::to_string(qty);
    }

    // Place the main order first, then add stop loss/take profit as separate orders
    Json::StreamWriterBuilder builder;
    std::string order_data = Json::writeString(builder, order);

    std::string response = makeRequest("/v2/orders", "POST", order_data);

    Json::Value result;
    Json::CharReaderBuilder reader_builder;
    std::string errors;
    std::istringstream iss(response);
    if (Json::parseFromStream(reader_builder, iss, &result, &errors)) {
        bool success = !result.isMember("message") || result["message"].asString().find("error") == std::string::npos;

        // Place additional risk management orders if main order succeeded
        if (success && (stop_loss > 0.0 || take_profit > 0.0)) {
            std::this_thread::sleep_for(std::chrono::seconds(1));

            if (stop_loss > 0.0) {
                Json::Value stop_order;
                stop_order["symbol"] = symbol;
                stop_order["side"] = (side == "buy") ? "sell" : "buy";
                stop_order["type"] = "stop";
                stop_order["time_in_force"] = "gtc";
                stop_order["stop_price"] = std::to_string(stop_loss);
                stop_order["qty"] = order["qty"];

                std::string stop_data = Json::writeString(builder, stop_order);
                makeRequest("/v2/orders", "POST", stop_data);
            }
        }

        return success;
    }
    return false;
}

bool AlpacaClient::closePosition(const std::string& symbol) {
    std::string endpoint = "/v2/positions/" + symbol;
    std::string response = makeRequest(endpoint, "DELETE");
    return !response.empty() && response.find("error") == std::string::npos;
}

Json::Value AlpacaClient::getOrders(const std::string& status) {
    std::string endpoint = "/v2/orders";
    if (status != "all") {
        endpoint += "?status=" + status;
    }
    std::string response = makeRequest(endpoint);
    
    Json::CharReaderBuilder builder;
    Json::CharReader* reader = builder.newCharReader();
    Json::Value result;
    std::string errors;
    std::istringstream iss(response);
    Json::parseFromStream(builder, iss, &result, &errors);
    delete reader;
    return result;
}

std::map<std::string, double> AlpacaClient::getBatchPrices(const std::vector<std::string>& symbols) {
    std::map<std::string, double> prices;
    
    // Fetch prices individually using CoinGecko API
    for (const std::string& symbol : symbols) {
        double price = getCurrentPrice(symbol);
        if (price > 0) {
            prices[symbol] = price;
        }
    }
    
    return prices;
}