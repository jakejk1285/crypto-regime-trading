/**
 * @file regime_data.cpp
 * @brief Regime data structure and reader implementation
 * 
 * @author Jake Kostoryz
 * @date 2025
 * @version 4.0
 */

#include "../include/regime_data.h"

// ============================================================================
// REGIME DATA IMPLEMENTATION
// ============================================================================

RegimeData::RegimeData() 
    : regime_id(-1), confidence(0.0),
      should_trade(false),
      pc1_market_factor(0.0), pc2_volatility_factor(0.0),
      pc3_factor(0.0), pc4_factor(0.0), pc5_factor(0.0),
      pc1_std(0.0), pc2_std(0.0),
      persistence(0.0), avg_duration(9.4), frequency_percentage(0.0), episodes(0),
      risk_multiplier(1.0), position_scale(1.0),
      stop_loss_multiplier(0.02), take_profit_multiplier(0.06),
      max_position_percent(0.20), market_stress_level(0.0),
      crypto_vix(0.0), volatility_adjustment(1.0),
      market_momentum_14d(0.0), bitcoin_rsi_14(0.0), ethereum_price_sma14_ratio(0.0),
      regime_switching_frequency(11.2), strategy("WAIT_AND_SEE"),
      data_freshness_hours(0.0), components_used(25), features_used(117),
      variance_explained(0.8968), silhouette_score(0.0) 
{
}

// ============================================================================
// REGIME DATA READER IMPLEMENTATION
// ============================================================================

RegimeDataReader::RegimeDataReader(const std::string& file_path)
    : m_regime_file_path(file_path), m_regime_changed(false)
{
    m_last_update = std::chrono::system_clock::now() - std::chrono::hours(25);
}

bool RegimeDataReader::loadRegimeData() {
    try {
        std::ifstream file(m_regime_file_path);
        if (!file.is_open()) {
            std::cerr << "❌ Failed to open regime file: " << m_regime_file_path << std::endl;
            return false;
        }

        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        file.close();

        if (parseRegimeFile(content)) {
            m_last_update = std::chrono::system_clock::now();
            return true;
        }
    } catch (const std::exception& e) {
        std::cerr << "❌ Error loading regime data: " << e.what() << std::endl;
    }
    return false;
}

bool RegimeDataReader::parseRegimeFile(const std::string& file_content) {
    try {
        Json::CharReaderBuilder builder;
        Json::CharReader* reader = builder.newCharReader();
        Json::Value root;
        std::string errors;

        bool parsingSuccessful = reader->parse(file_content.c_str(),
                                              file_content.c_str() + file_content.size(),
                                              &root, &errors);
        delete reader;

        if (!parsingSuccessful) {
            std::cerr << "❌ Failed to parse JSON: " << errors << std::endl;
            return false;
        }

        RegimeData new_regime;

        // Extract core regime identification data
        new_regime.regime_id = root.get("regime_id", -1).asInt();
        new_regime.regime_name = root.get("regime_name", "UNKNOWN").asString();
        new_regime.confidence = root.get("confidence", 0.0).asDouble();
        new_regime.timestamp = root.get("timestamp", "").asString();
        new_regime.should_trade = root.get("should_trade", false).asBool();

        // Extract PCA component values
        new_regime.pc1_market_factor = root.get("pc1_market_factor", 0.0).asDouble();
        new_regime.pc2_volatility_factor = root.get("pc2_volatility_factor", 0.0).asDouble();
        new_regime.pc3_factor = root.get("pc3_factor", 0.0).asDouble();
        new_regime.pc4_factor = root.get("pc4_factor", 0.0).asDouble();
        new_regime.pc5_factor = root.get("pc5_factor", 0.0).asDouble();
        new_regime.pc1_std = root.get("pc1_std", 0.0).asDouble();
        new_regime.pc2_std = root.get("pc2_std", 0.0).asDouble();

        // Extract regime stability metrics
        new_regime.persistence = root.get("persistence", 0.0).asDouble();
        new_regime.avg_duration = root.get("avg_duration", 0.0).asDouble();
        new_regime.frequency_percentage = root.get("frequency_percentage", 0.0).asDouble();
        new_regime.episodes = root.get("episodes", 0).asInt();

        // Extract trading configuration parameters
        new_regime.risk_multiplier = root.get("risk_multiplier", 1.0).asDouble();
        new_regime.position_scale = root.get("position_scale", 1.0).asDouble();
        new_regime.stop_loss_multiplier = root.get("stop_loss_multiplier", 0.02).asDouble();
        new_regime.take_profit_multiplier = root.get("take_profit_multiplier", 0.06).asDouble();
        new_regime.max_position_percent = root.get("max_position_percent", 0.20).asDouble();

        // Extract market context indicators
        new_regime.market_stress_level = root.get("market_stress_level", 0.0).asDouble();
        new_regime.volatility_adjustment = root.get("volatility_adjustment", 1.0).asDouble();
        new_regime.strategy = root.get("strategy", "WAIT_AND_SEE").asString();

        // Extract individual coin attractiveness scores
        const Json::Value& scores = root["coin_scores"];
        if (!scores.isNull() && scores.isObject()) {
            for (Json::Value::const_iterator it = scores.begin(); it != scores.end(); ++it) {
                new_regime.coin_scores[it.key().asString()] = it->asDouble();
            }
        }

        // Detect if regime has changed since last update
        if (new_regime.regime_id != m_previous_regime_id) {
            m_regime_changed = true;
            m_previous_regime_id = new_regime.regime_id;
        }

        m_current_regime = new_regime;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "❌ Error parsing regime file: " << e.what() << std::endl;
        return false;
    }
}

bool RegimeDataReader::shouldRefresh() const {
    auto now = std::chrono::system_clock::now();
    auto time_since_update = std::chrono::duration_cast<std::chrono::minutes>(now - m_last_update);
    return time_since_update.count() > 30;
}