/**
 * @file regime_data.h
 * @brief Market regime data structures and management
 * 
 * @author Jake Kostoryz
 * @date 2025
 * @version 4.0
 */

#pragma once

#include "common.h"

// ============================================================================
// REGIME DATA STRUCTURES
// ============================================================================

/**
 * @brief Research-based market regime data structure
 * 
 * Contains comprehensive market regime information including PCA factors,
 * stability metrics, trading parameters, and transition probabilities.
 * This data drives the regime-switching trading strategy.
 */
struct RegimeData {
    // Core regime identification
    int regime_id;                ///< Unique identifier for the current regime
    std::string regime_name;      ///< Descriptive name of the regime
    double confidence;            ///< Confidence level in regime classification (0-1)
    std::string timestamp;        ///< When this regime data was generated
    bool should_trade;            ///< Whether trading is recommended in this regime

    // PCA-Based Regime Characteristics
    double pc1_market_factor;     ///< First principal component (market factor)
    double pc2_volatility_factor; ///< Second principal component (volatility factor)
    double pc3_factor;            ///< Third principal component
    double pc4_factor;            ///< Fourth principal component
    double pc5_factor;            ///< Fifth principal component
    double pc1_std;               ///< Standard deviation of PC1
    double pc2_std;               ///< Standard deviation of PC2

    // Regime Stability Metrics
    double persistence;           ///< How stable this regime is (0-1)
    double avg_duration;          ///< Average duration of this regime in days
    double frequency_percentage;  ///< How often this regime occurs (%)
    int episodes;                 ///< Number of historical episodes

    // Trading Parameters
    double risk_multiplier;       ///< Risk adjustment multiplier for this regime
    double position_scale;        ///< Position sizing scale factor
    double stop_loss_multiplier;  ///< Stop loss distance multiplier
    double take_profit_multiplier; ///< Take profit distance multiplier
    double max_position_percent;  ///< Maximum position size as % of portfolio

    // Market Context Indicators
    double market_stress_level;   ///< Overall market stress indicator (0-100)
    double crypto_vix;            ///< Crypto volatility index
    double volatility_adjustment; ///< Volatility-based position adjustment
    double market_momentum_14d;   ///< 14-day market momentum indicator
    double bitcoin_rsi_14;        ///< Bitcoin 14-day RSI
    double ethereum_price_sma14_ratio; ///< ETH price to 14-day SMA ratio

    // Transition Intelligence
    std::map<int, double> transition_probabilities; ///< Probabilities of transitioning to other regimes
    double regime_switching_frequency; ///< How frequently regimes switch

    // Coin-specific scores
    std::map<std::string, double> coin_scores; ///< Individual coin attractiveness scores
    std::vector<std::string> active_signals;   ///< Currently active trading signals

    // Analysis Metadata
    std::string strategy;         ///< Strategy name used for this analysis
    std::string analysis_timestamp; ///< When this analysis was performed
    std::string data_source;      ///< Source of the market data
    double data_freshness_hours;  ///< How fresh the data is in hours
    int components_used;          ///< Number of PCA components used
    int features_used;            ///< Number of features in the analysis
    double variance_explained;    ///< Percentage of variance explained by PCA
    double silhouette_score;      ///< Quality of regime clustering (0-1)

    /// @brief Constructor to initialize regime data with defaults
    RegimeData();
};

// ============================================================================
// REGIME DATA MANAGEMENT
// ============================================================================

/**
 * @brief Reader class for regime data files
 * 
 * Handles loading and parsing of regime data from external files,
 * tracking regime changes, and providing access to current regime information.
 */
class RegimeDataReader {
private:
    std::string m_regime_file_path;  ///< Path to the regime data file
    RegimeData m_current_regime;     ///< Currently loaded regime data
    int m_previous_regime_id;        ///< ID of the previous regime for change detection
    bool m_regime_changed;           ///< Flag indicating if regime has changed
    std::chrono::time_point<std::chrono::system_clock> m_last_update; ///< Last update timestamp

    /// @brief Parse regime data from file content
    bool parseRegimeFile(const std::string& file_content);

public:
    /// @brief Constructor with file path
    explicit RegimeDataReader(const std::string& file_path);
    
    /// @brief Load regime data from file
    bool loadRegimeData();
    
    /// @brief Check if regime has changed since last check
    bool hasRegimeChanged() const { return m_regime_changed; }
    
    /// @brief Clear the regime change flag
    void clearRegimeChangeFlag() { m_regime_changed = false; }
    
    /// @brief Get reference to current regime data
    const RegimeData& getCurrentRegime() const { return m_current_regime; }
    
    /// @brief Check if data should be refreshed
    bool shouldRefresh() const;
};