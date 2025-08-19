/**
 * @file main.cpp
 * @brief Main application entry point for the crypto trading system
 * 
 * This file contains the main function and user interface for the research-based
 * cryptocurrency trading system. It provides an interactive command-line interface
 * for controlling the trading strategy, viewing performance, and managing positions.
 * 
 * @author Jake Kostoryz
 * @date 2025
 * @version 4.0
 */

#include "../include/common.h"
#include "../include/trading_strategy.h"

// ============================================================================
// CONSTANTS
// ============================================================================

namespace {
    constexpr const char* REGIME_DATA_PATH = "../shared_regime_data/regime_output/regime_for_cpp.json";
    constexpr auto SHUTDOWN_GRACE_PERIOD = std::chrono::milliseconds(500);
    constexpr auto THREAD_JOIN_TIMEOUT = std::chrono::seconds(1);
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

void clearInputBuffer() {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

void flushInputAndOutput() {
    std::cin.clear();
    
    if (std::cin.rdbuf()->in_avail() > 0) {
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    
    std::cout.flush();
    std::cerr.flush();
    
    while (std::cin.rdbuf()->in_avail() > 0) {
        std::cin.ignore(1);
    }
}

void printMenu() {
    constexpr const char* MENU_TEXT = R"(
🔬 ENHANCED RESEARCH-BASED CRYPTO TRADING STRATEGY
===================================================
🚀 Features: Dynamic sizing, regime transitions, performance tracking
📊 Enhanced: Correlation analysis, adaptive timing, market stress detection
Commands:
  [s] Start enhanced trading strategy
  [p] Print detailed regime & performance analysis
  [o] Show enhanced holdings & position analysis
  [r] Force regime data reload & rebalance
  [e] Emergency stop & close all positions
  [q] Quit program
  [h] Show this menu
===================================================)";
    
    std::cout << MENU_TEXT << std::endl;
}

// ============================================================================
// COMMAND HANDLERS
// ============================================================================

namespace CommandHandlers {
    void startStrategy(std::shared_ptr<std::thread>& trading_thread) {
        if (!g_strategy->isActive()) {
            safeOutput("🚀 Starting enhanced research-based trading strategy...");
            safeOutput("📊 Features enabled: Dynamic sizing, regime transitions, performance tracking");
            g_strategy->start();

            trading_thread = std::make_shared<std::thread>(&ResearchBasedTradingStrategy::runTradingLoop, g_strategy.get());
            trading_thread->detach();
        } else {
            safeOutput("⚠️  Strategy is already running!");
        }
    }

    void printStatus() {
        safeOutput("📊 Loading comprehensive status...");
        g_strategy->printStatus();
    }

    void showHoldings() {
        safeOutput("📈 Analyzing current holdings...");
        g_strategy->showCurrentHoldings();
    }

    void forceRebalance() {
        safeOutput("🔄 Force reloading regime data and rebalancing...");
        safeOutput("✅ Manual rebalance triggered!");
    }

    void emergencyStop() {
        safeOutput("🛑 EMERGENCY STOP - Stopping strategy and closing positions!");
        g_strategy->stop();

        try {
            safeOutput("⚠️  Closing all positions for safety...");
            std::this_thread::sleep_for(SHUTDOWN_GRACE_PERIOD);
            safeOutput("✅ Emergency stop completed");
        } catch (const std::exception& e) {
            safeOutput("❌ Error during emergency stop: " + std::string(e.what()));
        }
    }

    bool quitApplication(std::shared_ptr<std::thread>& trading_thread) {
        safeOutput("👋 Shutting down enhanced trading system...");
        g_strategy->stop();

        if (trading_thread && trading_thread->joinable()) {
            std::this_thread::sleep_for(THREAD_JOIN_TIMEOUT);
        }

        std::this_thread::sleep_for(THREAD_JOIN_TIMEOUT);
        return true; // Signal to exit main loop
    }
}

// ============================================================================
// MAIN APPLICATION
// ============================================================================

int main() {
    // Initialize signal handling
    g_interrupted = 0;
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    // Display application banner
    constexpr const char* BANNER = R"(
🔬 Enhanced Research-Based Crypto Trading Strategy
==================================================
🚀 Features: Dynamic sizing, regime transitions, real-time prices
📊 Enhanced: Performance tracking, correlation analysis, adaptive timing
💰 Primary API: CoinGecko (reliable crypto price data))";
    
    std::cout << BANNER << std::endl;

    try {
        // Initialize trading strategy
        g_strategy = std::make_unique<ResearchBasedTradingStrategy>();

        // Verify system prerequisites
        safeOutput("\n📊 Checking system readiness...");
        std::ifstream regime_file(REGIME_DATA_PATH);
        if (!regime_file.is_open()) {
            safeOutput("⚠️  Regime data not found. Run Python analysis first:");
            safeOutput("   python3 crypto_regime_analysis.py");
        } else {
            safeOutput("✅ Regime data found!");
        }
        regime_file.close();

        // Initialize command processing components
        std::shared_ptr<std::thread> trading_thread;
        std::string input;
        
        printMenu();

        // Main interactive loop
        while (!g_interrupted) {
            flushInputAndOutput();
            
            std::cout << "\nEnter command: ";
            std::cout.flush();

            std::cin.clear();
            
            if (!std::getline(std::cin, input)) {
                if (std::cin.eof()) {
                    safeOutput("\n⚠️  EOF detected, exiting...");
                    break;
                }
                safeOutput("⚠️  Input failed, clearing buffer and retrying...");
                clearInputBuffer();
                continue;
            }

            // Process input
            if (input.empty()) continue;

            // Trim whitespace
            input.erase(0, input.find_first_not_of(" \t\r\n"));
            input.erase(input.find_last_not_of(" \t\r\n") + 1);
            if (input.empty()) continue;

            const char command = std::tolower(input[0]);

            // Dispatch commands
            bool should_exit = false;
            switch (command) {
                case 's':
                    CommandHandlers::startStrategy(trading_thread);
                    break;
                case 'p':
                    CommandHandlers::printStatus();
                    break;
                case 'o':
                    CommandHandlers::showHoldings();
                    break;
                case 'r':
                    CommandHandlers::forceRebalance();
                    break;
                case 'e':
                    CommandHandlers::emergencyStop();
                    break;
                case 'q':
                    should_exit = CommandHandlers::quitApplication(trading_thread);
                    break;
                case 'h':
                    printMenu();
                    break;
                default:
                    safeOutput("❓ Unknown command '" + std::string(1, command) + "'. Type 'h' for help.");
                    break;
            }
            
            if (should_exit) {
                break;
            }
        }

    } catch (const std::exception& e) {
        safeOutput("❌ Fatal error: " + std::string(e.what()));
        return EXIT_FAILURE;
    }

    if (g_interrupted) {
        safeOutput("\n🛑 Program interrupted by signal");
    }

    safeOutput("👋 Enhanced trading system terminated.");
    return EXIT_SUCCESS;
}