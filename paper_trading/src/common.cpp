/**
 * @file common.cpp
 * @brief Common utility functions implementation
 * 
 * @author Jake Kostoryz
 * @date 2025
 * @version 4.0
 */

#include "../include/common.h"
#include "../include/trading_strategy.h"

// ============================================================================
// GLOBAL VARIABLES
// ============================================================================

std::unique_ptr<ResearchBasedTradingStrategy> g_strategy;
volatile sig_atomic_t g_interrupted;
std::mutex console_mutex;

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

void safeOutput(const std::string& message) {
    std::lock_guard<std::mutex> lock(console_mutex);
    std::cout << message << std::endl;
}

void signalHandler(int) {
    g_interrupted = 1;
    if (g_strategy) {
        g_strategy->stop();
    }
    safeOutput("\n🛑 Graceful shutdown initiated...");
}