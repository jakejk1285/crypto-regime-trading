/**
 * @file common.h
 * @brief Common includes and type definitions for the trading system
 * 
 * @author Jake Kostoryz
 * @date 2025
 * @version 4.0
 */

#pragma once

// Standard Library Headers
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// Third-party Library Headers
#include <curl/curl.h>
#include <json/json.h>

// ============================================================================
// GLOBAL VARIABLES AND UTILITY FUNCTIONS
// ============================================================================

// Forward declaration
class ResearchBasedTradingStrategy;

/// @brief Global strategy instance for signal handling
extern std::unique_ptr<ResearchBasedTradingStrategy> g_strategy;

/// @brief Interrupt flag for graceful shutdown
extern volatile sig_atomic_t g_interrupted;

/// @brief Mutex for thread-safe console operations
extern std::mutex console_mutex;

/// @brief Thread-safe console output
void safeOutput(const std::string& message);

/// @brief Signal handler for graceful shutdown
void signalHandler(int signal);