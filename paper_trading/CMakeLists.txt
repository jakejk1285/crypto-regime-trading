cmake_minimum_required(VERSION 3.16)
project(crypto_trading_strategy)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# Find CURL
find_package(CURL REQUIRED)
if(NOT CURL_FOUND)
    message(FATAL_ERROR "CURL not found. Install with: brew install curl (macOS) or apt-get install libcurl4-openssl-dev (Ubuntu)")
endif()

# Find JsonCpp - Enhanced macOS handling
pkg_check_modules(JSONCPP jsoncpp)
if(NOT JSONCPP_FOUND)
    # Try manual find for macOS
    find_path(JSONCPP_INCLUDE_DIRS json/json.h
            PATHS
            /opt/homebrew/include
            /usr/local/include
            /opt/local/include
    )
    find_library(JSONCPP_LIBRARIES
            NAMES jsoncpp
            PATHS
            /opt/homebrew/lib
            /usr/local/lib
            /opt/local/lib
    )

    if(JSONCPP_INCLUDE_DIRS AND JSONCPP_LIBRARIES)
        set(JSONCPP_FOUND TRUE)
        message(STATUS "JsonCpp found manually: ${JSONCPP_LIBRARIES}")
    else()
        message(FATAL_ERROR "JsonCpp not found. Install with: brew install jsoncpp")
    endif()
endif()

# macOS specific jsoncpp linking fix
if(APPLE AND JSONCPP_FOUND)
    # Check if we need to use the full library path
    if(NOT JSONCPP_LIBRARIES MATCHES "^/")
        # If it's just "jsoncpp", find the full path
        find_library(JSONCPP_FULL_PATH
                NAMES jsoncpp
                PATHS
                /opt/homebrew/lib
                /usr/local/lib
                /opt/local/lib
                REQUIRED
        )
        set(JSONCPP_LIBRARIES ${JSONCPP_FULL_PATH})
        message(STATUS "Using full jsoncpp library path: ${JSONCPP_LIBRARIES}")
    endif()
endif()

# Verify we have all required libraries
if(NOT CURL_FOUND)
    message(FATAL_ERROR "CURL library is required. Install with: brew install curl")
endif()

if(NOT JSONCPP_FOUND)
    message(FATAL_ERROR "JsonCpp library is required. Install with: brew install jsoncpp")
endif()

# Add the executable with the new modular source files
add_executable(crypto_paper_trading
        src/main.cpp
        src/common.cpp
        src/performance_tracker.cpp
        src/regime_data.cpp
        src/alpaca_client_impl.cpp
        src/trading_strategy.cpp
        src/trading_strategy_ui.cpp
)

# Set include directories
target_include_directories(crypto_paper_trading PRIVATE
        ${CURL_INCLUDE_DIRS}
        ${JSONCPP_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include  # For all header files
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link libraries
target_link_libraries(crypto_paper_trading
        ${CURL_LIBRARIES}
        ${JSONCPP_LIBRARIES}
        Threads::Threads
)

# macOS specific linking
if(APPLE)
    target_link_libraries(crypto_paper_trading
            "-framework CoreFoundation"
            "-framework Security"
    )
endif()

# Compiler options
target_compile_options(crypto_paper_trading PRIVATE
        -Wall -Wextra -O2
)

# Debug configuration
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(crypto_paper_trading PRIVATE DEBUG=1)
    target_compile_options(crypto_paper_trading PRIVATE -g)
endif()

# Copy .env file to build directory if it exists
if(EXISTS "${CMAKE_SOURCE_DIR}/.env")
    add_custom_command(TARGET crypto_paper_trading POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/.env
            $<TARGET_FILE_DIR:crypto_paper_trading>/.env
            COMMENT "Copying .env file to build directory"
    )
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "=== Crypto Trading Strategy Build (Multi-file) ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "System: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Source files:")
message(STATUS "  - src/main.cpp (Application entry point)")
message(STATUS "  - src/common.cpp (Global utilities)")
message(STATUS "  - src/performance_tracker.cpp (Performance tracking)")
message(STATUS "  - src/regime_data.cpp (Regime analysis)")
message(STATUS "  - src/alpaca_client_impl.cpp (API client)")
message(STATUS "  - src/trading_strategy.cpp (Core trading logic)")
message(STATUS "  - src/trading_strategy_ui.cpp (UI and main loop)")
message(STATUS "  - include/ (Modular headers)")
message(STATUS "CURL found: ${CURL_FOUND}")
message(STATUS "CURL libraries: ${CURL_LIBRARIES}")
message(STATUS "JSONCPP found: ${JSONCPP_FOUND}")
message(STATUS "JSONCPP libraries: ${JSONCPP_LIBRARIES}")
message(STATUS "JSONCPP include dirs: ${JSONCPP_INCLUDE_DIRS}")
message(STATUS "Threads found: ${Threads_FOUND}")
message(STATUS "============================================================")
message(STATUS "")