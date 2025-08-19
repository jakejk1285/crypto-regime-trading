#!/bin/bash

# ===============================================
# Crypto Trading Strategy Build Script
# ===============================================

echo "🚀 Building Crypto Trading Strategy (C++)..."
echo "🏗️  Modular architecture with regime-based trading"
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a Homebrew package is installed (macOS)
brew_package_installed() {
    brew list "$1" >/dev/null 2>&1
}

# Check system
echo "🔍 Checking system requirements..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "✅ macOS detected"
    SYSTEM="macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "✅ Linux detected"  
    SYSTEM="Linux"
else
    echo "❌ Unsupported system: $OSTYPE"
    exit 1
fi

# Check for CMake
if ! command_exists cmake; then
    echo "❌ CMake not found"
    if [ "$SYSTEM" = "macOS" ]; then
        echo "💡 Install with: brew install cmake"
    else
        echo "💡 Install with: sudo apt-get install cmake"
    fi
    exit 1
else
    CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
    echo "✅ CMake found: $CMAKE_VERSION"
fi

# Check for C++ compiler
if ! command_exists g++ && ! command_exists clang++; then
    echo "❌ C++ compiler not found"
    if [ "$SYSTEM" = "macOS" ]; then
        echo "💡 Install with: xcode-select --install"
    else
        echo "💡 Install with: sudo apt-get install build-essential"
    fi
    exit 1
else
    if command_exists clang++; then
        echo "✅ Clang++ found"
    elif command_exists g++; then
        echo "✅ G++ found"
    fi
fi

# Check for required libraries
echo ""
echo "📚 Checking required C++ libraries..."

# Check for libcurl
if [ "$SYSTEM" = "macOS" ]; then
    if ! brew_package_installed curl; then
        echo "⚠️  libcurl not found. Installing..."
        brew install curl
        if [ $? -ne 0 ]; then
            echo "❌ Failed to install libcurl"
            exit 1
        fi
    fi
    echo "✅ libcurl found"
else
    if ! dpkg -l | grep -q libcurl4-openssl-dev; then
        echo "⚠️  libcurl4-openssl-dev not found"
        echo "💡 Install with: sudo apt-get install libcurl4-openssl-dev"
        exit 1
    fi
    echo "✅ libcurl4-openssl-dev found"
fi

# Check for jsoncpp
if [ "$SYSTEM" = "macOS" ]; then
    if ! brew_package_installed jsoncpp; then
        echo "⚠️  jsoncpp not found. Installing..."
        brew install jsoncpp
        if [ $? -ne 0 ]; then
            echo "❌ Failed to install jsoncpp"
            exit 1
        fi
    fi
    echo "✅ jsoncpp found"
else
    if ! dpkg -l | grep -q libjsoncpp-dev; then
        echo "⚠️  libjsoncpp-dev not found"  
        echo "💡 Install with: sudo apt-get install libjsoncpp-dev"
        exit 1
    fi
    echo "✅ libjsoncpp-dev found"
fi

# Check for required source files
echo ""
echo "📄 Checking source files..."
REQUIRED_FILES=("CMakeLists.txt" "src/main.cpp" "src/trading_strategy.cpp" "include/trading_strategy.h")
MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
    echo "❌ Missing required files: ${MISSING_FILES[*]}"
    exit 1
else
    echo "✅ All required source files found"
fi

echo ""
echo "✅ All build prerequisites satisfied"

# Clean previous build if requested
if [ "$1" = "clean" ]; then
    echo "🧹 Cleaning previous build..."
    rm -rf build/
fi

# Create and enter build directory
mkdir -p build
cd build

# Configure with CMake  
echo "⚙️  Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo "❌ CMake configuration failed"
    exit 1
fi

# Build the project
echo ""
echo "🔨 Building project..."
CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
make -j$CORES

if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

# Check if executable was created
if [ -f "crypto_paper_trading" ]; then
    echo "✅ Build successful!"
    echo "🎯 Executable created: build/crypto_paper_trading"
    echo ""

    # Make executable
    chmod +x crypto_paper_trading

    # Check if .env file exists
    if [ -f "../.env" ]; then
        echo "✅ Found .env file"
    else
        echo "⚠️  .env file not found"
        echo "📝 Creating sample .env file..."
        cat > ../.env << 'EOF'
# Crypto Trading Strategy Configuration
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_PAPER=true
EOF
        echo "✅ Sample .env file created"
    fi

    echo ""
    echo "🎉 BUILD COMPLETE!"
    echo "=================="
    echo ""
    echo "📋 Next steps:"
    echo "1. Edit .env file with your Alpaca API keys"
    echo "2. Run: cd build && ./crypto_paper_trading"
    echo ""
    echo "🎯 Features:"
    echo "   ✅ Regime-based trading strategy"
    echo "   ✅ Dynamic position sizing"
    echo "   ✅ Real-time risk management"
    echo "   ✅ Paper trading integration"
    echo "   ✅ Performance tracking"
    echo ""
    
else
    echo "❌ Build completed but executable not found"
    ls -la
    exit 1
fi

echo "🎉 Crypto Trading Strategy is ready!"
echo "▶️  Start trading: cd build && ./crypto_paper_trading"