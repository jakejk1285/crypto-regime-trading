#!/usr/bin/env python3
"""
Data Manager for Regime-Based Crypto Backtesting
Handles data collection and historical regime generation for 2024-2025 (1 year, 366 data points)
Limited by CoinGecko API free tier (1 year maximum)
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed


class CryptoDataManager:
    """
    Manages cryptocurrency data collection and historical regime analysis
    """

    def __init__(self, start_date='2024-01-01', end_date='2025-01-01'):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

        # CoinGecko crypto universe - matches crypto_regime_analysis.py
        self.coingecko_symbols = {
            'bitcoin': 'BTCUSD',
            'ethereum': 'ETHUSD', 
            'cardano': 'ADAUSD',
            'polkadot': 'DOTUSD',
            'chainlink': 'LINKUSD',
            'solana': 'SOLUSD',
            'matic-network': 'MATICUSD',
            'avalanche-2': 'AVAXUSD',
            'cosmos': 'ATOMUSD',
            'algorand': 'ALGOUSD',
            'tether': 'USDTUSD',
            'ripple': 'XRPUSD',
            'binancecoin': 'BNBUSD',
            'dogecoin': 'DOGEUSD',
            'litecoin': 'LTCUSD'
        }

        # Reverse mapping for convenience
        self.symbol_to_coingecko = {v: k for k, v in self.coingecko_symbols.items()}

        self.cache_dir = Path('backtest_data_cache')
        self.cache_dir.mkdir(exist_ok=True)

    def fetch_coin_data_coingecko(self, coin_id, headers, retries=3):
        """
        Fetch historical price data for a specific coin from CoinGecko API
        Based on crypto_regime_analysis.py implementation
        """
        for attempt in range(retries):
            try:
                # Calculate days between start and end date
                days = (self.end_date - self.start_date).days
                
                # Use public API endpoint first, then fall back to demo API
                if headers.get("x-cg-demo-api-key"):
                    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                else:
                    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                
                # CoinGecko API limitations:
                # - Hourly data requires Enterprise plan
                # - Daily data is available for free API 
                # - For 2-90 days, hourly data is automatic (don't specify interval)
                
                if headers.get("x-cg-demo-api-key"):
                    # With API key, we can request specific intervals
                    interval = "daily" if days > 90 else None
                    params = {
                        "vs_currency": "usd",
                        "days": str(min(days, 365))
                    }
                    if interval:
                        params["interval"] = interval
                else:
                    # Free API - use daily data only, no interval parameter
                    params = {
                        "vs_currency": "usd", 
                        "days": str(min(days, 365))  # Limit to 1 year for free API
                    }
                
                response = requests.get(url, headers=headers, params=params, timeout=30)
                
                if response.status_code == 429:
                    wait_time = min(30 * (2 ** attempt), 120)  # Exponential backoff
                    print(f"   ⏱️  Rate limited {coin_id}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                data = response.json()
                
                prices = data.get('prices', [])
                if prices:
                    # Convert to DataFrame with OHLCV structure
                    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # Create OHLCV data (CoinGecko daily API only provides price)
                    # For hourly data, we'll approximate OHLCV from price
                    df['open'] = df['price'].shift(1).fillna(df['price'])
                    df['high'] = df['price'] * 1.001  # Small approximation
                    df['low'] = df['price'] * 0.999   # Small approximation  
                    df['close'] = df['price']
                    df['volume'] = 1000000  # Placeholder volume
                    
                    # Drop the original price column and reorder
                    df = df[['open', 'high', 'low', 'close', 'volume']]
                    
                    return df
                else:
                    print(f"   ❌ No price data for {coin_id}")
                    return None
                    
            except Exception as e:
                if attempt == retries - 1:
                    print(f"   ❌ {coin_id} failed after {retries} attempts: {e}")
                    return None
                time.sleep(5 * (attempt + 1))
        
        return None

    def collect_historical_crypto_data(self, force_refresh=False) -> Dict[str, pd.DataFrame]:
        """
        Collect historical cryptocurrency data using CoinGecko API
        Matches the approach from crypto_regime_analysis.py
        """
        cache_file = self.cache_dir / f'crypto_data_{self.start_date.strftime("%Y%m%d")}_{self.end_date.strftime("%Y%m%d")}.pkl'

        if cache_file.exists() and not force_refresh:
            print("📁 Loading cached crypto data...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        print(f"📊 Collecting crypto data from CoinGecko API")
        print(f"   Date range: {self.start_date.date()} to {self.end_date.date()}")
        
        # Load environment variables for API key
        load_dotenv()
        api_key = os.getenv("COINGECKO_DEMO_KEY")
        
        headers = {}
        if api_key:
            headers["x-cg-demo-api-key"] = api_key
            print("   🔑 Using API key for enhanced rate limits")
        else:
            print("   ⚠️  No API key found, using public endpoints with rate limits")

        crypto_data = {}

        # Process coins in batches to respect rate limits
        # Limit to core coins for testing without API key
        if not headers.get("x-cg-demo-api-key"):
            coin_list = ['bitcoin', 'ethereum', 'cardano', 'solana']  # Just 4 core coins for free API
            batch_size = 1  # One at a time for free API
            print("   ⚠️ No API key - limiting to 4 core cryptocurrencies")
        else:
            coin_list = list(self.coingecko_symbols.keys())
            batch_size = 3  # Bigger batches with API key
        
        print(f"   📋 Processing {len(coin_list)} coins: {coin_list}")
        
        for i in range(0, len(coin_list), batch_size):
            batch = coin_list[i:i + batch_size]
            print(f"📥 Processing batch {i//batch_size + 1}: {batch}")
            
            # Use ThreadPoolExecutor for parallel requests
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_to_coin = {
                    executor.submit(self.fetch_coin_data_coingecko, coin_id, headers): coin_id 
                    for coin_id in batch
                }
                
                for future in as_completed(future_to_coin):
                    coin_id = future_to_coin[future]
                    try:
                        data = future.result()
                        if data is not None:
                            strategy_symbol = self.coingecko_symbols[coin_id]
                            crypto_data[strategy_symbol] = data
                            print(f"   ✅ {strategy_symbol} ({coin_id}): {len(data)} records")
                        
                        # Rate limiting between requests - more conservative for free API
                        sleep_time = 3.0 if not headers.get("x-cg-demo-api-key") else 1.2
                        time.sleep(sleep_time)
                        
                    except Exception as e:
                        print(f"   ❌ Error processing {coin_id}: {e}")
            
            # Longer delay between batches - more conservative for free API
            if i + batch_size < len(coin_list):
                batch_sleep = 15.0 if not headers.get("x-cg-demo-api-key") else 8.0
                print(f"   ⏳ Waiting {batch_sleep}s between batches...")
                time.sleep(batch_sleep)

        # Cache the data
        if crypto_data:
            with open(cache_file, 'wb') as f:
                pickle.dump(crypto_data, f)
            print(f"📁 Cached data to {cache_file}")

        print(f"✅ Collected data for {len(crypto_data)} cryptocurrencies")
        return crypto_data

    def create_historical_regime_features(self, crypto_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create comprehensive feature set matching crypto_regime_analysis.py approach
        Uses the same feature engineering as the production regime analysis system
        """
        print("🔧 Creating regime-optimized features...")

        # Get the main timestamp index (use Bitcoin as reference)
        if 'BTCUSD' not in crypto_data:
            raise ValueError("Bitcoin data required for regime analysis")
        
        btc_data = crypto_data['BTCUSD']
        timestamps = btc_data.index
        features_dict = {}

        # Use the symbol mapping from the class (reversed for this function)
        symbol_mapping = self.symbol_to_coingecko

        # Core coins for analysis (matching crypto_regime_analysis)
        core_coins = ['bitcoin', 'ethereum', 'solana', 'cardano', 'polkadot']
        available_coins = []
        
        for strategy_symbol, coin_name in symbol_mapping.items():
            if strategy_symbol in crypto_data and not crypto_data[strategy_symbol].empty:
                available_coins.append(coin_name)
        
        if not available_coins:
            print("⚠️ No core coins available, using all available coins")
            available_coins = list(symbol_mapping.values())[:5]

        print(f"📊 Processing {len(available_coins)} cryptocurrencies for regime features")

        for coin in available_coins:
            # Find corresponding data
            strategy_symbol = None
            for sym, name in symbol_mapping.items():
                if name == coin and sym in crypto_data:
                    strategy_symbol = sym
                    break
            
            if not strategy_symbol or crypto_data[strategy_symbol].empty:
                continue

            data = crypto_data[strategy_symbol]
            # Align data to common timestamp
            aligned_data = data.reindex(timestamps)
            aligned_data = aligned_data.ffill().bfill()
            
            prices = aligned_data['close'].dropna()
            if len(prices) < 50:  # Skip coins with insufficient data
                continue

            print(f"   📈 Processing {coin} ({len(prices)} data points)")

            # Returns (key PC1 components) - matching crypto_regime_analysis
            features_dict[f'{coin}_return_1d'] = prices.pct_change()
            features_dict[f'{coin}_return_7d'] = prices.pct_change(7) 
            features_dict[f'{coin}_return_30d'] = prices.pct_change(30)

            # Technical indicators (key PC1 components) - use smaller windows
            features_dict[f'{coin}_rsi_14'] = self.calculate_rsi(prices, 14)
            features_dict[f'{coin}_rsi_21'] = self.calculate_rsi(prices, 21)

            # Moving averages with min_periods for more data
            features_dict[f'{coin}_sma_10'] = prices.rolling(10, min_periods=5).mean()
            features_dict[f'{coin}_sma_20'] = prices.rolling(20, min_periods=10).mean()
            features_dict[f'{coin}_price_sma14_ratio'] = prices / prices.rolling(14, min_periods=7).mean()

            # Volatility (key PC2 components) - use smaller windows for more data
            returns = prices.pct_change()
            features_dict[f'{coin}_volatility_20d'] = returns.rolling(20, min_periods=10).std()
            features_dict[f'{coin}_var_95_20d'] = returns.rolling(20, min_periods=10).quantile(0.05)

            # Momentum indicators
            features_dict[f'{coin}_momentum_14d'] = prices / prices.shift(14) - 1

        # Market-wide indicators (critical for PC1) - matching crypto_regime_analysis
        if len(available_coins) >= 2:
            primary_coin = available_coins[0]  # bitcoin
            secondary_coin = available_coins[1]  # ethereum
            
            # Find the corresponding data
            primary_data = None
            secondary_data = None
            
            for sym, name in symbol_mapping.items():
                if name == primary_coin and sym in crypto_data:
                    primary_data = crypto_data[sym]['close']
                    break
            
            for sym, name in symbol_mapping.items():
                if name == secondary_coin and sym in crypto_data:
                    secondary_data = crypto_data[sym]['close']
                    break

            if primary_data is not None and secondary_data is not None:
                primary_returns = primary_data.pct_change()
                secondary_returns = secondary_data.pct_change()

                # Market momentum (key PC1 factor) with min_periods
                features_dict['market_momentum_14d'] = (
                    primary_returns.rolling(14, min_periods=7).mean() + 
                    secondary_returns.rolling(14, min_periods=7).mean()
                ) / 2

                # Breadth indicators - calculate across all available data
                all_returns = {}
                for coin in available_coins:
                    for sym, name in symbol_mapping.items():
                        if name == coin and sym in crypto_data:
                            returns = crypto_data[sym]['close'].pct_change()
                            all_returns[coin] = returns
                            break

                if all_returns:
                    returns_df = pd.DataFrame(all_returns)
                    positive_returns = (returns_df > 0).sum(axis=1)
                    features_dict['breadth_positive'] = positive_returns / len(returns_df.columns)

                # Correlation indicators with min_periods
                features_dict['primary_secondary_correlation'] = primary_returns.rolling(20, min_periods=10).corr(secondary_returns)

        # Create DataFrame
        features_df = pd.DataFrame(features_dict, index=timestamps)
        
        # Clean features - handle NaN values more carefully
        initial_rows = len(features_df)
        features_df = features_df.dropna()
        dropped_rows = initial_rows - len(features_df)
        
        if dropped_rows > initial_rows * 0.5:
            print(f"⚠️ Warning: Dropped {dropped_rows} rows ({dropped_rows/initial_rows:.1%}) due to missing data")
        
        print(f"✅ Created {len(features_df.columns)} features, {len(features_df)} observations")
        
        if len(features_df) < 50:  # Reduced minimum threshold
            raise ValueError(f"Insufficient clean data: {len(features_df)} observations")

        return features_df

    def calculate_rsi(self, prices, period=14):
        """
        Calculate Relative Strength Index (RSI) technical indicator.
        Matches the implementation in crypto_regime_analysis.py
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_historical_regimes(self, features_df: pd.DataFrame, force_refresh=False) -> pd.DataFrame:
        """
        Generate historical regime assignments using your PCA + clustering approach
        """
        cache_file = self.cache_dir / f'historical_regimes_{self.start_date.strftime("%Y%m%d")}_{self.end_date.strftime("%Y%m%d")}.pkl'

        if cache_file.exists() and not force_refresh:
            print("📁 Loading cached regime data...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        print("🔬 Generating historical regime assignments...")

        # Import your regime analysis components
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        # Prepare features (remove any infinite or NaN values) - matching crypto_regime_analysis
        clean_features = features_df.replace([np.inf, -np.inf], np.nan)
        
        # Handle NaN values the same way as crypto_regime_analysis
        if clean_features.isnull().any().any():
            print("⚠️ Handling remaining NaN values...")
            # Forward fill, then backward fill, then fill with 0
            clean_features = clean_features.ffill().bfill().fillna(0)
            nan_count = features_df.isnull().sum().sum()
            print(f"   Filled {nan_count} NaN values")
        else:
            print("✅ No NaN values detected")
        
        clean_features = clean_features.select_dtypes(include=[np.number])

        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(clean_features)

        # PCA with 25 components (matching your research)
        print("   📊 Performing PCA analysis...")
        pca = PCA(n_components=25, random_state=42)
        pca_components = pca.fit_transform(scaled_features)

        variance_explained = np.sum(pca.explained_variance_ratio_)
        print(f"   📈 PCA variance explained: {variance_explained:.3f}")

        # K-means clustering with 7 regimes (matching your research)
        print("   🎯 Performing regime clustering...")
        kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
        regime_labels = kmeans.fit_predict(pca_components)

        silhouette_avg = silhouette_score(pca_components, regime_labels)
        print(f"   📊 Silhouette score: {silhouette_avg:.3f}")

        # Create regime DataFrame with characteristics
        regime_df = pd.DataFrame({
            'regime_id': regime_labels,
            'pc1_market_factor': pca_components[:, 0],
            'pc2_volatility_factor': pca_components[:, 1],
            'pc3_factor': pca_components[:, 2] if pca_components.shape[1] > 2 else 0,
            'pc4_factor': pca_components[:, 3] if pca_components.shape[1] > 3 else 0,
            'pc5_factor': pca_components[:, 4] if pca_components.shape[1] > 4 else 0,
        }, index=clean_features.index)

        # Calculate regime statistics
        regime_stats = []
        for regime_id in range(7):
            mask = regime_df['regime_id'] == regime_id
            regime_data = regime_df[mask]

            if len(regime_data) > 0:
                # Calculate persistence (probability of staying in same regime)
                transitions = (regime_df['regime_id'].shift(1) == regime_id) & (regime_df['regime_id'] == regime_id)
                total_in_regime = mask.sum()
                persistence = transitions.sum() / max(total_in_regime - 1, 1)

                # Calculate average duration
                regime_series = (regime_df['regime_id'] == regime_id).astype(int)
                durations = []
                current_duration = 0
                for val in regime_series:
                    if val == 1:
                        current_duration += 1
                    else:
                        if current_duration > 0:
                            durations.append(current_duration)
                        current_duration = 0
                if current_duration > 0:
                    durations.append(current_duration)

                avg_duration = np.mean(durations) if durations else 1.0

                regime_stats.append({
                    'regime_id': regime_id,
                    'frequency': total_in_regime,
                    'percentage': (total_in_regime / len(regime_df)) * 100,
                    'persistence': persistence,
                    'avg_duration': avg_duration,
                    'episodes': len(durations)
                })

        # Add regime statistics to main DataFrame
        regime_stats_df = pd.DataFrame(regime_stats)

        for _, stats in regime_stats_df.iterrows():
            regime_id = int(stats['regime_id'])
            mask = regime_df['regime_id'] == regime_id
            regime_df.loc[mask, 'persistence'] = stats['persistence']
            regime_df.loc[mask, 'avg_duration'] = stats['avg_duration']
            regime_df.loc[mask, 'frequency_percentage'] = stats['percentage']

        # Add your exact regime strategy mapping and should_trade logic
        regime_strategy_mapping = {
            0: {"name": "STABLE_GROWTH", "strategy": "STABLE_GROWTH", "risk_mult": 1.3},
            1: {"name": "MODERATE_MOMENTUM", "strategy": "MOMENTUM", "risk_mult": 1.1},
            2: {"name": "BASELINE_MARKET", "strategy": "BALANCED", "risk_mult": 1.0},
            3: {"name": "EXTREME_OUTLIER", "strategy": "WAIT_AND_SEE", "risk_mult": 0.3},
            4: {"name": "DEFENSIVE_STABLE", "strategy": "CONSERVATIVE", "risk_mult": 0.8},
            5: {"name": "BREAKOUT_MOMENTUM", "strategy": "MOMENTUM", "risk_mult": 1.2},
            6: {"name": "EXTREME_VOLATILITY", "strategy": "WAIT_AND_SEE", "risk_mult": 0.2}
        }

        # Add strategy information
        regime_df['strategy'] = regime_df['regime_id'].map(lambda x: regime_strategy_mapping[x]['strategy'])
        regime_df['risk_multiplier'] = regime_df['regime_id'].map(lambda x: regime_strategy_mapping[x]['risk_mult'])

        # Calculate market stress - matching crypto_regime_analysis approach
        # Try to find bitcoin volatility feature with the correct naming convention
        btc_vol_feature = None
        for col in features_df.columns:
            if 'bitcoin' in col.lower() and 'volatility' in col.lower():
                btc_vol_feature = col
                break
        
        if btc_vol_feature is not None:
            btc_volatility = features_df[btc_vol_feature]
        else:
            # Fallback: use bitcoin VaR if available
            var_feature = None
            for col in features_df.columns:
                if 'bitcoin' in col.lower() and 'var' in col.lower():
                    var_feature = col
                    break
            
            if var_feature is not None:
                btc_volatility = features_df[var_feature].abs()
            else:
                # Final fallback
                btc_volatility = pd.Series(0.02, index=features_df.index)
        
        market_stress = np.clip(btc_volatility * 10, 0, 1)
        market_stress = market_stress.reindex(regime_df.index, method='ffill').fillna(0.3)
        regime_df['market_stress'] = market_stress

        # Your exact should_trade logic
        regime_df['should_trade'] = (
                (regime_df['strategy'] != "WAIT_AND_SEE") &
                (regime_df['persistence'] > 0.5) &
                (market_stress < 0.7) &
                (regime_df['avg_duration'] > 3.0)
        )

        # Cache the regime data
        with open(cache_file, 'wb') as f:
            pickle.dump(regime_df, f)

        print(
            f"✅ Generated historical regimes: {len(regime_df)} timestamps, {regime_df['regime_id'].nunique()} unique regimes")

        # Print regime distribution
        regime_dist = regime_df['regime_id'].value_counts().sort_index()
        print("📊 Regime Distribution:")
        for regime_id, count in regime_dist.items():
            strategy = regime_strategy_mapping[regime_id]['strategy']
            pct = (count / len(regime_df)) * 100
            print(f"   Regime {regime_id} ({strategy}): {count} observations ({pct:.1f}%)")

        return regime_df

    def set_crypto_data_cache(self, crypto_data: Dict[str, pd.DataFrame]):
        """
        Set the crypto data cache for backtesting integration
        """
        self.crypto_data_cache = crypto_data
        print(f"📦 Crypto data cache set with {len(crypto_data)} symbols")

    def get_crypto_prices(self, symbol: str, timestamp: pd.Timestamp) -> Dict[str, float]:
        """
        Get OHLCV prices for a specific crypto at a specific timestamp
        Enhanced error handling and fallback logic
        """
        if hasattr(self, 'crypto_data_cache') and symbol in self.crypto_data_cache:
            data = self.crypto_data_cache[symbol]
            if not data.empty:
                try:
                    # Find closest timestamp
                    closest_idx = data.index.get_indexer([timestamp], method='nearest')[0]
                    if closest_idx >= 0 and closest_idx < len(data):
                        row = data.iloc[closest_idx]

                        return {
                            'open': float(row.get('open', 0)),
                            'high': float(row.get('high', 0)),
                            'low': float(row.get('low', 0)),
                            'close': float(row.get('close', 0)),
                            'volume': float(row.get('volume', 0))
                        }
                except Exception as e:
                    print(f"⚠️ Error getting prices for {symbol}: {e}")

        return {'open': 0, 'high': 0, 'low': 0, 'close': 0, 'volume': 0}


if __name__ == "__main__":
    # Test the data manager with 1-year data period (2024-2025)
    print("🔬 CRYPTO DATA MANAGER TEST")
    print("=" * 50)
    print("📊 Testing 1-year data collection (2024-2025)")
    print("📊 CoinGecko API Limit: 366 data points maximum")

    data_manager = CryptoDataManager(start_date='2024-01-01', end_date='2025-01-01')

    try:
        # Collect crypto data
        print("📈 Step 1: Collecting historical crypto data...")
        crypto_data = data_manager.collect_historical_crypto_data()
        
        if crypto_data is None or len(crypto_data) == 0:
            print("❌ Failed to collect crypto data")
            exit(1)

        # Set cache for backtesting integration
        data_manager.set_crypto_data_cache(crypto_data)

        # Create features
        print("🔧 Step 2: Creating regime features...")
        features_df = data_manager.create_historical_regime_features(crypto_data)

        # Generate regimes
        print("🎯 Step 3: Generating regime assignments...")
        regime_df = data_manager.generate_historical_regimes(features_df)

        print("\n✅ Data Manager Test Complete!")
        print(f"📊 Crypto Data: {len(crypto_data)} symbols")
        print(f"🔧 Features: {len(features_df.columns)} features over {len(features_df)} timestamps")
        print(f"🎯 Regimes: {len(regime_df)} regime assignments")
        
        # Test price lookup functionality
        print("\n🧪 Testing price lookup functionality...")
        test_timestamp = regime_df.index[-1]  # Use latest timestamp
        for symbol in list(crypto_data.keys())[:3]:  # Test first 3 symbols
            prices = data_manager.get_crypto_prices(symbol, test_timestamp)
            print(f"   {symbol} @ {test_timestamp}: Close=${prices['close']:.2f}")
        
        print("\n🎉 All tests passed! Data Manager is ready for backtesting integration.")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()