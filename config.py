"""
Configuration for Macro Quadrant Trading Strategy
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === HYPERLIQUID CONFIGURATION ===
PRIVATE_KEY = os.getenv("PRIVATE_KEY", "0x04793e9c32fb5def7a646610fd7a4bbb2c769b3b110b8049ef926e8815082d30")
VAULT_ADDRESS = os.getenv("VAULT_ADDRESS", "0x8bc7d5bf1afe96613e6ff67ade9fcf7d9223165e")

# Multi-Vault Configuration
VAULT_CONFIGS = [
    {
        'name': 'Vault 1',
        'api_key': os.getenv("HL_API_KEY_1", "your_api_key_1"),
        'secret_key': os.getenv("HL_SECRET_KEY_1", "your_secret_key_1"),
        'vault_address': os.getenv("VAULT_ADDRESS_1", "0x1234567890123456789012345678901234567890"),
        'weight': 1.0
    },
    {
        'name': 'Vault 2',
        'api_key': os.getenv("HL_API_KEY_2", "your_api_key_2"),
        'secret_key': os.getenv("HL_SECRET_KEY_2", "your_secret_key_2"),
        'vault_address': os.getenv("VAULT_ADDRESS_2", "0x2345678901234567890123456789012345678901"),
        'weight': 1.0
    },
    {
        'name': 'Vault 3',
        'api_key': os.getenv("HL_API_KEY_3", "your_api_key_3"),
        'secret_key': os.getenv("HL_SECRET_KEY_3", "your_secret_key_3"),
        'vault_address': os.getenv("VAULT_ADDRESS_3", "0x3456789012345678901234567890123456789012"),
        'weight': 1.0
    }
]

# === STRATEGY CONFIGURATION ===
# Position sizing rules
POSITION_RULES = {
    'Q1': 2.00,  # 200% leverage in Goldilocks regime
    'Q2': 0.00,  # Flat in Reflation regime
    'Q3': 1.00,  # 100% leverage in Stagflation regime
    'Q4': 0.00   # Flat in Deflation regime
}

# Risk management
MAX_POSITIONS = 5  # Maximum number of concurrent positions
POSITION_SIZE_PCT = 0.20  # 20% of account value per position
MAX_DRAWDOWN_PCT = 0.15  # 15% maximum drawdown before stopping

# Technical indicators
MOMENTUM_LOOKBACK_DAYS = 20  # 20-day momentum for quadrant calculation
EMA_PERIOD = 50  # 50-day EMA filter
MIN_EMA_FILTER = True  # Whether to use EMA filter

# === ASSET CONFIGURATION ===
# Core assets for quadrant analysis
CORE_ASSETS = {
    # Q1 Assets (Growth ↑, Inflation ↓)
    'QQQ': 'NASDAQ 100 (Growth)',
    'VUG': 'Vanguard Growth ETF', 
    'IWM': 'Russell 2000 (Small Caps)',
    'BTC-USD': 'Bitcoin (BTC)',
    
    # Q2 Assets (Growth ↑, Inflation ↑)
    'XLE': 'Energy Sector ETF',
    'DBC': 'Broad Commodities ETF',
    
    # Q3 Assets (Growth ↓, Inflation ↑)
    'GLD': 'Gold ETF',
    'LIT': 'Lithium & Battery Tech ETF',
    
    # Q4 Assets (Growth ↓, Inflation ↓)
    'TLT': '20+ Year Treasury Bonds',
    'XLU': 'Utilities Sector ETF',
    'VIXY': 'Short-Term VIX Futures ETF',
}

# Trading assets on Hyperliquid
TRADING_ASSETS = {
    'BTC': 'Bitcoin',
    'ETH': 'Ethereum', 
    'SOL': 'Solana',
    'BNB': 'Binance Coin',
    'XRP': 'Ripple',
    'ADA': 'Cardano',
    'AVAX': 'Avalanche',
    'DOGE': 'Dogecoin',
    'DOT': 'Polkadot',
    'LINK': 'Chainlink',
    'MATIC': 'Polygon',
    'SHIB': 'Shiba Inu',
    'UNI': 'Uniswap',
    'LTC': 'Litecoin',
    'ATOM': 'Cosmos',
    'XLM': 'Stellar',
    'ALGO': 'Algorand',
    'NEAR': 'NEAR Protocol',
    'AAVE': 'Aave',
    'TRX': 'TRON'
}

# Trading Configuration
TRADING_UNIVERSE = ['BTC', 'ETH']  # Only BTC and ETH
MAX_POSITION_SIZE = 0.5  # Maximum 50% of capital per position
MIN_POSITION_SIZE = 0.1  # Minimum 10% of capital per position

# Risk Management
MAX_LEVERAGE = 3.0  # Maximum leverage allowed

# Position Sizing by Regime
REGIME_POSITION_SIZES = {
    'Q1': 2.0,  # 200% of capital per position (Goldilocks)
    'Q2': 0.0,  # 0% of capital per position (Reflation - flat)
    'Q3': 1.0,  # 100% of capital per position (Stagflation)
    'Q4': 0.0   # 0% of capital per position (Deflation - flat)
}

# Leverage by Regime
REGIME_LEVERAGE = {
    'Q1': 2.0,  # 2x leverage (Goldilocks)
    'Q2': 1.0,  # 1x leverage (Reflation - no leverage)
    'Q3': 1.0,  # 1x leverage (Stagflation)
    'Q4': 1.0   # 1x leverage (Deflation - no leverage)
}

# === NOTIFICATION CONFIGURATION ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7206335521:AAGQeuhik1SrN_qMakb9bxkI1iAJmg8A3Wo")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "7119645510")

# === QUADRANT DESCRIPTIONS ===
QUADRANT_DESCRIPTIONS = {
    'Q1': 'Growth ↑, Inflation ↓ (Goldilocks)',
    'Q2': 'Growth ↑, Inflation ↑ (Reflation)', 
    'Q3': 'Growth ↓, Inflation ↑ (Stagflation)',
    'Q4': 'Growth ↓, Inflation ↓ (Deflation)'
}

# === EXECUTION SETTINGS ===
EXECUTION_INTERVAL_MINUTES = 60  # Check for signals every hour
RETRY_ATTEMPTS = 3  # Number of retry attempts for failed orders
ORDER_TIMEOUT_SECONDS = 30  # Timeout for order execution

# === LOGGING ===
LOG_LEVEL = "INFO"
LOG_FILE = "macro_quadrant_strategy.log"

# === QUADRANT ALLOCATIONS (FOR DASHBOARD) ===
# Portfolio Allocations per Quadrant - ORIGINAL PROVEN UNIVERSE
QUAD_ALLOCATIONS = {
    'Q1': {
        'QQQ': 0.60 * 0.40,      # 40% of 60% Growth
        'BTC-USD': 0.60 * 0.40,     # 40% of 60% Growth (Bitcoin)
        'ARKK': 0.60 * 0.30,     # 30% of 60% Growth
        'IWM': 0.60 * 0.15,      # 15% of 60% Growth (Small Caps)
        'XLC': 0.15 * 0.50,      # 50% of 15% Consumer Disc
        'XLY': 0.15 * 0.50,      # 50% of 15% Consumer Disc
        'TLT': 0.10 * 0.50,      # 50% of 10% Bonds
        'LQD': 0.10 * 0.50,      # 50% of 10% Bonds
    },
    'Q2': {
        'XLE': 0.35 * 0.20,      # 20% of 35% Commodities
        'DBC': 0.35 * 0.20,      # 20% of 35% Commodities
        'GCC': 0.35 * 0.20,      # 20% of 35% Commodities
        'LIT': 0.35 * 0.10,      # 10% of 35% Commodities (Lithium)
        'XLF': 0.30 * 0.333,     # 33% of 30% Cyclicals
        'XLI': 0.30 * 0.333,     # 33% of 30% Cyclicals
        'XLB': 0.30 * 0.334,     # 34% of 30% Cyclicals
        'USO': 0.30 * 0.334,     # 34% of 30% Cyclicals
        'XOP': 0.15 * 0.333,     # 33% of 15% Energy
        'FCG': 0.15 * 0.333,     # 33% of 15% Energy
        'USO': 0.15 * 0.334,     # 34% of 15% Energy (Crude Oil)
        'VNQ': 0.10 * 0.50,      # 50% of 10% Real Assets
        'PAVE': 0.10 * 0.50,     # 50% of 10% Real Assets
        'VTV': 0.10 * 0.50,      # 50% of 10% Value
        'IWD': 0.10 * 0.50,      # 50% of 10% Value
        'EEM': 0.25 * 0.333,     # 33% of 25% Energy
    },
    'Q3': {
        'FCG': 0.25 * 0.333,     # 33% of 25% Energy
        'EEM': 0.25 * 0.333,     # 33% of 25% Energy
        'XLE': 0.25 * 0.333,     # 33% of 25% Energy
        'XOP': 0.25 * 0.334,     # 34% of 25% Energy
        'GLD': 0.30 * 0.12,      # 12% of 30% Commodities
        'DBC': 0.30 * 0.12,      # 12% of 30% Commodities
        'DBA': 0.30 * 0.12,      # 12% of 30% Commodities
        'REMX': 0.30 * 0.12,     # 12% of 30% Commodities
        'URA': 0.30 * 0.12,      # 12% of 30% Commodities (Uranium)
        'LIT': 0.30 * 0.10,      # 10% of 30% Commodities (Lithium)
        'TIP': 0.20 * 0.50,      # 50% of 20% TIPS
        'VTIP': 0.20 * 0.50,     # 50% of 20% TIPS
        'VNQ': 0.10 * 0.50,      # 50% of 10% Real Assets
        'PAVE': 0.10 * 0.50,     # 50% of 10% Real Assets
        'XLV': 0.15 * 0.333,     # 33% of 15% Equities
        'XLU': 0.15 * 0.333,     # 33% of 15% Equities
    },
    'Q4': {
        'VGLT': 0.50 * 0.50,     # 50% of 50% Long Duration
        'IEF': 0.50 * 0.50,      # 50% of 50% Long Duration
        'LQD': 0.20 * 0.50,      # 50% of 20% IG Credit
        'MUB': 0.20 * 0.50,      # 50% of 20% IG Credit
        'XLU': 0.15 * 0.25,     # 25% of 15% Defensive
        'XLP': 0.15 * 0.25,     # 25% of 15% Defensive
        'XLV': 0.15 * 0.25,     # 25% of 15% Defensive
        # Cash allocation (15%) represented as staying in cash - no ticker
    }
}

# Quadrant indicator assets (for scoring)
QUAD_INDICATORS = {
    'Q1': ['QQQ', 'VUG', 'IWM', 'BTC-USD'],
    'Q2': ['XLE', 'DBC', 'GCC', 'LIT'],
    'Q3': ['GLD', 'DBC', 'DBA', 'REMX', 'URA', 'LIT'],
    'Q4': ['TLT', 'XLU', 'VIXY']
} 

# === CRYPTO PROXY CONFIGURATION ===
# If your broker cannot trade spot crypto tickers like 'BTC-USD', the strategy can
# keep BTC-USD in the model but execute that sleeve using tradeable proxy equities.
#
# Behavior:
# - Any BTC-USD weight produced by the strategy is reallocated into these proxies
# - Proxies must pass the same EMA filter (above 50d EMA) to be eligible
# - Within the crypto sleeve, weights are set by volatility-chasing (higher vol = higher weight)
# - The sleeve is capped to at most BTC_PROXY_MAX_POSITIONS tickers (default 10)
BTC_PROXY_BASKET = {
    # BTC Treasury Companies
    'MSTR': 0.10,
    'STRIVE': 0.05,

    # ETH Treasury Companies
    'SBET': 0.05,

    # Pure Play Miners
    'MARA': 0.03,
    'CLSK': 0.02,
    'RIOT': 0.03,
    'BITF': 0.02,
    'CIFR': 0.02,

    # Diversified Miners
    'IREN': 0.03,
    'CORZ': 0.03,
    'HUT': 0.02,
    'BTDR': 0.02,
    'BTBT': 0.02,
    'WULF': 0.02,
    'HIVE': 0.02,

    # Altcoin Digital Asset Treasuries
    'ASST': 0.015,
    'UPXI': 0.015,
    'AVX': 0.01,
    'DFDV': 0.01,

    # Infrastructure/Exchanges
    'COIN': 0.03,
    'GLXY': 0.02,
    'CRCL': 0.02,
    'HOOD': 0.02,
    'BLSH': 0.015,
    'CAN': 0.015,
    'SQ': 0.02,
}

BTC_PROXY_MAX_POSITIONS = 10