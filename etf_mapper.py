"""
ETF-to-Constituents Mapper
==========================

Maps ETFs to their underlying stock holdings.
Non-equity ETFs (commodities, bonds, crypto) are skipped.

Usage:
    from etf_mapper import get_stock_universe, get_constituent_tickers_for_universe

    etf_weights = {'XLE': 0.15, 'XLF': 0.12, 'GLD': 0.08, ...}
    stocks = get_stock_universe(etf_weights)
    # Returns: {'XOM', 'CVX', 'JPM', 'BRK.B', ...}  (GLD skipped)

    # For data fetch: all constituent tickers for equity ETFs in a set
    tickers = get_constituent_tickers_for_universe(etf_ticker_set)
"""

# ETFs that don't hold stocks (commodities, bonds, crypto, volatility)
NON_EQUITY_ETFS = {
    'BTC-USD', 'GLD', 'DBC', 'GCC', 'DBA', 'USO',
    'TLT', 'LQD', 'IEF', 'VGLT', 'MUB', 'TIP', 'VTIP',
    'VIXY',
}

# Constituent mappings — equity ETFs only
ETF_CONSTITUENTS = {
    'XLE': [
        'XOM', 'CVX', 'COP', 'WMB', 'EOG', 'MPC', 'PSX', 'SLB', 'VLO', 'KMI',
        'BKR', 'OKE', 'EQT', 'TRGP', 'OXY', 'FANG', 'EXE', 'DVN', 'HAL', 'CTRA',
        'TPL', 'APA',
    ],
    'LIT': [
        'RIO', 'ALB', 'SQM', 'TSLA', 'ENS', 'LAC', 'SGML',
    ],
    'XLF': [
        'BRK.B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'SPGI', 'AXP', 'PGR',
        'MS', 'FI', 'C', 'BLK', 'MMC', 'CB', 'ICE', 'SCHW', 'CME', 'AON',
        'MCO', 'COF', 'USB', 'AJG', 'PNC', 'TFC', 'AIG', 'TRV', 'AFL', 'MET',
        'BK', 'ALL', 'ACGL', 'DFS', 'FIS', 'NDAQ', 'WTW', 'FITB', 'STT', 'RF',
        'HBAN', 'NTRS', 'CINF', 'CFG', 'KEY', 'SYF', 'BRO', 'RJF', 'MKTX', 'ERIE',
    ],
    'XLI': [
        'GE', 'CAT', 'RTX', 'UNP', 'HON', 'ETN', 'ADP', 'BA', 'DE', 'LMT',
        'UBER', 'TT', 'WM', 'GD', 'ITW', 'NOC', 'EMR', 'CSX', 'PH', 'FDX',
        'CTAS', 'CARR', 'NSC', 'GEV', 'PCAR', 'AXON', 'FAST', 'AME', 'CMI', 'VRSK',
        'RSG', 'ODFL', 'TDG', 'CPRT', 'PWR', 'IR', 'HWM', 'ROK', 'DAL', 'WAB',
        'DOV', 'XYL', 'OTIS', 'UAL', 'GWW', 'SWK', 'LUV', 'J',
    ],
    'XLB': [
        'LIN', 'NEM', 'FCX', 'SHW', 'CRH', 'APD', 'ECL', 'DD', 'CTVA', 'DOW',
        'NUE', 'VMC', 'MLM', 'PPG', 'IFF', 'STLD', 'BALL', 'PKG', 'LYB', 'IP',
        'ALB', 'FMC', 'CE', 'EMN', 'CF', 'MOS', 'AMCR',
    ],
    'XOP': [
        'VG', 'XOM', 'CVX', 'CRC', 'TPL', 'COP', 'EQT', 'VLO', 'FANG', 'MPC',
        'PSX', 'EOG', 'CHRD', 'OXY', 'HES', 'DVN', 'EXE', 'CTRA', 'PBF', 'AR',
        'DK', 'MTDR', 'RRC', 'PR', 'DINO', 'SM', 'OVV', 'MGY', 'MUR', 'APA',
    ],
    'FCG': [
        'EQT', 'AR', 'RRC', 'EXE', 'CTRA', 'DVN', 'COP', 'EOG', 'OVV', 'AM',
        'CNX', 'BKR', 'WMB', 'KMI', 'OKE', 'HAL', 'SLB', 'TRGP', 'SM', 'MTDR',
        'PR', 'MUR', 'FANG', 'OXY',
    ],
    'VNQ': [
        'PLD', 'AMT', 'EQIX', 'WELL', 'DLR', 'SPG', 'PSA', 'CCI', 'CBRE', 'O',
        'VICI', 'EXR', 'IRM', 'SBAC', 'AVB', 'WY', 'ARE', 'EQR', 'INVH', 'ESS',
        'MAA', 'KIM', 'REG', 'UDR', 'CPT', 'HST', 'DOC', 'CUBE', 'BXP', 'SUI',
    ],
    'PAVE': [
        'ETN', 'URI', 'PWR', 'EME', 'FAST', 'NUE', 'VMC', 'MLM', 'STLD', 'CMI',
        'TT', 'CARR', 'DOV', 'XYL', 'IR', 'J', 'WMS', 'GNRC', 'FLR', 'MAS',
        'ROK', 'RS', 'MTZ', 'CLF', 'CSL', 'TREX', 'ACM', 'SWK', 'AWI', 'FIX',
    ],
    'VTV': [
        'BRK.B', 'JPM', 'UNH', 'XOM', 'JNJ', 'PG', 'HD', 'ABBV', 'BAC', 'CVX',
        'MRK', 'WFC', 'KO', 'PEP', 'CSCO', 'WMT', 'ABT', 'PM', 'DIS', 'CMCSA',
        'VZ', 'T', 'NEE', 'DHR', 'BMY', 'INTC', 'GS', 'RTX', 'SPGI', 'CAT',
    ],
    'REMX': [
        'ALB', 'MP', 'SQM', 'SGML', 'LAC', 'UUUU', 'TREM', 'CENX', 'ALTM',
    ],
    'URA': [
        'CCJ', 'NXE', 'UEC', 'DNN', 'UUUU', 'SRUUF', 'ENCUF', 'LEU', 'SMR',
        'LTBR', 'URG',
    ],
    'XLV': [
        'LLY', 'UNH', 'JNJ', 'ABBV', 'MRK', 'TMO', 'ABT', 'AMGN', 'DHR', 'PFE',
        'ISRG', 'BSX', 'VRTX', 'SYK', 'GILD', 'MDT', 'REGN', 'ELV', 'ZTS', 'BDX',
        'CI', 'HCA', 'EW', 'IDXX', 'A', 'MCK', 'IQV', 'GEHC', 'RMD', 'DXCM',
        'MTD', 'COO', 'HOLX', 'TFX', 'WST', 'ALGN', 'TECH', 'PODD', 'BAX', 'VTRS',
        'ZBH', 'HSIC', 'CRL', 'XRAY', 'DGX', 'LH', 'INCY', 'BIIB', 'MOH', 'CNC',
        'HUM', 'CAH',
    ],
    'XLU': [
        'NEE', 'SO', 'DUK', 'CEG', 'VST', 'SRE', 'AEP', 'D', 'PCG', 'PEG',
        'EXC', 'XEL', 'ED', 'WEC', 'ES', 'AWK', 'ETR', 'FE', 'PPL', 'CMS',
        'ATO', 'CNP', 'EVRG', 'NI', 'AES', 'LNT', 'NRG', 'PNW', 'DTE', 'OGE',
        'BKH',
    ],
    'QQQ': [
        'NVDA', 'MSFT', 'AAPL', 'AVGO', 'AMZN', 'TSLA', 'META', 'GOOGL', 'GOOG',
        'NFLX', 'PLTR', 'COST', 'CSCO', 'TMUS', 'AMD', 'LIN', 'APP', 'INTU', 'PEP',
        'MU', 'SHOP', 'QCOM', 'BKNG', 'TXN', 'LRCX', 'AMAT', 'ISRG', 'ADBE', 'INTC',
        'AMGN', 'KLAC', 'GILD', 'PANW', 'HON', 'MELI', 'CRWD', 'ADI', 'ADP', 'CMCSA',
        'DASH', 'CEG', 'VRTX', 'SBUX', 'CDNS', 'ORLY', 'SNPS', 'PDD', 'MSTR', 'CTAS',
        'ASML', 'MDLZ', 'MAR', 'MRVL', 'TRI', 'ADSK', 'CSX', 'PYPL', 'MNST', 'FTNT',
        'AEP', 'REGN', 'NXPI', 'FAST', 'AXON', 'ROP', 'WDAY', 'ABNB', 'PCAR', 'EA',
        'IDXX', 'BKR', 'ROST', 'TTWO', 'XEL', 'DDOG', 'WBD', 'ZS', 'PAYX', 'EXC',
        'AZN', 'CPRT', 'FANG', 'CCEP', 'CHTR', 'CSGP', 'KDP', 'VRSK', 'MCHP', 'GEHC',
        'CTSH', 'KHC', 'ODFL', 'TEAM', 'DXCM', 'TTD', 'CDW', 'GFS', 'LULU', 'BIIB',
        'ON', 'ARM',
    ],
    'ARKK': [
        'TSLA', 'CRSP', 'TEM', 'ROKU', 'SHOP', 'AMD', 'HOOD', 'COIN', 'RBLX', 'BEAM',
        'TER', 'PLTR', 'TWST', 'CRCL', 'ACHR', 'BLSH', 'TXG', 'BMNR', 'CRWV', 'NTLA',
        'TSM', 'AMZN', 'NVDA', 'DE', 'BIDU', 'NTRA', 'RXRX', 'BWXT', 'KTOS',
        'ILMN', 'VCYT', 'META', 'WGS', 'GOOG', 'BABA', 'ABNB', 'SOFI', 'PD', 'PACB',
        'CERS', 'TTD',
    ],
    'XLC': [
        'META', 'GOOGL', 'GOOG', 'WBD', 'EA', 'NFLX', 'DIS', 'TTWO', 'OMC', 'VZ',
        'CMCSA', 'T', 'TMUS', 'LYV', 'CHTR', 'TTD', 'FOXA', 'TKO', 'NWSA', 'FOX',
        'MTCH', 'PSKY', 'NWS',
    ],
    'XLY': [
        'AMZN', 'TSLA', 'HD', 'MCD', 'TJX', 'BKNG', 'LOW', 'SBUX', 'ORLY', 'DASH',
        'NKE', 'GM', 'MAR', 'RCL', 'HLT', 'AZO', 'ROST', 'ABNB', 'F', 'CMG', 'DHI',
        'YUM', 'EBAY', 'GRMN', 'EXPE', 'DECK', 'LEN', 'POOL', 'NVR', 'LVS', 'APTV',
        'BBY', 'DRI', 'PHM', 'ULTA', 'DPZ', 'GPC', 'TPR', 'WSM', 'LULU', 'CCL', 'TSCO',
        'KMX', 'HAS', 'MGM', 'CZR', 'WYNN', 'RL', 'ETSY', 'MHK',
    ],
    'XLP': [
        'PG', 'COST', 'WMT', 'KO', 'PEP', 'PM', 'MDLZ', 'CL', 'MO', 'TGT', 'GIS',
        'SYY', 'KMB', 'STZ', 'HSY', 'KHC', 'CHD', 'MKC', 'CAG', 'CLX', 'SJM',
        'TSN', 'HRL', 'ADM', 'KR', 'WBA', 'TAP', 'CPB', 'LW', 'EL', 'MNST',
        'KVUE', 'BG', 'FDP', 'CASY', 'USFD',
    ],
}

# ETFs in QUAD_ALLOCATIONS that we don't have mappings for yet
UNMAPPED_EQUITY_ETFS = {'IWM', 'IWD', 'EEM', 'VUG', 'EWX', 'VWO'}


def get_stock_universe(etf_weights: dict, top_n: int = 4) -> set:
    """
    Given a dict of {etf_ticker: weight}, return the combined set of
    constituent stocks for the top N equity ETFs by weight.

    Non-equity ETFs and unmapped ETFs are skipped.

    Args:
        etf_weights: dict from signal generator, e.g. {'XLE': 0.15, 'GLD': 0.08}
        top_n: number of top ETFs (by weight) to pull stocks for (default 4)

    Returns:
        Set of stock ticker strings
    """
    sorted_etfs = sorted(etf_weights.items(), key=lambda x: x[1], reverse=True)

    stocks = set()
    skipped = []
    unmapped = []
    mapped = []

    for etf, weight in sorted_etfs:
        if etf in NON_EQUITY_ETFS:
            skipped.append(etf)
            continue
        if etf in ETF_CONSTITUENTS:
            if len(mapped) < top_n:
                stocks.update(ETF_CONSTITUENTS[etf])
                mapped.append(etf)
        elif etf in UNMAPPED_EQUITY_ETFS:
            unmapped.append(etf)
        else:
            skipped.append(etf)

    if mapped:
        print(f"  Pulling stocks for top {len(mapped)} ETFs: {', '.join(mapped)}")
    if skipped:
        print(f"  Skipped (non-equity): {', '.join(sorted(skipped))}")
    if unmapped:
        print(f"  Unmapped (need constituents): {', '.join(sorted(unmapped))}")

    return stocks


def get_constituent_tickers_for_universe(etf_tickers: set) -> set:
    """
    Return the set of all constituent stock tickers for every mapped equity ETF
    in etf_tickers. Use this to build the full data-fetch universe (ETFs + stocks).

    Non-equity and unmapped ETFs are skipped (no constituents added).
    """
    out = set()
    for etf in etf_tickers:
        if etf in NON_EQUITY_ETFS or etf in UNMAPPED_EQUITY_ETFS:
            continue
        if etf in ETF_CONSTITUENTS:
            out.update(ETF_CONSTITUENTS[etf])
    return out


def get_etf_for_stock(stock: str) -> list:
    """Return which ETFs a given stock belongs to."""
    return [etf for etf, holdings in ETF_CONSTITUENTS.items() if stock in holdings]


def is_equity_etf_mapped(etf: str) -> bool:
    """True if we have constituent data and it's an equity ETF (not bonds/commodities/crypto)."""
    return etf not in NON_EQUITY_ETFS and etf in ETF_CONSTITUENTS


if __name__ == "__main__":
    print("ETF Constituent Mapper")
    print("=" * 40)
    total = 0
    for etf, stocks in sorted(ETF_CONSTITUENTS.items()):
        print(f"  {etf:<6} {len(stocks):>3} holdings")
        total += len(stocks)

    all_stocks = set()
    for stocks in ETF_CONSTITUENTS.values():
        all_stocks.update(stocks)

    print(f"  {'':->40}")
    print(f"  Total: {total} holdings, {len(all_stocks)} unique stocks")
    print()
    print(f"  Unmapped equity ETFs: {', '.join(sorted(UNMAPPED_EQUITY_ETFS))}")
    print(f"  Non-equity ETFs: {', '.join(sorted(NON_EQUITY_ETFS))}")
