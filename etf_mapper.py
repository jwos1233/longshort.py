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
        'XOM', 'CVX', 'COP', 'WMB', 'SLB', 'EOG', 'KMI', 'BKR', 'VLO', 'PSX',
        'MPC', 'OKE', 'TRGP', 'EQT', 'OXY', 'FANG', 'HAL', 'TPL', 'DVN', 'EXE',
        'CTRA', 'APA',
    ],
    'LIT': [
        'LCID', 'LAC', 'SGML', 'SLI', 'RIO', 'ALB', 'SQM', 'TSLA',
    ],
    'XLF': [
        'BRK.B', 'JPM', 'V', 'MA', 'BAC', 'GS', 'WFC', 'MS', 'C', 'AXP',
        'SCHW', 'BLK', 'SPGI', 'COF', 'PGR', 'CB', 'CME', 'ICE', 'MRSH', 'USB',
        'PNC', 'BX', 'BK', 'MCO', 'AON', 'TRV', 'TFC', 'KKR', 'HOOD', 'AJG',
        'ALL', 'AFL', 'APO', 'FITB', 'AMP', 'AIG', 'MSCI', 'PYPL', 'MET', 'COIN',
        'NDAQ', 'HIG', 'STT', 'ACGL', 'PRU', 'HBAN', 'MTB', 'FISV', 'IBKR', 'CBOE',
        'XYZ', 'WTW', 'RJF', 'NTRS', 'CFG', 'SYF', 'FIS', 'CINF', 'RF', 'ARES',
        'CPAY', 'WRB', 'TROW', 'BRO', 'KEY', 'PFG', 'L', 'GPN', 'EG', 'IVZ',
        'JKHY', 'GL', 'AIZ', 'BEN', 'FDS', 'ERIE',
    ],
    'XLI': [
        'GE', 'CAT', 'RTX', 'GEV', 'BA', 'UNP', 'DE', 'UBER', 'HON', 'ETN',
        'LMT', 'PH', 'HWM', 'TT', 'NOC', 'GD', 'MMM', 'ADP', 'JCI', 'WM',
        'UPS', 'EMR', 'PWR', 'FDX', 'CMI', 'CSX', 'ITW', 'TDG', 'NSC', 'CTAS',
        'LHX', 'PCAR', 'URI', 'AME', 'FAST', 'FIX', 'CARR', 'GWW', 'ROK', 'DAL',
        'RSG', 'WAB', 'AXON', 'UAL', 'ODFL', 'OTIS', 'IR', 'EME', 'CPRT', 'XYL',
        'DOV', 'PAYX', 'VRSK', 'HUBB', 'LUV', 'EFX', 'VLTO', 'LDOS', 'CHRW', 'BR',
        'SNA', 'EXPD', 'FTV', 'ROL', 'TXT', 'LII', 'HII', 'JBHT', 'J', 'PNR',
        'IEX', 'NDSN', 'MAS', 'ALLE', 'GNRC', 'SWK', 'BLDR', 'AOS', 'PAYC',
    ],
    'XLB': [
        'LIN', 'NEM', 'FCX', 'SHW', 'CRH', 'ECL', 'CTVA', 'APD', 'MLM', 'NUE',
        'VMC', 'PPG', 'STLD', 'SW', 'IP', 'AMCR', 'ALB', 'DD', 'DOW', 'IFF',
        'PKG', 'BALL', 'CF', 'AVY', 'LYB', 'MOS',
    ],
    'XOP': [
        'TPL', 'VG', 'XOM', 'OXY', 'CVX', 'PR', 'CRC', 'MGY', 'OVV', 'VLO',
        'COP', 'VNOM', 'DVN', 'CTRA', 'PBF', 'APA', 'EOG', 'MTDR', 'PSX', 'MPC',
        'RRC', 'CHRD', 'EQT', 'FANG', 'DINO', 'GPOR', 'CNX', 'MUR', 'AR', 'EXE',
        'CRGY', 'DK', 'NOG', 'SM', 'PARR', 'CRK', 'CLMT', 'KOS', 'TALO', 'WKC',
        'GPRE', 'SOC', 'CVI', 'BKV', 'VTS', 'GEVO', 'REX', 'EGY', 'SD', 'CLNE',
    ],
    'VNQ': [
        'PLD', 'AMT', 'EQIX', 'WELL', 'DLR', 'SPG', 'PSA', 'CCI', 'CBRE', 'O',
        'VICI', 'EXR', 'IRM', 'SBAC', 'AVB', 'WY', 'ARE', 'EQR', 'INVH', 'ESS',
        'MAA', 'KIM', 'REG', 'UDR', 'CPT', 'HST', 'DOC', 'CUBE', 'BXP', 'SUI',
    ],
    'FCG': [
        'COP', 'OXY', 'HESM', 'EOG', 'FANG', 'PR', 'DVN', 'WES', 'CTRA', 'EQT',
        'APA', 'OVV', 'EXE', 'MTDR', 'WDS', 'RRC', 'AR', 'CHRD', 'NFG', 'MGY',
        'CNX', 'MUR', 'SM', 'KOS', 'BTE', 'NOG', 'GPOR', 'CRGY', 'VET', 'CRK',
        'BKV', 'VTS', 'SD', 'NUAI', 'REPX', 'OBE', 'GRNT', 'WTI',
    ],
    'PAVE': [
        'PWR', 'HWM', 'CSX', 'DE', 'TT', 'UNP', 'ETN', 'SRE', 'NSC', 'PH',
        'FAST', 'EMR', 'CRH', 'ROK', 'URI', 'MLM', 'VMC', 'NUE', 'AMRZ', 'EME',
        'STLD', 'HUBB', 'WWD', 'MTZ', 'ATI', 'CRS', 'FTV', 'RBC', 'CSL', 'RS',
        'J', 'AA', 'PNR', 'TRMB', 'LECO', 'IEX', 'GGG', 'RRX', 'RPM', 'WCC',
        'WMS', 'MLI', 'STRL', 'BLD', 'ACM', 'WLK', 'DY',
    ],
    'VTV': [
        'BRK.B', 'JPM', 'UNH', 'XOM', 'JNJ', 'PG', 'HD', 'ABBV', 'BAC', 'CVX',
        'MRK', 'WFC', 'KO', 'PEP', 'CSCO', 'WMT', 'ABT', 'PM', 'DIS', 'CMCSA',
        'VZ', 'T', 'NEE', 'DHR', 'BMY', 'INTC', 'GS', 'RTX', 'SPGI', 'CAT',
    ],
    'REMX': [
        'ALB', 'MP', 'SQM', 'IPX', 'SLI', 'LAC', 'TROX', 'AA',
    ],
    'URA': [
        'OKLO', 'UEC', 'EFR', 'U-U', 'LEU', 'SMR', 'NNE', 'URG', 'URC', 'EU', 'LI',
        'CCJ', 'NXE', 'DNN', 'UUUU',
    ],
    'XLV': [
        'LLY', 'JNJ', 'ABBV', 'MRK', 'UNH', 'AMGN', 'ABT', 'TMO', 'ISRG', 'GILD',
        'PFE', 'SYK', 'DHR', 'MDT', 'BMY', 'VRTX', 'MCK', 'BSX', 'CVS', 'HCA',
        'REGN', 'CI', 'ELV', 'COR', 'ZTS', 'CAH', 'IDXX', 'BDX', 'EW', 'GEHC',
        'RMD', 'A', 'WAT', 'IQV', 'DXCM', 'MTD', 'BIIB', 'STE', 'LH', 'DGX',
        'HUM', 'CNC', 'ZBH', 'MRNA', 'WST', 'PODD', 'VTRS', 'HOLX', 'COO', 'INCY',
        'ALGN', 'SOLV', 'UHS', 'RVTY', 'BAX', 'TECH', 'CRL', 'HSIC', 'MOH', 'DVA',
    ],
    'XLU': [
        'NEE', 'SO', 'CEG', 'DUK', 'AEP', 'SRE', 'VST', 'D', 'XEL', 'EXC',
        'ETR', 'PEG', 'PCG', 'ED', 'WEC', 'NRG', 'DTE', 'AEE', 'ATO', 'EIX',
        'PPL', 'ES', 'CNP', 'FE', 'AWK', 'CMS', 'NI', 'EVRG', 'LNT', 'PNW',
        'AES',
    ],
    'QQQ': [
        'NVDA', 'AAPL', 'MSFT', 'AMZN', 'TSLA', 'META', 'GOOGL', 'WMT', 'GOOG',
        'AVGO', 'MU', 'COST', 'NFLX', 'AMD', 'PLTR', 'CSCO', 'LRCX', 'AMAT',
        'TMUS', 'LIN', 'PEP', 'INTC', 'AMGN', 'KLAC', 'TXN', 'ISRG', 'GILD',
        'ADI', 'QCOM', 'SHOP', 'HON', 'BKNG', 'APP', 'ASML', 'VRTX', 'PANW',
        'CEG', 'SBUX', 'CMCSA', 'INTU', 'ADBE', 'WDC', 'CRWD', 'MAR', 'ADP',
        'MELI', 'STX', 'MNST', 'CDNS', 'CTAS', 'REGN', 'CSX', 'SNPS', 'MDLZ',
        'ORLY', 'DASH', 'WBD', 'AEP', 'PDD', 'MRVL', 'ROST', 'PCAR', 'BKR',
        'FTNT', 'NXPI', 'ABNB', 'MPWR', 'FER', 'IDXX', 'FAST', 'EA', 'CCEP',
        'ADSK', 'XEL', 'EXC', 'FANG', 'TRI', 'ALNY', 'AXON', 'PYPL', 'ODFL',
        'KDP', 'MCHP', 'TTWO', 'GEHC', 'ROP', 'DDOG', 'CPRT', 'MSTR', 'PAYX',
        'INSM', 'CTSH', 'WDAY', 'CHTR', 'KHC', 'DXCM', 'VRSK', 'ZS', 'CSGP',
        'ARM', 'TEAM',
    ],
    'ARKK': [
        'TSLA', 'CRSP', 'TEM', 'ROKU', 'SHOP', 'COIN', 'HOOD', 'RBLX', 'BEAM', 'AMD',
        'CRCL', 'PLTR', 'TER', 'TWST', 'TXG', 'BLSH', 'ACHR', 'BMNR', 'NTLA', 'CRWV',
        'TSM', 'DE', 'AMZN', 'NVDA', 'ILMN', 'RXRX', 'NTRA', 'XYZ', 'VCYT', 'BIDU',
        'BWXT',
    ],
    'XLC': [
        'META', 'GOOGL', 'GOOG', 'VZ', 'T', 'CMCSA', 'NFLX', 'TMUS', 'EA', 'WBD',
        'DIS', 'TTWO', 'OMC', 'LYV', 'CHTR', 'TKO', 'FOXA', 'TTD', 'NWSA', 'FOX',
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
        'WMT', 'COST', 'PG', 'KO', 'PM', 'CL', 'PEP', 'MO', 'MDLZ', 'MNST',
        'TGT', 'SYY', 'KDP', 'KR', 'KMB', 'KVUE', 'HSY', 'DG', 'ADM', 'EL',
        'CHD', 'DLTR', 'GIS', 'STZ', 'KHC', 'TSN', 'MKC', 'BG', 'CLX', 'SJM',
        'CAG', 'TAP', 'HRL', 'LW', 'CPB',
    ],
    # IWM: Russell 2000 — top 40 by market cap (representative list; reconstitution varies)
    'IWM': [
        'CRDO', 'IONQ', 'BE', 'OKLO', 'KTOS', 'FN', 'CDE', 'STRL', 'RMBS', 'HIMS',
        'NXT', 'RGTI', 'SATS', 'AVAV', 'ENSG', 'CSU', 'ELF', 'CVNA', 'ONTO', 'SSD',
        'VKTX', 'BLDR', 'MTH', 'WSM', 'FIX', 'ITCI', 'PODD', 'AXSM', 'NBIX', 'EXEL',
        'BPMC', 'SRPT', 'ALGM', 'LEN', 'DHI', 'PHM', 'TOL', 'MHO', 'MDC', 'NVR',
    ],
    'ROBO': [
        'TER', 'IPGP', 'NOVT', 'NDSN', 'ROK', 'DE', 'CGNX', 'COHR', 'EMR', 'ONDS',
        'JBT', 'ISRG', 'ILMN', 'SYM', 'AMBA', 'GMED', 'NVDA', 'CLS', 'XPEV', 'MCHP',
        'SSYS', 'GXO', 'ZBRA', 'PTC', 'CDNS', 'MANH', 'APTV', 'TRMB', 'QCOM', 'SERV',
        'ADSK', 'TSLA', 'JOBY', 'IOT',
    ],
    'JEDI': [
        'PL', 'RKLB', 'ASTS', 'VSAT', 'SATS', 'LUNR', 'HXL', 'GSAT', 'IRDM', 'NN',
        'DCO', 'RDW', 'BKSY',
    ],
}

# ETFs in QUAD_ALLOCATIONS that we don't have mappings for yet
UNMAPPED_EQUITY_ETFS = {'IWD', 'EEM', 'VUG', 'EWX', 'VWO'}


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
