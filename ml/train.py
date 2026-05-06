"""
ml/train.py — Upgraded self-learning training script v4.

What's new vs v3:
  ┌─────────────────────────────────────────────────────────────────────┐
  │ FEATURES                                                            │
  │  + Volume Spike  (abnormal volume = potential big move)             │
  │  + ATR           (Average True Range — volatility in price terms)   │
  │  + 52W High/Low  (proximity to yearly support/resistance)           │
  │  + Market Return (Nifty 50 daily return as market context)          │
  │  + Market Regime (is Nifty above its 200 MA? bull vs bear)         │
  │  + Earnings Season (binary — results months have different patterns)│
  │  + Sector        (Banking/IT/Auto etc — stock personality)          │
  ├─────────────────────────────────────────────────────────────────────┤
  │ TARGET                                                              │
  │  Changed: 5-day forward return > 0.5% (was: next-day > 0%)        │
  │  Why: Filters noise. More stable and tradeable signal.             │
  ├─────────────────────────────────────────────────────────────────────┤
  │ MODELS                                                              │
  │  Per-stock models  → DATA_DIR/models/{TICKER}.pkl                  │
  │  Global fallback   → DATA_DIR/model.pkl                            │
  │  Per-stock is used when available; global as fallback for new tickers│
  ├─────────────────────────────────────────────────────────────────────┤
  │ SIGNAL                                                              │
  │  BUY     if prob > 0.65                                            │
  │  NEUTRAL if 0.35 <= prob <= 0.65  (too uncertain)                 │
  │  SELL    if prob < 0.35                                            │
  └─────────────────────────────────────────────────────────────────────┘
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(BASE_DIR)
DATA_DIR    = os.environ.get("DATA_DIR", os.path.join(ROOT_DIR, "data"))
MODELS_DIR  = os.path.join(DATA_DIR, "models")   # per-stock models live here
os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

STOCKS_FILE = os.path.join(DATA_DIR, "known_stocks.json")
MODEL_PATH  = os.path.join(DATA_DIR, "model.pkl")          # global fallback
LE_PATH     = os.path.join(DATA_DIR, "label_encoder.pkl")
META_PATH   = os.path.join(DATA_DIR, "model_metadata.json")

# ── Config ──────────────────────────────────────────────────────────────────────
# Complete Nifty 50 universe (as of 2025)
SEED_STOCKS = [
    # Banking
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS",
    "KOTAKBANK.NS", "INDUSINDBK.NS",
    # Finance & Insurance
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "SBILIFE.NS", "HDFCLIFE.NS",
    "SHRIRAMFIN.NS",
    # Information Technology
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS",
    # Energy & Oil
    "RELIANCE.NS", "ONGC.NS", "BPCL.NS", "COALINDIA.NS", "NTPC.NS",
    "POWERGRID.NS",
    # FMCG & Consumer
    "HINDUNILVR.NS", "ITC.NS", "BRITANNIA.NS", "NESTLEIND.NS", "TATACONSUM.NS",
    # Automobile
    "TATAMOTORS.NS", "MARUTI.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS",
    "EICHERMOT.NS", "M&M.NS",
    # Infrastructure & Cement
    "LT.NS", "ADANIPORTS.NS", "ULTRACEMCO.NS", "GRASIM.NS",
    # Pharma & Healthcare
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS",
    # Metals & Mining
    "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS",
    # Other / Consumer / Telecom
    "ASIANPAINT.NS", "TITAN.NS", "TRENT.NS", "BHARTIARTL.NS",
]
PERIOD       = "5y"
N_SPLITS     = 5       # for global model CV
BUY_THRESH   = 0.65
SELL_THRESH  = 0.35
RETURN_DAYS  = 5       # predict 5-day forward return
RETURN_MIN   = 0.005   # must be > 0.5% to count as BUY

# ── Sector map ──────────────────────────────────────────────────────────────────
# 0=Banking, 1=Finance/Insurance, 2=IT, 3=Auto, 4=Energy, 5=FMCG,
# 6=Infra/Cement, 7=Pharma, 8=Metals, 9=Other/Telecom/Consumer
SECTOR_MAP = {
    # Banking
    "HDFCBANK.NS":0,   "ICICIBANK.NS":0,  "SBIN.NS":0,      "AXISBANK.NS":0,
    "KOTAKBANK.NS":0,  "INDUSINDBK.NS":0, "BANDHANBNK.NS":0,
    # Finance & Insurance
    "BAJFINANCE.NS":1, "BAJAJFINSV.NS":1, "SBILIFE.NS":1,   "HDFCLIFE.NS":1,
    "SHRIRAMFIN.NS":1, "MUTHOOTFIN.NS":1,
    # IT
    "TCS.NS":2,        "INFY.NS":2,       "WIPRO.NS":2,     "HCLTECH.NS":2,
    "TECHM.NS":2,      "LTIM.NS":2,       "MPHASIS.NS":2,   "COFORGE.NS":2,
    # Auto
    "TATAMOTORS.NS":3, "MARUTI.NS":3,     "BAJAJ-AUTO.NS":3,"HEROMOTOCO.NS":3,
    "EICHERMOT.NS":3,  "M&M.NS":3,        "TVSMOTOR.NS":3,
    # Energy & Oil
    "RELIANCE.NS":4,   "ONGC.NS":4,       "BPCL.NS":4,      "COALINDIA.NS":4,
    "NTPC.NS":4,       "POWERGRID.NS":4,  "IOC.NS":4,       "GAIL.NS":4,
    "ADANIGREEN.NS":4,
    # FMCG & Consumer Staples
    "HINDUNILVR.NS":5, "ITC.NS":5,        "BRITANNIA.NS":5, "NESTLEIND.NS":5,
    "TATACONSUM.NS":5, "DABUR.NS":5,      "MARICO.NS":5,    "COLPAL.NS":5,
    # Infrastructure & Cement
    "LT.NS":6,         "ADANIPORTS.NS":6, "ULTRACEMCO.NS":6,"GRASIM.NS":6,
    "ADANIENT.NS":6,   "SIEMENS.NS":6,    "ABB.NS":6,
    # Pharma & Healthcare
    "SUNPHARMA.NS":7,  "DRREDDY.NS":7,    "CIPLA.NS":7,     "DIVISLAB.NS":7,
    "APOLLOHOSP.NS":7, "MAXHEALTH.NS":7,  "FORTIS.NS":7,
    # Metals & Mining
    "TATASTEEL.NS":8,  "JSWSTEEL.NS":8,   "HINDALCO.NS":8,  "VEDL.NS":8,
    "SAIL.NS":8,       "NMDC.NS":8,
    # Other / Consumer Discretionary / Telecom / Retail
    "ASIANPAINT.NS":9, "TITAN.NS":9,      "TRENT.NS":9,     "BHARTIARTL.NS":9,
    "PIDILITIND.NS":9, "DMART.NS":9,      "NYKAA.NS":9,     "ZOMATO.NS":9,
}

# ── Feature column definitions ──────────────────────────────────────────────────
# Per-stock model uses these (no Ticker/Sector — they'd be constant per stock)
STOCK_FEATURES = [
    # Core technical
    "RSI", "MA50", "MA200", "MA_Cross", "Volatility",
    "MACD", "MACD_Signal", "MACD_Hist", "BB_Width",
    "Volume_Log", "Volume_Spike", "ATR",
    "High52W_Pct", "Low52W_Pct",
    # Market context
    "Market_Return", "Market_Regime", "Earnings_Season",
    # Phase 1: Momentum + Beta
    "Return_1d", "Return_5d_lag", "Return_20d",
    "Beta_60d", "Rel_Strength",
    # Phase 4: Market structure — WHERE are we in the cycle?
    "Dist_MA20",    # % distance from 20-day MA
    "Dist_MA50",    # % distance from 50-day MA
    "MA20_Slope",   # 5-day slope of MA20 (trend angle)
    "MA50_Slope",   # 10-day slope of MA50
    "BB_Position",  # 0=lower band, 0.5=middle, 1=upper band
    # Phase 5B: Macro — Dollar, Crude, Seasonal
    "USDINR_Return",   "USDINR_20d_Mom",
    "Crude_Return",    "Crude_20d_Mom",
    "Month_Sin",       "Month_Cos",
    "Is_Budget_Month", "Is_Monsoon",
    # Phase 6: Global macro (universal)
    "SP500_Return",    "SP500_5d",
    "VIX_US_Level",    "VIX_IN_ROC5",   "VIX_IN_Pct",
    "US10Y_Level",     "US10Y_Chg",
    "FII_Proxy",
    "Copper_Return",   "Shanghai_Return",
    # Phase 6: Sector-conditional signals
    "NASDAQ_IT",       "USD_Export",
    "Crude_Sector",    "Copper_Sector",
    "Shanghai_Sector", "Yield_Banking",
    "Monsoon_FMCG",
]
# Global fallback model adds stock-identity features
GLOBAL_FEATURES = STOCK_FEATURES + ["Ticker", "Sector"]

# ── Stock registry helpers ────────────────────────────────────────────────────────
def load_stocks() -> list:
    if os.path.exists(STOCKS_FILE):
        with open(STOCKS_FILE) as f:
            existing = json.load(f)
        # Merge any new SEED_STOCKS not already in the registry
        new_seeds = [s for s in SEED_STOCKS if s not in existing]
        if new_seeds:
            merged = sorted(set(existing + new_seeds))
            save_stocks(merged)
            print(f"📋 Registry expanded: {len(existing)} → {len(merged)} stocks "
                  f"(added: {', '.join(new_seeds)})")
            return merged
        print(f"📋 Registry: {len(existing)} stocks")
        return existing
    print(f"📋 No registry — seeding with {len(SEED_STOCKS)} Nifty 50 stocks")
    save_stocks(SEED_STOCKS)
    return SEED_STOCKS.copy()

def save_stocks(stocks: list):
    with open(STOCKS_FILE, "w") as f:
        json.dump(sorted(set(stocks)), f, indent=2)

# ── Indicators ──────────────────────────────────────────────────────────────────
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.where(delta > 0, 0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / loss))

def calc_macd(series: pd.Series, fast=12, slow=26, sig=9):
    ml = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()
    sl = ml.ewm(span=sig, adjust=False).mean()
    return ml, sl

def calc_bollinger_width(series: pd.Series, period: int = 20) -> pd.Series:
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (sma + 2*std - (sma - 2*std)) / sma

def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def is_earnings_season(index: pd.DatetimeIndex) -> pd.Series:
    """
    Indian companies report quarterly results in:
      Q4: Apr 15 – May 31   |  Q1: Jul 15 – Aug 31
      Q2: Oct 15 – Nov 30   |  Q3: Jan 15 – Feb 28
    Returns a binary Series aligned to the index.
    """
    m = index.month
    d = index.day
    mask = (
        ((m == 4) & (d >= 15)) | (m == 5) |
        ((m == 7) & (d >= 15)) | (m == 8) |
        ((m == 10) & (d >= 15))| (m == 11)|
        ((m == 1) & (d >= 15)) | (m == 2)
    )
    return pd.Series(mask.astype(int), index=index)

# ── Feature builder ──────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame,
                   ticker_code:  int,
                   sector_code:  int,
                   nifty_close:  pd.Series,
                   nifty_ma200:  pd.Series,
                   nifty_return: pd.Series,
                   usdinr_close:   pd.Series = None,
                   crude_close:    pd.Series = None,
                   sp500_close:    pd.Series = None,
                   nasdaq_close:   pd.Series = None,
                   vix_us_close:   pd.Series = None,
                   vix_in_close:   pd.Series = None,
                   us10y_close:    pd.Series = None,
                   copper_close:   pd.Series = None,
                   shanghai_close: pd.Series = None) -> pd.DataFrame:
    df    = df.copy()
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    vol   = df["Volume"].squeeze()

    # ── Core indicators (existing) ──
    df["RSI"]        = calc_rsi(close)
    df["MA50"]       = close.rolling(50).mean()
    df["MA200"]      = close.rolling(200).mean()
    df["MA_Cross"]   = df["MA50"] - df["MA200"]
    df["Volatility"] = close.pct_change().rolling(10).std()
    ml, sl           = calc_macd(close)
    df["MACD"]       = ml
    df["MACD_Signal"]= sl
    df["MACD_Hist"]  = ml - sl
    df["BB_Width"]   = calc_bollinger_width(close)
    df["Volume_Log"] = np.log1p(vol)

    # ── New features ──
    # Volume spike: today vs 20-day average
    df["Volume_Spike"]  = vol / vol.rolling(20).mean()

    # ATR: volatility in price terms
    df["ATR"]           = calc_atr(high, low, close)

    # 52-week position
    df["High52W_Pct"]   = close / close.rolling(252).max()
    df["Low52W_Pct"]    = close / close.rolling(252).min()

    # Market context from Nifty (aligned by date)
    nifty_c = nifty_close.reindex(df.index).ffill()
    nifty_r = nifty_return.reindex(df.index).fillna(0)
    nifty_m = nifty_ma200.reindex(df.index).ffill()

    df["Market_Return"] = nifty_r
    df["Market_Regime"] = (nifty_c > nifty_m).astype(int)

    # Earnings season
    df["Earnings_Season"] = is_earnings_season(df.index)

    # ── Phase 1: Momentum features ──────────────────────────────────────────
    daily_ret            = close.pct_change()
    df["Return_1d"]      = daily_ret                          # yesterday's return
    df["Return_5d_lag"]  = close.pct_change(5)               # 5-day lagged return
    df["Return_20d"]     = close.pct_change(20)              # 20-day momentum

    # ── Phase 1: Rolling Beta vs Nifty ──────────────────────────────────────
    # Beta = cov(stock, nifty) / var(nifty) over 60-day rolling window
    cov_60   = daily_ret.rolling(60).cov(nifty_r)
    var_60   = nifty_r.rolling(60).var()
    df["Beta_60d"]       = (cov_60 / var_60).clip(-3, 3)    # clip extremes

    # ── Phase 1: Relative strength vs Nifty (20d) ───────────────────────────
    stock_20d = close.pct_change(20)
    nifty_20d = nifty_c.pct_change(20)
    df["Rel_Strength"]   = stock_20d - nifty_20d             # outperformance

    # ── Phase 4: Market structure features ──────────────────────────────────
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    # Distance from MAs (where are we relative to the mean?)
    df["Dist_MA20"]   = (close - ma20) / (ma20 + 1e-9)
    df["Dist_MA50"]   = (close - ma50) / (ma50 + 1e-9)

    # MA slope: rate of change of the MA itself (trend angle)
    df["MA20_Slope"]  = ma20.pct_change(5)     # 5-day slope
    df["MA50_Slope"]  = ma50.pct_change(10)    # 10-day slope

    # Bollinger Band position (0=lower band, 0.5=midline, 1=upper band)
    std20 = close.rolling(20).std()
    bb_lower = ma20 - 2 * std20
    bb_upper = ma20 + 2 * std20
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    df["BB_Position"] = ((close - bb_lower) / bb_range).clip(0, 1)

    # ── Phase 5B: Macro features ────────────────────────────────────────────

    # USD/INR — dollar strength (weak rupee helps IT/Pharma, hurts Energy/FMCG)
    if usdinr_close is not None and not usdinr_close.empty:
        usd = usdinr_close.reindex(df.index).ffill().bfill()
        df["USDINR_Return"]  = usd.pct_change().fillna(0)
        df["USDINR_20d_Mom"] = usd.pct_change(20).fillna(0)
    else:
        df["USDINR_Return"]  = 0.0
        df["USDINR_20d_Mom"] = 0.0

    # Crude oil (Brent) — affects Energy, Paints, Airlines, Cement
    if crude_close is not None and not crude_close.empty:
        crude = crude_close.reindex(df.index).ffill().bfill()
        df["Crude_Return"]   = crude.pct_change().fillna(0)
        df["Crude_20d_Mom"]  = crude.pct_change(20).fillna(0)
    else:
        df["Crude_Return"]   = 0.0
        df["Crude_20d_Mom"]  = 0.0

    # Seasonal — cyclical month encoding + Indian market events
    month = df.index.month
    df["Month_Sin"]       = np.sin(2 * np.pi * month / 12)
    df["Month_Cos"]       = np.cos(2 * np.pi * month / 12)
    df["Is_Budget_Month"] = (month == 2).astype(int)           # Union Budget
    df["Is_Monsoon"]      = month.isin([6, 7, 8, 9]).astype(int) # Monsoon


    # ── Phase 6: Global Macro + Sector-Conditional Features ─────────────────
    def _sr(s, idx, p=1):
        if s is None or (hasattr(s,'empty') and s.empty):
            return pd.Series(0.0, index=idx)
        return s.reindex(idx).ffill().bfill().pct_change(p).fillna(0)

    def _sl(s, idx, default=0.0):
        if s is None or (hasattr(s,'empty') and s.empty):
            return pd.Series(default, index=idx)
        return s.reindex(idx).ffill().bfill().fillna(default)

    idx = df.index

    # S&P 500 — global risk sentiment
    sp500_r              = _sr(sp500_close, idx)
    df["SP500_Return"]   = sp500_r
    df["SP500_5d"]       = _sr(sp500_close, idx, p=5)

    # FII proxy: Nifty return − S&P return  (negative = FII selling India)
    nifty_r_aligned      = nifty_return.reindex(idx).fillna(0)
    df["FII_Proxy"]      = nifty_r_aligned - sp500_r

    # US VIX
    df["VIX_US_Level"]   = _sl(vix_us_close, idx) / 100

    # India VIX enhancements
    if vix_in_close is not None and not vix_in_close.empty:
        vix_in            = vix_in_close.reindex(idx).ffill().bfill()
        df["VIX_IN_ROC5"] = vix_in.pct_change(5).fillna(0)
        df["VIX_IN_Pct"]  = vix_in.rolling(252, min_periods=30).rank(pct=True).fillna(0.5)
    else:
        df["VIX_IN_ROC5"] = 0.0
        df["VIX_IN_Pct"]  = 0.5

    # US 10Y Treasury
    if us10y_close is not None and not us10y_close.empty:
        us10y             = us10y_close.reindex(idx).ffill().bfill()
        df["US10Y_Level"] = us10y / 100
        df["US10Y_Chg"]   = us10y.diff().fillna(0)
    else:
        df["US10Y_Level"] = 0.04
        df["US10Y_Chg"]   = 0.0

    # Copper + Shanghai
    copper_r              = _sr(copper_close, idx)
    df["Copper_Return"]   = copper_r
    df["Shanghai_Return"] = _sr(shanghai_close, idx)
    nasdaq_r              = _sr(nasdaq_close, idx)

    # ── Sector-conditional features ──────────────────────────────────────────
    is_it      = int(sector_code == 2)
    is_pharma  = int(sector_code == 7)
    is_energy  = int(sector_code == 4)
    is_fmcg    = int(sector_code == 5)
    is_metals  = int(sector_code == 8)
    is_infra   = int(sector_code == 6)
    is_banking = int(sector_code in [0, 1])
    is_export  = int(sector_code in [2, 7])     # IT + Pharma

    df["NASDAQ_IT"]       = nasdaq_r                * is_it
    df["USD_Export"]      = df["USDINR_Return"]     * is_export
    df["Crude_Sector"]    = df["Crude_Return"]      * is_energy
    df["Copper_Sector"]   = copper_r                * is_metals
    df["Shanghai_Sector"] = df["Shanghai_Return"]   * int(is_metals or is_infra)
    df["Yield_Banking"]   = df["US10Y_Chg"]         * is_banking
    df["Monsoon_FMCG"]    = df["Is_Monsoon"]        * is_fmcg

    # Stock identity (for global model)
    df["Ticker"] = ticker_code
    df["Sector"] = sector_code

    # ── Target: 5-day forward return > 0.5% ──
    df["Return_5d"] = close.pct_change(RETURN_DAYS).shift(-RETURN_DAYS)
    df["Target"]    = (df["Return_5d"] > RETURN_MIN).astype(int)

    df.dropna(inplace=True)
    return df

# ── Per-stock model path ─────────────────────────────────────────────────────────
def stock_model_path(ticker: str) -> str:
    safe = ticker.replace(".", "_").replace("^", "")
    return os.path.join(MODELS_DIR, f"{safe}.pkl")

# ── Main ────────────────────────────────────────────────────────────────────────
def train():
    stocks = load_stocks()

    # Download Nifty 50 once for market context
    print("⬇  Downloading Nifty 50 (market context) …")
    nifty_raw    = yf.download("^NSEI", period=PERIOD, interval="1d",
                                auto_adjust=True, progress=False)
    nifty_close  = nifty_raw["Close"].squeeze()
    nifty_ma200  = nifty_close.rolling(200).mean()
    nifty_return = nifty_close.pct_change()
    print(f"   ✅ Nifty: {len(nifty_raw)} rows")

    # ── Download USD/INR ────────────────────────────────────────────────────
    print("⬇  Downloading USD/INR (USDINR=X) …")
    try:
        usdinr_raw   = yf.download("USDINR=X", period=PERIOD, interval="1d",
                                    auto_adjust=True, progress=False)
        usdinr_close = usdinr_raw["Close"].squeeze() if not usdinr_raw.empty else None
        print(f"   ✅ USD/INR: {len(usdinr_raw)} rows")
    except Exception as e:
        print(f"   ⚠  USD/INR download failed: {e}")
        usdinr_close = None

    # ── Download Brent Crude ─────────────────────────────────────────────────
    print("⬇  Downloading Brent Crude (BZ=F) …")
    try:
        crude_raw   = yf.download("BZ=F", period=PERIOD, interval="1d",
                                   auto_adjust=True, progress=False)
        crude_close = crude_raw["Close"].squeeze() if not crude_raw.empty else None
        print(f"   ✅ Crude: {len(crude_raw)} rows\n")
    except Exception as e:
        print(f"   ⚠  Crude download failed: {e}")
        crude_close = None

    # ── Download Phase 6 global data (parallel) ─────────────────────────────
    print("⬇  Downloading global macro data (S&P, NASDAQ, VIX, Yields, Copper, Shanghai)…")
    import concurrent.futures as _cf

    _SYMS = {
        "sp500":    "^GSPC",
        "nasdaq":   "^IXIC",
        "vix_us":   "^VIX",
        "vix_in":   "^INDIAVIX",
        "us10y":    "^TNX",
        "copper":   "HG=F",
        "shanghai": "000001.SS",
    }

    def _dl_sym(sym):
        try:
            df = yf.download(sym, period=PERIOD, interval="1d",
                             auto_adjust=True, progress=False)
            return df["Close"].squeeze() if not df.empty else None
        except Exception:
            return None

    with _cf.ThreadPoolExecutor(max_workers=7) as _ex:
        _futs = {k: _ex.submit(_dl_sym, v) for k, v in _SYMS.items()}
        _macro = {k: f.result(timeout=30) for k, f in _futs.items()}

    sp500_close    = _macro["sp500"]
    nasdaq_close   = _macro["nasdaq"]
    vix_us_close   = _macro["vix_us"]
    vix_in_close   = _macro["vix_in"]
    us10y_close    = _macro["us10y"]
    copper_close   = _macro["copper"]
    shanghai_close = _macro["shanghai"]

    _ok = [k for k, v in _macro.items() if v is not None]
    _fail = [k for k, v in _macro.items() if v is None]
    print(f"   ✅ Downloaded: {', '.join(_ok)}")
    if _fail: print(f"   ⚠  Failed (will use 0 fallback): {', '.join(_fail)}")
    print()

    # Label encoder
    le = LabelEncoder()
    le.fit(stocks)

    all_frames = []
    failed     = []

    for s in stocks:
        print(f"⬇  Downloading {s} …")
        raw = None
        for _attempt in range(3):   # retry up to 3 times for rate-limit 404s
            try:
                raw = yf.download(s, period=PERIOD, interval="1d",
                                  auto_adjust=True, progress=False)
                if not raw.empty:
                    break
            except Exception:
                pass
            import time as _time; _time.sleep(2)  # 2s back-off between retries
        if raw is None or raw.empty or len(raw) < 100:
            print(f"   ⚠  Insufficient data for {s} after retries, skipping.")
            failed.append(s)
            continue

        ticker_code = int(le.transform([s])[0])
        sector_code = SECTOR_MAP.get(s, 7)   # 7 = Other

        feat_df = build_features(
            raw, ticker_code, sector_code,
            nifty_close, nifty_ma200, nifty_return,
            usdinr_close   = usdinr_close,
            crude_close    = crude_close,
            sp500_close    = sp500_close,
            nasdaq_close   = nasdaq_close,
            vix_us_close   = vix_us_close,
            vix_in_close   = vix_in_close,
            us10y_close    = us10y_close,
            copper_close   = copper_close,
            shanghai_close = shanghai_close,
        )
        if len(feat_df) < 100:
            print(f"   ⚠  Too few rows after feature build for {s}, skipping.")
            failed.append(s)
            continue

        print(f"   ✅ {s}: {len(feat_df)} rows | Sector {sector_code}")
        all_frames.append(feat_df)

        # ── Per-stock model ──────────────────────────────────────────────────────
        X_s = feat_df[STOCK_FEATURES]
        y_s = feat_df["Target"]
        pm  = XGBClassifier(n_estimators=150, max_depth=4,
                            learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric="logloss", verbosity=0)
        pm.fit(X_s, y_s)
        joblib.dump(pm, stock_model_path(s))
        print(f"   💾 Per-stock model saved for {s}")

    # Clean failed stocks from registry
    if failed:
        stocks = [s for s in stocks if s not in failed]
        save_stocks(stocks)
        le = LabelEncoder(); le.fit(stocks)

    if not all_frames:
        print("❌ No data — aborting."); sys.exit(1)

    df_all = pd.concat(all_frames, ignore_index=True)
    print(f"\n✅ Global dataset: {len(df_all):,} rows | {len(stocks)} stocks\n")

    X = df_all[GLOBAL_FEATURES]
    y = df_all["Target"]

    # ── Global model CV ──────────────────────────────────────────────────────────
    tscv      = TimeSeriesSplit(n_splits=N_SPLITS)
    cv_scores = []
    print(f"Running {N_SPLITS}-fold TimeSeriesSplit CV on global model …")
    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        m = XGBClassifier(n_estimators=150, max_depth=4,
                          learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric="logloss", verbosity=0)
        m.fit(X.iloc[tr], y.iloc[tr])
        score = accuracy_score(y.iloc[te], m.predict(X.iloc[te]))
        cv_scores.append(score)
        print(f"   Fold {fold}: {score:.4f}")

    print(f"\n📊 Global CV: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # ── Global model (final) ─────────────────────────────────────────────────────
    print("\n🏋  Training final global model …")
    final = XGBClassifier(n_estimators=150, max_depth=4,
                          learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric="logloss", verbosity=0)
    final.fit(X, y)

    joblib.dump(final, MODEL_PATH)
    joblib.dump(le,    LE_PATH)

    metadata = {
        "trained_at":       datetime.now().isoformat(),
        "stocks":           stocks,
        "n_stocks":         len(stocks),
        "stock_features":   STOCK_FEATURES,
        "global_features":  GLOBAL_FEATURES,
        "period":           PERIOD,
        "n_samples":        int(len(df_all)),
        "cv_folds":         N_SPLITS,
        "cv_accuracy_mean": float(np.mean(cv_scores)),
        "cv_accuracy_std":  float(np.std(cv_scores)),
        "buy_threshold":    BUY_THRESH,
        "sell_threshold":   SELL_THRESH,
        "return_days":      RETURN_DAYS,
        "return_min_pct":   RETURN_MIN * 100,
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n💾 Saved global model + {len(stocks)} per-stock models to {DATA_DIR}")
    print(f"🎯 Done — CV {np.mean(cv_scores):.2%} | Target: {RETURN_DAYS}d return > {RETURN_MIN*100}%")

    # ── Build leader-lagger matrix ────────────────────────────────────────────
    print("\n📊 Building leader-lagger matrix…")
    try:
        import sys as _sys
        _sys.path.insert(0, ROOT_DIR)
        from engines.leader_lagger import build_leader_matrix
        build_leader_matrix(stocks, period="2y")
    except Exception as _e:
        print(f"⚠  Leader matrix build skipped: {_e}")

    return metadata

if __name__ == "__main__":
    train()
