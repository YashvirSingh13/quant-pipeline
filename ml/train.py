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
SEED_STOCKS = [
    # Banking & Finance
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS",
    "BAJFINANCE.NS", "BAJAJFINSV.NS",
    # Information Technology
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
    # Energy
    "RELIANCE.NS", "ONGC.NS", "BPCL.NS",
    # FMCG
    "HINDUNILVR.NS", "ITC.NS", "BRITANNIA.NS",
    # Automobile
    "TATAMOTORS.NS", "MARUTI.NS", "BAJAJ-AUTO.NS",
    # Infrastructure & Others
    "LT.NS", "NTPC.NS", "SUNPHARMA.NS", "ASIANPAINT.NS",
]
PERIOD       = "10y"
N_SPLITS     = 5       # for global model CV
BUY_THRESH   = 0.65
SELL_THRESH  = 0.35
RETURN_DAYS  = 5       # predict 5-day forward return
RETURN_MIN   = 0.005   # must be > 0.5% to count as BUY

# ── Sector map ──────────────────────────────────────────────────────────────────
# 0=Banking/Finance, 1=IT, 2=Auto, 3=Energy, 4=FMCG, 5=Infra, 6=Pharma, 7=Other
SECTOR_MAP = {
    "HDFCBANK.NS":0,  "SBIN.NS":0,      "AXISBANK.NS":0,  "ICICIBANK.NS":0,
    "KOTAKBANK.NS":0, "BAJFINANCE.NS":0, "INDUSINDBK.NS":0,"BANDHANBNK.NS":0,
    "TCS.NS":1,       "INFY.NS":1,       "WIPRO.NS":1,     "HCLTECH.NS":1,
    "TECHM.NS":1,     "MPHASIS.NS":1,    "LTIM.NS":1,
    "TATAMOTORS.NS":2,"MARUTI.NS":2,     "BAJAJ-AUTO.NS":2,"HEROMOTOCO.NS":2,
    "EICHERMOT.NS":2, "M&M.NS":2,
    "RELIANCE.NS":3,  "ONGC.NS":3,       "COALINDIA.NS":3, "BPCL.NS":3,
    "IOC.NS":3,       "GAIL.NS":3,
    "HINDUNILVR.NS":4,"BRITANNIA.NS":4,  "NESTLEIND.NS":4, "ITC.NS":4,
    "DABUR.NS":4,     "MARICO.NS":4,
    "ADANIPORTS.NS":5,"NTPC.NS":5,       "POWERGRID.NS":5, "LT.NS":5,
    "SUNPHARMA.NS":6, "DRREDDY.NS":6,    "CIPLA.NS":6,     "DIVISLAB.NS":6,
    "ASIANPAINT.NS":7,"PIDILITIND.NS":7, "TITAN.NS":7,     "ULTRACEMCO.NS":7,
}

# ── Feature column definitions ──────────────────────────────────────────────────
# Per-stock model uses these (no Ticker/Sector — they'd be constant per stock)
STOCK_FEATURES = [
    "RSI", "MA50", "MA200", "MA_Cross", "Volatility",
    "MACD", "MACD_Signal", "MACD_Hist", "BB_Width",
    "Volume_Log", "Volume_Spike", "ATR",
    "High52W_Pct", "Low52W_Pct",
    "Market_Return", "Market_Regime", "Earnings_Season",
]
# Global fallback model adds stock-identity features
GLOBAL_FEATURES = STOCK_FEATURES + ["Ticker", "Sector"]

# ── Stock registry helpers ────────────────────────────────────────────────────────
def load_stocks() -> list:
    if os.path.exists(STOCKS_FILE):
        with open(STOCKS_FILE) as f:
            stocks = json.load(f)
        print(f"📋 Registry: {len(stocks)} stocks — {', '.join(stocks)}")
        return stocks
    print(f"📋 No registry — seeding with {len(SEED_STOCKS)} stocks")
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
                   nifty_return: pd.Series) -> pd.DataFrame:
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
    df["Market_Return"] = nifty_return.reindex(df.index).fillna(0)
    df["Market_Regime"] = (
        nifty_close.reindex(df.index) > nifty_ma200.reindex(df.index)
    ).astype(int).fillna(0)

    # Earnings season
    df["Earnings_Season"] = is_earnings_season(df.index)

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
    print(f"   ✅ Nifty: {len(nifty_raw)} rows\n")

    # Label encoder
    le = LabelEncoder()
    le.fit(stocks)

    all_frames = []
    failed     = []

    for s in stocks:
        print(f"⬇  Downloading {s} …")
        raw = yf.download(s, period=PERIOD, interval="1d",
                          auto_adjust=True, progress=False)
        if raw.empty or len(raw) < 300:
            print(f"   ⚠  Insufficient data for {s}, skipping.")
            failed.append(s)
            continue

        ticker_code = int(le.transform([s])[0])
        sector_code = SECTOR_MAP.get(s, 7)   # 7 = Other

        feat_df = build_features(
            raw, ticker_code, sector_code,
            nifty_close, nifty_ma200, nifty_return
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
    return metadata

if __name__ == "__main__":
    train()
