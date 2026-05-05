"""
ml/train.py — Self-learning training script.

Reads the stock list from  $DATA_DIR/known_stocks.json
Saves all artefacts to     $DATA_DIR/

DATA_DIR defaults to  <project_root>/data  (locally)
                  or  /data               (Railway Volume)

Seed stocks (used only on very first run when no JSON exists):
    RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

STOCKS_FILE = os.path.join(DATA_DIR, "known_stocks.json")
MODEL_PATH  = os.path.join(DATA_DIR, "model.pkl")
LE_PATH     = os.path.join(DATA_DIR, "label_encoder.pkl")
META_PATH   = os.path.join(DATA_DIR, "model_metadata.json")

# ── Config ──────────────────────────────────────────────────────────────────────
SEED_STOCKS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
PERIOD      = "10y"
N_SPLITS    = 5
THRESHOLD   = 0.6

FEATURES = [
    "RSI", "MA50", "MA200", "MA_Cross", "Volatility",
    "MACD", "MACD_Signal", "MACD_Hist", "BB_Width",
    "Volume_Log", "Ticker"
]

# ── Stock registry ───────────────────────────────────────────────────────────────
def load_stocks() -> list:
    """Load known stocks from registry, or create it from seed list."""
    if os.path.exists(STOCKS_FILE):
        with open(STOCKS_FILE) as f:
            stocks = json.load(f)
        print(f"📋 Loaded {len(stocks)} stocks from registry: {', '.join(stocks)}")
        return stocks
    print(f"📋 No registry found — seeding with {len(SEED_STOCKS)} stocks")
    save_stocks(SEED_STOCKS)
    return SEED_STOCKS.copy()

def save_stocks(stocks: list):
    with open(STOCKS_FILE, "w") as f:
        json.dump(sorted(set(stocks)), f, indent=2)

# ── Technical indicators ────────────────────────────────────────────────────────
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.where(delta > 0, 0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def calc_macd(series: pd.Series, fast=12, slow=26, signal_span=9):
    ema_fast  = series.ewm(span=fast,        adjust=False).mean()
    ema_slow  = series.ewm(span=slow,        adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line  = macd_line.ewm(span=signal_span, adjust=False).mean()
    return macd_line, sig_line

def calc_bollinger_width(series: pd.Series, period: int = 20) -> pd.Series:
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (sma + 2*std - (sma - 2*std)) / sma

def build_features(df: pd.DataFrame, ticker_code: int) -> pd.DataFrame:
    df    = df.copy()
    close = df["Close"].squeeze()
    vol   = df["Volume"].squeeze()

    df["RSI"]        = calc_rsi(close)
    df["MA50"]       = close.rolling(50).mean()
    df["MA200"]      = close.rolling(200).mean()
    df["MA_Cross"]   = df["MA50"] - df["MA200"]
    df["Volatility"] = close.pct_change().rolling(10).std()

    macd_line, macd_sig = calc_macd(close)
    df["MACD"]        = macd_line
    df["MACD_Signal"] = macd_sig
    df["MACD_Hist"]   = macd_line - macd_sig

    df["BB_Width"]   = calc_bollinger_width(close)
    df["Volume_Log"] = np.log1p(vol)
    df["Ticker"]     = ticker_code

    df["Return"] = close.pct_change().shift(-1)
    df["Target"] = (df["Return"] > 0).astype(int)
    df.dropna(inplace=True)
    return df

# ── Main training routine ───────────────────────────────────────────────────────
def train():
    stocks = load_stocks()

    # Fit label encoder on full registry
    le = LabelEncoder()
    le.fit(stocks)

    # Download & build features
    all_frames = []
    failed     = []
    for s in stocks:
        print(f"⬇  Downloading {s} ({PERIOD}) …")
        raw = yf.download(s, period=PERIOD, interval="1d",
                          auto_adjust=True, progress=False)
        if raw.empty or len(raw) < 250:
            print(f"   ⚠  Insufficient data for {s}, skipping.")
            failed.append(s)
            continue
        code = int(le.transform([s])[0])
        all_frames.append(build_features(raw, code))
        print(f"   ✅ {s}: {len(raw)} rows")

    if not all_frames:
        print("❌ No data downloaded — aborting.")
        sys.exit(1)

    # Remove failed stocks from registry so they don't keep blocking
    if failed:
        good = [s for s in stocks if s not in failed]
        save_stocks(good)
        le = LabelEncoder()
        le.fit(good)
        stocks = good

    df_all = pd.concat(all_frames, ignore_index=True)
    print(f"\n✅ Dataset: {len(df_all):,} rows across {len(stocks)} stocks\n")

    X = df_all[FEATURES]
    y = df_all["Target"]

    # TimeSeriesSplit CV
    tscv      = TimeSeriesSplit(n_splits=N_SPLITS)
    cv_scores = []
    print(f"Running {N_SPLITS}-fold TimeSeriesSplit …")
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
        m = XGBClassifier(n_estimators=100, max_depth=4,
                          eval_metric="logloss", verbosity=0)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        score = accuracy_score(y.iloc[te_idx], m.predict(X.iloc[te_idx]))
        cv_scores.append(score)
        print(f"   Fold {fold}: {score:.4f}")

    print(f"\n📊 CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # Final model on full data
    print("\n🏋  Training final model on full dataset …")
    final = XGBClassifier(n_estimators=100, max_depth=4,
                          eval_metric="logloss", verbosity=0)
    final.fit(X, y)

    # Save artefacts
    joblib.dump(final, MODEL_PATH)
    joblib.dump(le,    LE_PATH)

    metadata = {
        "trained_at":       datetime.now().isoformat(),
        "stocks":           stocks,
        "n_stocks":         len(stocks),
        "features":         FEATURES,
        "period":           PERIOD,
        "n_samples":        int(len(df_all)),
        "cv_folds":         N_SPLITS,
        "cv_accuracy_mean": float(np.mean(cv_scores)),
        "cv_accuracy_std":  float(np.std(cv_scores)),
        "threshold":        THRESHOLD,
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n💾 Saved to {DATA_DIR}")
    print(f"🎯 Done — {len(stocks)} stocks | CV {np.mean(cv_scores):.2%} accuracy")
    return metadata

if __name__ == "__main__":
    train()
