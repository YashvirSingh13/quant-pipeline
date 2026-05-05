import os
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

# ── Config ─────────────────────────────────────────────────────────────────────
STOCKS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
PERIOD  = "5y"
N_SPLITS = 5
THRESHOLD = 0.6
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# ── Technical Indicators ────────────────────────────────────────────────────────
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.where(delta > 0, 0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def calc_macd(series: pd.Series, fast=12, slow=26, signal_span=9):
    ema_fast    = series.ewm(span=fast,   adjust=False).mean()
    ema_slow    = series.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
    return macd_line, signal_line

def calc_bollinger_width(series: pd.Series, period: int = 20) -> pd.Series:
    sma   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (upper - lower) / sma           # normalised band width

def build_features(df: pd.DataFrame, ticker_code: int) -> pd.DataFrame:
    df    = df.copy()
    close = df["Close"].squeeze()
    vol   = df["Volume"].squeeze()

    df["RSI"]         = calc_rsi(close)
    df["MA50"]        = close.rolling(50).mean()
    df["MA200"]       = close.rolling(200).mean()
    df["MA_Cross"]    = df["MA50"] - df["MA200"]   # sign → trend direction
    df["Volatility"]  = close.pct_change().rolling(10).std()

    macd_line, macd_sig = calc_macd(close)
    df["MACD"]        = macd_line
    df["MACD_Signal"] = macd_sig
    df["MACD_Hist"]   = macd_line - macd_sig

    df["BB_Width"]    = calc_bollinger_width(close)

    # Log-normalise volume → removes cross-stock scale differences
    df["Volume_Log"]  = np.log1p(vol)

    df["Ticker"]      = ticker_code

    # Target: will next day's close be higher?
    df["Return"]  = close.pct_change().shift(-1)
    df["Target"]  = (df["Return"] > 0).astype(int)

    df.dropna(inplace=True)
    return df

# ── Feature columns (must match predict / server) ───────────────────────────────
FEATURES = [
    "RSI", "MA50", "MA200", "MA_Cross", "Volatility",
    "MACD", "MACD_Signal", "MACD_Hist", "BB_Width",
    "Volume_Log", "Ticker"
]

# ── Download & build dataset ────────────────────────────────────────────────────
le = LabelEncoder()
le.fit(STOCKS)

all_frames = []
for s in STOCKS:
    print(f"⬇  Downloading {s} …")
    raw = yf.download(s, period=PERIOD, interval="1d", auto_adjust=True, progress=False)
    if raw.empty:
        print(f"   ⚠ No data for {s}, skipping.")
        continue
    code = int(le.transform([s])[0])
    all_frames.append(build_features(raw, code))

df_all = pd.concat(all_frames, ignore_index=True)
print(f"\n✅ Dataset: {len(df_all):,} rows, {len(FEATURES)} features\n")

X = df_all[FEATURES]
y = df_all["Target"]

# ── Time-series cross-validation (no future leakage) ───────────────────────────
tscv      = TimeSeriesSplit(n_splits=N_SPLITS)
cv_scores = []

print(f"Running {N_SPLITS}-fold TimeSeriesSplit …")
for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    m = XGBClassifier(
        n_estimators=100, max_depth=4,
        eval_metric="logloss", verbosity=0
    )
    m.fit(X_tr, y_tr)
    score = accuracy_score(y_te, m.predict(X_te))
    cv_scores.append(score)
    print(f"   Fold {fold}: {score:.4f}")

print(f"\n📊 CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# ── Final model trained on full dataset ─────────────────────────────────────────
print("\n🏋  Training final model on full dataset …")
final_model = XGBClassifier(
    n_estimators=100, max_depth=4,
    eval_metric="logloss", verbosity=0
)
final_model.fit(X, y)

# ── Save artefacts ──────────────────────────────────────────────────────────────
model_path = os.path.join(ROOT_DIR, "model.pkl")
le_path    = os.path.join(ROOT_DIR, "label_encoder.pkl")
meta_path  = os.path.join(ROOT_DIR, "model_metadata.json")

joblib.dump(final_model, model_path)
joblib.dump(le, le_path)

metadata = {
    "trained_at":        datetime.now().isoformat(),
    "stocks":            STOCKS,
    "features":          FEATURES,
    "period":            PERIOD,
    "n_samples":         int(len(df_all)),
    "cv_folds":          N_SPLITS,
    "cv_accuracy_mean":  float(np.mean(cv_scores)),
    "cv_accuracy_std":   float(np.std(cv_scores)),
    "threshold":         THRESHOLD,
}
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n💾 Saved  model.pkl | label_encoder.pkl | model_metadata.json")
print(f"🎯 Done — CV mean accuracy {np.mean(cv_scores):.2%}")
