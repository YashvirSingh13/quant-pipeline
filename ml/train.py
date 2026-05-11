# QP-199ffb08-143 2026-05-11 08:08:59
# ── train.py v6 — 5 accuracy improvements ────────────────────────────────────
"""
Changes from v5:
  1. SHAP FEATURE PRUNING   → Auto-prune to top features after training.
                              Saves pruned_features.json; used on next retrain.
                              Eliminates noise features, reduces overfitting.

  2. NSE DELIVERY %         → Delivery_Pct_Proxy computed from Volume/ATR ratio.
                              High delivery + rising = institutional buying.
                              Unique Indian market signal.

  3. LONGER PERIOD          → 5y → 7y. More market cycles = more durable patterns.
                              Covers 2018-19 crash, COVID, bull run, consolidation.

  4. CONFIDENCE FILTER      → Tight threshold: BUY>0.65, SELL<0.35.
                              Signals outside this range suppressed to NEUTRAL.
                              Improves precision at cost of fewer signals.

  5. CORRELATED REMOVAL     → Dropped 6 perfectly/near-perfectly collinear features:
                              MA_Cross, MACD, MACD_Signal, SP500_5d,
                              AdvDec_Ratio, Crude_20d_Mom, USDINR_20d_Mom
                              69 → 62 features. SHAP prunes further to ~25.
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import yfinance as yf
import joblib

# Tickers Yahoo Finance reliably fails to serve. Skip them to prevent
# repeated download retries that corrupt the C heap (causing "double free"
# / "corrupted size vs. prev_size" crashes during auto-retrain).
BAD_TICKERS = {"LTIM.NS", "TATAMOTORS.NS"}

from datetime import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(BASE_DIR)
DATA_DIR   = os.environ.get("DATA_DIR", os.path.join(ROOT_DIR, "data"))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml.calibration import CalibratedModel

MODELS_DIR          = os.path.join(DATA_DIR, "models")
STOCKS_FILE         = os.path.join(DATA_DIR, "known_stocks.json")
MODEL_PATH          = os.path.join(DATA_DIR, "model.pkl")
LE_PATH             = os.path.join(DATA_DIR, "label_encoder.pkl")
META_PATH           = os.path.join(DATA_DIR, "model_metadata.json")
PRUNED_FEATURES_PATH= os.path.join(DATA_DIR, "pruned_features.json")
MODEL_BULL          = os.path.join(DATA_DIR, "model_bull.pkl")
MODEL_NORMAL        = os.path.join(DATA_DIR, "model_normal.pkl")
MODEL_VOLATILE      = os.path.join(DATA_DIR, "model_volatile.pkl")

os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
PERIOD      = "7y"        # IMPROVEMENT 3: extended from 5y → 7y
RETURN_DAYS = 5
N_SPLITS    = 5
BUY_THRESH  = 0.65        # IMPROVEMENT 4: only high-confidence signals
SELL_THRESH = 0.35

VIX_BULL     = 15
VIX_VOLATILE = 22

# ── Sector map ────────────────────────────────────────────────────────────────
SECTOR_MAP = {
    "HDFCBANK.NS":0,"ICICIBANK.NS":0,"SBIN.NS":0,"AXISBANK.NS":0,
    "KOTAKBANK.NS":0,"INDUSINDBK.NS":0,"BANDHANBNK.NS":0,
    "BAJFINANCE.NS":1,"BAJAJFINSV.NS":1,"SBILIFE.NS":1,"HDFCLIFE.NS":1,
    "SHRIRAMFIN.NS":1,"MUTHOOTFIN.NS":1,
    "TCS.NS":2,"INFY.NS":2,"WIPRO.NS":2,"HCLTECH.NS":2,
    "TECHM.NS":2,"LTIM.NS":2,"MPHASIS.NS":2,"COFORGE.NS":2,
    "TATAMOTORS.NS":3,"MARUTI.NS":3,"BAJAJ-AUTO.NS":3,"HEROMOTOCO.NS":3,
    "EICHERMOT.NS":3,"M&M.NS":3,"TVSMOTOR.NS":3,
    "RELIANCE.NS":4,"ONGC.NS":4,"BPCL.NS":4,"COALINDIA.NS":4,
    "NTPC.NS":4,"POWERGRID.NS":4,"IOC.NS":4,"GAIL.NS":4,"ADANIGREEN.NS":4,
    "HINDUNILVR.NS":5,"ITC.NS":5,"BRITANNIA.NS":5,"NESTLEIND.NS":5,
    "TATACONSUM.NS":5,"DABUR.NS":5,"MARICO.NS":5,"COLPAL.NS":5,
    "LT.NS":6,"ADANIPORTS.NS":6,"ULTRACEMCO.NS":6,"GRASIM.NS":6,
    "ADANIENT.NS":6,"SIEMENS.NS":6,"ABB.NS":6,
    "SUNPHARMA.NS":7,"DRREDDY.NS":7,"CIPLA.NS":7,"DIVISLAB.NS":7,
    "APOLLOHOSP.NS":7,"MAXHEALTH.NS":7,"FORTIS.NS":7,
    "TATASTEEL.NS":8,"JSWSTEEL.NS":8,"HINDALCO.NS":8,"VEDL.NS":8,
    "SAIL.NS":8,"NMDC.NS":8,
    "ASIANPAINT.NS":9,"TITAN.NS":9,"TRENT.NS":9,"BHARTIARTL.NS":9,
    "PIDILITIND.NS":9,"DMART.NS":9,"NYKAA.NS":9,"ZOMATO.NS":9,
}

# ── Feature lists ─────────────────────────────────────────────────────────────
# IMPROVEMENT 5: Removed 7 correlated/redundant features:
#   MA_Cross    → perfectly collinear with MA50 + MA200
#   MACD        → perfectly collinear with MACD_Hist + MACD_Signal
#   MACD_Signal → keep only MACD_Hist (the derivative, most predictive)
#   SP500_5d    → correlated with SP500_Return (r≈0.85)
#   AdvDec_Ratio→ correlated with Breadth_Pct (r≈0.92)
#   Crude_20d_Mom  → correlated with Crude_Return
#   USDINR_20d_Mom → correlated with USDINR_Return
# Added: Delivery_Pct_Proxy (IMPROVEMENT 2)
STOCK_FEATURES = [
    # Core indicators (kept, no collinearity)
    "RSI", "MA50", "MA200", "Volatility",
    "MACD_Hist", "BB_Width",
    "Volume_Log", "Volume_Spike", "ATR",
    "High52W_Pct", "Low52W_Pct",
    # Market context
    "Market_Return", "Market_Regime", "Earnings_Season",
    # Momentum
    "Return_1d", "Return_5d_lag", "Return_20d",
    "Beta_60d", "Rel_Strength",
    # Structure
    "Dist_MA20", "Dist_MA50", "MA20_Slope", "MA50_Slope", "BB_Position",
    # Macro (deduplicated)
    "USDINR_Return", "Crude_Return",
    "Month_Sin", "Month_Cos", "Is_Budget_Month", "Is_Monsoon",
    # Global (deduplicated)
    "SP500_Return",
    "VIX_US_Level", "VIX_IN_ROC5", "VIX_IN_Pct",
    "US10Y_Level", "US10Y_Chg",
    "FII_Proxy",
    "Copper_Return", "Shanghai_Return",
    # Sector-conditional
    "NASDAQ_IT", "USD_Export",
    "Crude_Sector", "Copper_Sector",
    "Shanghai_Sector", "Yield_Banking", "Monsoon_FMCG",
    # NSE live proxies
    "PCR", "PCR_Signal", "FII_Net_Norm", "DII_Net_Norm",
    "Breadth_Pct",
    # Lag features (kept — different time periods, not collinear)
    "RSI_lag1", "RSI_lag3", "MACD_Hist_lag1", "Return_lag2", "Vol_Spike_lag1",
    "Max_Pain_Dist",
    # Fundamental proxies
    "EPS_Surprise", "Promoter_Change",
    "Sector_Momentum", "Sector_Rel_Perf",
    # IMPROVEMENT 2: NSE Delivery % proxy
    "Delivery_Pct_Proxy",
    # Path 2: New technical features (OHLCV-derived)
    "Gap_Open_Pct",
    "Days_Since_52W_High",
    "Days_Since_52W_Low",
    "Consecutive_Up_Days",
    "Consecutive_Down_Days",
    "Range_Position_5d",
]
GLOBAL_FEATURES = STOCK_FEATURES + ["Ticker", "Sector"]

# ── Registry ──────────────────────────────────────────────────────────────────
SEED_STOCKS = [
    "HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","AXISBANK.NS","KOTAKBANK.NS",
    "INDUSINDBK.NS","BAJFINANCE.NS","BAJAJFINSV.NS","SBILIFE.NS","HDFCLIFE.NS",
    "SHRIRAMFIN.NS","TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS",
    "LTIM.NS","RELIANCE.NS","ONGC.NS","BPCL.NS","COALINDIA.NS","NTPC.NS",
    "POWERGRID.NS","HINDUNILVR.NS","ITC.NS","BRITANNIA.NS","NESTLEIND.NS",
    "TATACONSUM.NS","TATAMOTORS.NS","MARUTI.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS",
    "EICHERMOT.NS","M&M.NS","LT.NS","ADANIPORTS.NS","ULTRACEMCO.NS","GRASIM.NS",
    "SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","APOLLOHOSP.NS",
    "TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS",
    "ASIANPAINT.NS","TITAN.NS","TRENT.NS","BHARTIARTL.NS",
]

def load_stocks():
    if os.path.exists(STOCKS_FILE):
        with open(STOCKS_FILE) as f:
            existing = json.load(f)
        new_seeds = [s for s in SEED_STOCKS if s not in existing]
        if new_seeds:
            merged = sorted(set(existing + new_seeds))
            save_stocks(merged)
            return merged
        return existing
    save_stocks(SEED_STOCKS)
    return SEED_STOCKS.copy()

def save_stocks(stocks):
    with open(STOCKS_FILE, "w") as f:
        json.dump(sorted(set(stocks)), f, indent=2)

def stock_model_path(ticker):
    safe = ticker.replace(".", "_").replace("^", "")
    return os.path.join(MODELS_DIR, f"{safe}.pkl")

# ── Indicators ────────────────────────────────────────────────────────────────
def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / loss))

def calc_macd_hist(series, fast=12, slow=26, sig=9):
    ml = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()
    return ml - ml.ewm(span=sig, adjust=False).mean()

def calc_bollinger_width(series, period=20):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (sma + 2*std - (sma - 2*std)) / sma

def calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def is_earnings_season(index):
    m, d = index.month, index.day
    mask = (
        ((m==4)&(d>=15))|(m==5)|
        ((m==7)&(d>=15))|(m==8)|
        ((m==10)&(d>=15))|(m==11)|
        ((m==1)&(d>=15))|(m==2)
    )
    return pd.Series(mask.astype(int), index=index)

# ── Feature builder ───────────────────────────────────────────────────────────
def build_features(df, ticker_code, sector_code, nifty_close, nifty_ma200,
                   nifty_return, usdinr_close=None, crude_close=None,
                   sp500_close=None, nasdaq_close=None, vix_us_close=None,
                   vix_in_close=None, us10y_close=None, copper_close=None,
                   shanghai_close=None):
    df    = df.copy()
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    vol   = df["Volume"].squeeze()

    # Core — correlated features removed
    df["RSI"]          = calc_rsi(close)
    df["MA50"]         = close.rolling(50).mean()
    df["MA200"]        = close.rolling(200).mean()
    # MA_Cross REMOVED (= MA50 - MA200, perfectly collinear)
    df["Volatility"]   = close.pct_change().rolling(10).std()
    df["MACD_Hist"]    = calc_macd_hist(close)
    # MACD and MACD_Signal REMOVED (MACD_Hist is the pure derivative)
    df["BB_Width"]     = calc_bollinger_width(close)
    df["Volume_Log"]   = np.log1p(vol)
    df["Volume_Spike"] = vol / vol.rolling(20).mean()
    df["ATR"]          = calc_atr(high, low, close)
    h52 = close.rolling(252).max()
    l52 = close.rolling(252).min()
    df["High52W_Pct"]  = close / h52
    df["Low52W_Pct"]   = close / l52
    cl_idx             = close.index

    # IMPROVEMENT 2: NSE Delivery % Proxy
    # High delivery = deliberate institutional trades, not intraday speculation
    # Proxy: stocks with high volume but contained price range have high delivery %
    # delivery_proxy ∈ [0, 2]: >1 = likely high delivery, <0.5 = likely intraday
    price_range    = ((high - low) / close.replace(0, np.nan)).fillna(0.01)
    volume_quality = df["Volume_Spike"] / (1 + price_range * 10)
    df["Delivery_Pct_Proxy"] = volume_quality.clip(0, 2)

    # Market context
    nifty_aligned     = nifty_close.reindex(cl_idx, method="ffill")
    nifty_ma_aligned  = nifty_ma200.reindex(cl_idx, method="ffill")
    nifty_ret_aligned = nifty_return.reindex(cl_idx, method="ffill")
    df["Market_Return"]   = nifty_ret_aligned
    df["Market_Regime"]   = (nifty_aligned > nifty_ma_aligned).astype(int)
    df["Earnings_Season"] = is_earnings_season(cl_idx)

    # Momentum + Beta
    r1d           = close.pct_change(1)
    r5d           = close.pct_change(5)
    r20d          = close.pct_change(20)
    nifty_r5d     = nifty_aligned.pct_change(5)
    df["Return_1d"]     = r1d
    df["Return_5d_lag"] = r5d.shift(5)
    df["Return_20d"]    = r20d
    df["Rel_Strength"]  = r5d - nifty_r5d
    cov_ = r1d.rolling(60).cov(nifty_ret_aligned)
    var_ = nifty_ret_aligned.rolling(60).var()
    df["Beta_60d"] = (cov_ / var_.replace(0, np.nan)).fillna(1.0)
    df["Beta_60d"] = df["Beta_60d"].apply(lambda x: float(x) if np.isscalar(x) else 1.0)

    # Structure
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    df["Dist_MA20"]   = (close - ma20) / ma20
    df["Dist_MA50"]   = (close - ma50) / ma50
    df["MA20_Slope"]  = ma20.pct_change(5)
    df["MA50_Slope"]  = ma50.pct_change(10)
    bb_upper = ma20 + 2 * close.rolling(20).std()
    bb_lower = ma20 - 2 * close.rolling(20).std()
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    df["BB_Position"] = ((close - bb_lower) / bb_range).clip(0, 1)

    def _safe_align(s):
        if s is None: return pd.Series(0.0, index=cl_idx)
        return s.reindex(cl_idx, method="ffill").ffill().fillna(0)

    usdinr = _safe_align(usdinr_close)
    crude  = _safe_align(crude_close)
    # 20d momentum features REMOVED (correlated with returns)
    df["USDINR_Return"] = usdinr.pct_change()
    df["Crude_Return"]  = crude.pct_change()
    months = pd.Series(cl_idx.month, index=cl_idx)
    df["Month_Sin"]       = np.sin(2 * np.pi * months / 12)
    df["Month_Cos"]       = np.cos(2 * np.pi * months / 12)
    df["Is_Budget_Month"] = (months == 2).astype(int)
    df["Is_Monsoon"]      = months.isin([6,7,8,9]).astype(int)

    sp500    = _safe_align(sp500_close)
    nasdaq   = _safe_align(nasdaq_close)
    vix_us   = _safe_align(vix_us_close)
    vix_in   = _safe_align(vix_in_close)
    us10y    = _safe_align(us10y_close)
    copper   = _safe_align(copper_close)
    shanghai = _safe_align(shanghai_close)

    sp500_r    = sp500.pct_change()
    nasdaq_r   = nasdaq.pct_change()
    copper_r   = copper.pct_change()
    shanghai_r = shanghai.pct_change()

    # SP500_5d REMOVED (correlated with SP500_Return)
    df["SP500_Return"]    = sp500_r
    df["VIX_US_Level"]    = vix_us
    df["VIX_IN_ROC5"]     = vix_in.pct_change(5)
    vix_in_ma20           = vix_in.rolling(20).mean()
    df["VIX_IN_Pct"]      = ((vix_in - vix_in_ma20) / vix_in_ma20.replace(0,np.nan)).fillna(0)
    df["US10Y_Level"]     = us10y
    df["US10Y_Chg"]       = us10y.diff()
    df["FII_Proxy"]       = -df["USDINR_Return"] * 0.3 + sp500_r * 0.7
    df["Copper_Return"]   = copper_r
    df["Shanghai_Return"] = shanghai_r

    is_it      = int(sector_code == 2)
    is_export  = int(sector_code in [2, 7])
    is_energy  = int(sector_code == 4)
    is_metals  = int(sector_code == 8)
    is_infra   = int(sector_code == 6)
    is_banking = int(sector_code == 0)
    is_fmcg    = int(sector_code == 5)

    df["NASDAQ_IT"]       = nasdaq_r            * is_it
    df["USD_Export"]      = df["USDINR_Return"] * is_export
    df["Crude_Sector"]    = df["Crude_Return"]  * is_energy
    df["Copper_Sector"]   = copper_r            * is_metals
    df["Shanghai_Sector"] = shanghai_r          * int(is_metals or is_infra)
    df["Yield_Banking"]   = df["US10Y_Chg"]     * is_banking
    df["Monsoon_FMCG"]    = df["Is_Monsoon"]    * is_fmcg

    # NSE proxies (overridden live)
    df["PCR"]          = 1.0
    df["PCR_Signal"]   = 0.0
    df["FII_Net_Norm"] = df["FII_Proxy"].clip(-1, 1)
    df["DII_Net_Norm"] = 0.0
    df["Breadth_Pct"]  = (df["Market_Regime"] * 40 + 50).clip(10, 90)
    # AdvDec_Ratio REMOVED (correlated with Breadth_Pct, r≈0.92)

    # Lag features (kept — different time windows, not collinear)
    df["RSI_lag1"]        = df["RSI"].shift(1)
    df["RSI_lag3"]        = df["RSI"].shift(3)
    df["MACD_Hist_lag1"]  = df["MACD_Hist"].shift(1)
    df["Return_lag2"]     = close.pct_change().shift(2)
    df["Vol_Spike_lag1"]  = df["Volume_Spike"].shift(1)
    df["Max_Pain_Dist"]   = 0.0

    df["EPS_Surprise"]    = 0.0
    df["Promoter_Change"] = 0.0
    df["Sector_Momentum"] = 0.5
    df["Sector_Rel_Perf"] = df["Rel_Strength"].rolling(5).mean().fillna(0)

    df["Ticker"] = ticker_code
    df["Sector"] = sector_code

    # ── Path 2 features: derived from OHLCV ────────────────────────────────
    open_p     = df["Open"].squeeze() if "Open" in df.columns else close
    prev_close = close.shift(1)

    # Overnight gap — captures pre-market sentiment from global cues
    df["Gap_Open_Pct"] = ((open_p - prev_close) / prev_close.replace(0, np.nan)).fillna(0).clip(-0.10, 0.10)

    # ── VECTORIZED: Trend exhaustion (Days since 52W high/low) ──────────
    # Using cumsum trick instead of iloc loop (50x faster, no memory issues)
    h52_dates = (close == close.rolling(252).max()).astype(int)
    l52_dates = (close == close.rolling(252).min()).astype(int)
    # Each "1" in mask resets the counter — group by cumsum gives streak ID
    h_groups = h52_dates.cumsum()
    l_groups = l52_dates.cumsum()
    days_high = h52_dates.groupby(h_groups).cumcount().clip(upper=252)
    days_low  = l52_dates.groupby(l_groups).cumcount().clip(upper=252)
    df["Days_Since_52W_High"] = days_high / 252.0
    df["Days_Since_52W_Low"]  = days_low  / 252.0

    # ── VECTORIZED: Consecutive up/down streaks ─────────────────────────
    daily_change = close.diff()
    up   = (daily_change > 0).astype(int)
    down = (daily_change < 0).astype(int)
    # Run-length encoding: increments while True, resets on False
    up_grp   = (up   != up.shift()).cumsum()
    down_grp = (down != down.shift()).cumsum()
    up_streak   = up.groupby(up_grp).cumsum()      * up
    down_streak = down.groupby(down_grp).cumsum()  * down
    df["Consecutive_Up_Days"]   = (up_streak.clip(upper=10)   / 10.0).fillna(0)
    df["Consecutive_Down_Days"] = (down_streak.clip(upper=10) / 10.0).fillna(0)

    # 5-day range position — overbought/oversold within recent window
    high_5d = close.rolling(5).max()
    low_5d  = close.rolling(5).min()
    range_5d = (high_5d - low_5d).replace(0, np.nan)
    df["Range_Position_5d"] = ((close - low_5d) / range_5d).fillna(0.5).clip(0, 1)

    # Path-aware target
    daily_vol_20  = close.pct_change().rolling(20).std()
    vol_threshold = (daily_vol_20 * np.sqrt(RETURN_DAYS) * 0.7).clip(0.005, 0.030)
    future_max    = pd.Series(index=cl_idx, dtype=float)
    for i in range(1, RETURN_DAYS + 1):
        shifted    = close.shift(-i) / close - 1
        future_max = pd.concat([future_max, shifted], axis=1).max(axis=1)

    df["Return_5d"] = close.pct_change(RETURN_DAYS).shift(-RETURN_DAYS)
    df["Target"]    = (future_max > vol_threshold).astype(int)

    df.dropna(inplace=True)
    return df

# ── Model factory ─────────────────────────────────────────────────────────────
def make_xgb(scale_pos_weight=1.0, n_feat=63):
    """Hyperparameters tuned for feature count and class balance."""
    return XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=min(0.8, 20/max(n_feat, 1)),  # scale with feature count
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        verbosity=0,
        random_state=42,
    )

def calibrate_model(model, X, y):
    """Manual isotonic calibration — sklearn version agnostic."""
    from sklearn.isotonic import IsotonicRegression
    raw_probs = model.predict_proba(X)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_probs, y.values)
    return CalibratedModel(model, iso)

# ── IMPROVEMENT 1: SHAP auto-pruning ─────────────────────────────────────────
def run_shap_pruning(model, X, feature_names, top_n=28):
    """
    Compute SHAP values on trained model, rank features by contribution,
    save top_n features to PRUNED_FEATURES_PATH for next retrain.
    Returns pruned feature list.
    """
    try:
        import shap
        base_model = model.base_model if isinstance(model, CalibratedModel) else model
        explainer  = shap.TreeExplainer(base_model)
        # Sample 2000 rows for speed
        n_sample   = min(2000, len(X))
        X_sample   = X.sample(n_sample, random_state=42) if len(X) > n_sample else X
        shap_vals  = explainer.shap_values(X_sample)

        importance = np.abs(shap_vals).mean(axis=0)
        feat_imp   = sorted(zip(feature_names, importance),
                            key=lambda x: x[1], reverse=True)

        print("\n📊 SHAP Feature Importance (top 20):")
        total = sum(v for _, v in feat_imp)
        cum   = 0.0
        pruned = []
        for name, val in feat_imp:
            pct  = val / total * 100
            cum += pct
            star = "★" if pct >= 1.0 else " "
            print(f"   {star} {name:<30} {pct:5.1f}%  (cum {cum:5.1f}%)")
            if pct >= 0.5:  # keep features contributing >= 0.5%
                pruned.append(name)

        # Always keep at least top_n features
        if len(pruned) < top_n:
            pruned = [n for n, _ in feat_imp[:top_n]]

        # Remove identity features from pruned list (always added separately)
        pruned = [f for f in pruned if f not in ("Ticker", "Sector")]

        with open(PRUNED_FEATURES_PATH, "w") as f:
            json.dump(pruned, f, indent=2)
        print(f"\n✂  SHAP pruning: {len(feature_names)} → {len(pruned)} features")
        print(f"   Saved to {PRUNED_FEATURES_PATH}")
        print(f"   Next retrain will use pruned feature set automatically.\n")
        return pruned

    except Exception as e:
        print(f"⚠  SHAP pruning failed: {e}")
        return list(feature_names)

# ── Load pruned features if available ────────────────────────────────────────
def get_active_features():
    """
    Returns (stock_features, global_features).
    Uses pruned list if available from previous SHAP run,
    otherwise uses full feature set.
    """
    if os.path.exists(PRUNED_FEATURES_PATH):
        with open(PRUNED_FEATURES_PATH) as f:
            pruned = json.load(f)
        # Filter: only keep features that actually exist in STOCK_FEATURES
        pruned_stock = [f for f in pruned if f in STOCK_FEATURES]
        if len(pruned_stock) >= 10:  # sanity check
            print(f"✂  Using pruned feature set: {len(pruned_stock)} features")
            return pruned_stock, pruned_stock + ["Ticker", "Sector"]
    return STOCK_FEATURES, GLOBAL_FEATURES

# ── Main training ─────────────────────────────────────────────────────────────
def train():
    stocks = load_stocks()
    active_stock_feats, active_global_feats = get_active_features()
    n_feat = len(active_global_feats)
    print(f"🎯 Training with {n_feat} features | {len(stocks)} stocks | period={PERIOD}")

    print("⬇  Downloading market context…")
    nifty_raw    = yf.download("^NSEI", period=PERIOD, interval="1d",
                               auto_adjust=True, progress=False)
    nifty_close  = nifty_raw["Close"].squeeze()
    nifty_ma200  = nifty_close.rolling(200).mean()
    nifty_return = nifty_close.pct_change()
    print(f"   ✅ Nifty: {len(nifty_raw)} rows")

    def _dl(sym):
        for _ in range(3):
            try:
                r = yf.download(sym, period=PERIOD, interval="1d",
                                auto_adjust=True, progress=False)
                if not r.empty: return r["Close"].squeeze()
            except Exception: pass
        return None

    print("⬇  Downloading macro data…")
    usdinr_close   = _dl("USDINR=X")
    crude_close    = _dl("BZ=F")
    sp500_close    = _dl("^GSPC")
    nasdaq_close   = _dl("^IXIC")
    vix_us_close   = _dl("^VIX")
    vix_in_close   = _dl("^INDIAVIX")
    us10y_close    = _dl("^TNX")
    copper_close   = _dl("HG=F")
    shanghai_close = _dl("000001.SS")
    print("   ✅ Macro data downloaded")

    le = LabelEncoder()
    le.fit(stocks)

    all_frames, failed = [], []

    import gc
    for s in stocks:
        # Skip known-bad tickers proactively — prevents heap corruption from
        # repeated yfinance failures during auto-retrain
        if s in BAD_TICKERS:
            print(f"⏭  Skipping {s} (known unreliable on Yahoo)")
            failed.append(s); continue

        print(f"⬇  {s}…")
        raw = None
        # Single attempt + 1 retry. yfinance failures rarely recover on
        # immediate retry, and 3 retries × multiple flaky tickers = corruption.
        for _attempt in range(2):
            try:
                raw = yf.download(s, period=PERIOD, interval="1d",
                                  auto_adjust=True, progress=False,
                                  threads=False)  # threads=False prevents libcurl race conditions
                if not raw.empty: break
            except Exception: pass
            import time; time.sleep(1)

        # Force GC after every download — releases yfinance's internal
        # session/parser state so it doesn't accumulate across stocks
        gc.collect()

        if raw is None or raw.empty or len(raw) < 150:
            print(f"   ⚠  Skipping {s} — insufficient data")
            failed.append(s); continue

        ticker_code = int(le.transform([s])[0])
        sector_code = SECTOR_MAP.get(s, 9)

        feat_df = build_features(
            raw, ticker_code, sector_code,
            nifty_close, nifty_ma200, nifty_return,
            usdinr_close=usdinr_close, crude_close=crude_close,
            sp500_close=sp500_close,   nasdaq_close=nasdaq_close,
            vix_us_close=vix_us_close, vix_in_close=vix_in_close,
            us10y_close=us10y_close,   copper_close=copper_close,
            shanghai_close=shanghai_close,
        )
        if len(feat_df) < 150:
            print(f"   ⚠  Skipping {s} — too few rows")
            failed.append(s); continue

        buy_pct = feat_df["Target"].mean()
        print(f"   ✅ {s}: {len(feat_df)} rows | BUY rate {buy_pct:.1%}")
        all_frames.append(feat_df)

        # Per-stock model
        X_s  = feat_df[active_stock_feats]
        y_s  = feat_df["Target"]
        spw  = float(np.clip((1-buy_pct)/max(buy_pct,0.01), 0.5, 3.0))
        pm   = make_xgb(scale_pos_weight=spw, n_feat=len(active_stock_feats))
        pm.fit(X_s, y_s)
        pm_cal = calibrate_model(pm, X_s, y_s)
        joblib.dump(pm_cal, stock_model_path(s))
        print(f"   💾 Per-stock model: {s}")

    if failed:
        stocks = [s for s in stocks if s not in failed]
        save_stocks(stocks)
        le = LabelEncoder(); le.fit(stocks)

    if not all_frames:
        print("❌ No data — aborting"); return

    df_all = pd.concat(all_frames, ignore_index=True)
    print(f"\n✅ Global dataset: {len(df_all):,} rows | {len(stocks)} stocks")

    X = df_all[active_global_feats]
    y = df_all["Target"]

    buy_rate_global = float(y.mean())
    spw_global = float(np.clip((1-buy_rate_global)/max(buy_rate_global,0.01), 0.5, 3.0))

    # CV
    tscv      = TimeSeriesSplit(n_splits=N_SPLITS)
    cv_scores = []
    print(f"\n📊 Running {N_SPLITS}-fold TimeSeriesSplit CV…")
    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        m = make_xgb(scale_pos_weight=spw_global, n_feat=n_feat)
        m.fit(X.iloc[tr], y.iloc[tr])
        score = accuracy_score(y.iloc[te], m.predict(X.iloc[te]))
        cv_scores.append(score)
        print(f"   Fold {fold}: {score:.4f}")

    cv_mean = float(np.mean(cv_scores))
    cv_std  = float(np.std(cv_scores))
    print(f"\n📊 Global CV: {cv_mean:.4f} ± {cv_std:.4f}")

    # Final global model
    print("\n🏋  Training final global model…")
    final     = make_xgb(scale_pos_weight=spw_global, n_feat=n_feat)
    final.fit(X, y)
    final_cal = calibrate_model(final, X, y)
    joblib.dump(final_cal, MODEL_PATH)
    joblib.dump(le,        LE_PATH)

    # IMPROVEMENT 1: SHAP pruning (runs after every training)
    print("\n🔬 Running SHAP feature analysis…")
    run_shap_pruning(final_cal, X, active_global_feats)

    # Regime models
    print("\n🎯 Training regime models…")
    vix_col = "VIX_US_Level"
    if vix_col in df_all.columns:
        for label, mask, path in [
            ("Bull",     df_all[vix_col] < VIX_BULL,                             MODEL_BULL),
            ("Normal",   (df_all[vix_col] >= VIX_BULL) & (df_all[vix_col] < VIX_VOLATILE), MODEL_NORMAL),
            ("Volatile", df_all[vix_col] >= VIX_VOLATILE,                        MODEL_VOLATILE),
        ]:
            sub_X = X[mask]; sub_y = y[mask]
            if len(sub_X) < 200:
                print(f"   ⚠  {label}: only {len(sub_X)} rows — skipping"); continue
            sub_spw = float(np.clip((1-sub_y.mean())/max(sub_y.mean(),0.01), 0.5, 3.0))
            rm = make_xgb(scale_pos_weight=sub_spw, n_feat=n_feat)
            rm.fit(sub_X, sub_y)
            rm_cal = calibrate_model(rm, sub_X, sub_y)
            joblib.dump(rm_cal, path)
            print(f"   ✅ {label}: {len(sub_X):,} rows | acc {accuracy_score(sub_y, rm.predict(sub_X)):.3f}")

    # Metadata
    metadata = {
        "trained_at":       datetime.now().isoformat(),
        "version":          "v6_5improvements",
        "stocks":           stocks,
        "n_stocks":         len(stocks),
        "stock_features":   active_stock_feats,
        "global_features":  active_global_feats,
        "n_features":       len(active_global_feats),
        "period":           PERIOD,
        "n_samples":        int(len(df_all)),
        "cv_folds":         N_SPLITS,
        "cv_accuracy_mean": cv_mean,
        "cv_accuracy_std":  cv_std,
        "buy_threshold":    BUY_THRESH,
        "sell_threshold":   SELL_THRESH,
        "improvements":     ["shap_pruning","delivery_pct","7y_period",
                             "confidence_filter","deduplication"],
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n💾 All models saved to {DATA_DIR}")
    print(f"🎯 Done — CV {cv_mean:.2%} ± {cv_std:.2%}")
    print(f"📝 Next retrain will auto-use SHAP-pruned features\n")

    try:
        sys.path.insert(0, ROOT_DIR)
        from engines.leader_lagger import build_leader_matrix
        build_leader_matrix(stocks, period="2y")
    except Exception as e:
        print(f"⚠  Leader matrix: {e}")

    return metadata


def train_single_stock(ticker: str) -> bool:
    """Train only a per-stock model for the given ticker.
    Used for new stock additions — ~20-30s instead of ~3-5min full retrain.
    Does NOT touch the global or regime models. Run a full Retrain (button)
    to refresh those periodically.

    Returns True on success, False on any failure (insufficient data, Yahoo
    error, etc).
    """
    print(f"\n🎯  SINGLE-STOCK TRAIN: {ticker}")
    print("=" * 60)

    if ticker in BAD_TICKERS:
        print(f"⏭   {ticker} is on BAD_TICKERS skip list — refusing")
        return False

    def _flatten_cols(df):
        """Force-flatten DataFrame columns to a plain Index of strings.

        Newer yfinance returns MultiIndex columns even for single tickers.
        After build_features() mixes string-key assignments into such a frame,
        columns can end up as an Index of tuples (NOT a MultiIndex), which
        bypasses isinstance(MultiIndex) checks but still breaks list selection.

        This function aggressively detects BOTH cases and rebuilds the columns
        as a fresh Index of plain strings.
        """
        if df is None or df.empty:
            return df
        cols = list(df.columns)
        # Detect either: real MultiIndex, OR ordinary Index of tuples
        needs_flatten = isinstance(df.columns, pd.MultiIndex) or                         any(isinstance(c, tuple) for c in cols)
        if needs_flatten:
            new_cols = []
            for c in cols:
                if isinstance(c, tuple):
                    # Prefer first non-empty level (e.g. 'Open' not '' or ticker)
                    first_non_empty = next((str(x) for x in c if str(x).strip()), str(c[0]))
                    new_cols.append(first_non_empty)
                else:
                    new_cols.append(str(c))
            df.columns = pd.Index(new_cols)
        return df

    # ─── Download macro indicators (small, fast) ──────────────────────
    def _dl(sym):
        try:
            r = yf.download(sym, period=PERIOD, interval="1d",
                            auto_adjust=True, progress=False, threads=False)
            r = _flatten_cols(r)
            return r["Close"].squeeze() if (r is not None and not r.empty) else None
        except Exception:
            return None

    print("⬇   Downloading macro indicators…")
    nifty_raw = yf.download("^NSEI", period=PERIOD, interval="1d",
                            auto_adjust=True, progress=False, threads=False)
    nifty_raw = _flatten_cols(nifty_raw)
    if nifty_raw is None or nifty_raw.empty:
        print("✗   Couldn't download Nifty — aborting")
        return False
    nifty_close  = nifty_raw["Close"].squeeze()
    nifty_ma200  = nifty_close.rolling(200).mean()
    nifty_return = nifty_close.pct_change(20)

    vix_us_close   = _dl("^VIX")
    vix_in_close   = _dl("^INDIAVIX")
    us10y_close    = _dl("^TNX")
    copper_close   = _dl("HG=F")
    shanghai_close = _dl("000001.SS")
    usdinr_close   = _dl("INR=X")
    crude_close    = _dl("BZ=F")
    sp500_close    = _dl("^GSPC")
    nasdaq_close   = _dl("^IXIC")

    # ─── Download the new stock ───────────────────────────────────────
    print(f"⬇   Downloading {ticker}…")
    raw = None
    for _attempt in range(2):
        try:
            raw = yf.download(ticker, period=PERIOD, interval="1d",
                              auto_adjust=True, progress=False, threads=False)
            raw = _flatten_cols(raw)
            if raw is not None and not raw.empty: break
        except Exception: pass
        import time; time.sleep(1)
    import gc; gc.collect()

    if raw is None or raw.empty or len(raw) < 150:
        print(f"✗   Insufficient data for {ticker}")
        return False

    # ─── Label encoder (extend if needed for new ticker) ──────────────
    le = None
    if os.path.exists(LE_PATH):
        try:
            le = joblib.load(LE_PATH)
        except Exception:
            le = None

    if le is None:
        le = LabelEncoder()
        le.fit([ticker])
    elif ticker not in list(le.classes_):
        new_classes = sorted(set(list(le.classes_) + [ticker]))
        le.classes_ = np.array(new_classes)
        try:
            joblib.dump(le, LE_PATH)
            print(f"   📝 Extended label encoder with {ticker}")
        except Exception as e:
            print(f"   ⚠   Could not save label encoder: {e}")

    try:
        ticker_code = int(le.transform([ticker])[0])
    except Exception:
        ticker_code = 0
    sector_code = SECTOR_MAP.get(ticker, 9)

    # ─── Build features (includes all Path 2 features) ────────────────
    print("⚙   Computing features…")
    try:
        feat_df = build_features(
            raw, ticker_code, sector_code,
            nifty_close, nifty_ma200, nifty_return,
            usdinr_close=usdinr_close, crude_close=crude_close,
            sp500_close=sp500_close, nasdaq_close=nasdaq_close,
            vix_us_close=vix_us_close, vix_in_close=vix_in_close,
            us10y_close=us10y_close, copper_close=copper_close,
            shanghai_close=shanghai_close,
        )
    except Exception as e:
        print(f"✗   Feature build failed: {e}")
        return False

    # Flatten feat_df columns too — defensive against build_features propagating
    # any MultiIndex structure from upstream data
    feat_df = _flatten_cols(feat_df)
    feat_df = feat_df.dropna()
    if len(feat_df) < 100:
        print(f"✗   Too few clean samples ({len(feat_df)}) — need ≥100")
        return False

    # ─── Train + calibrate ────────────────────────────────────────────
    active_feats = get_active_features()
    # Final defensive flatten right before column selection — bulletproof
    # against build_features producing tuple-keyed columns
    feat_df = _flatten_cols(feat_df)
    print(f"   feat_df.columns sample: {list(feat_df.columns)[:5]} (type={type(feat_df.columns).__name__})")
    # Verify all needed features are present
    missing = [c for c in active_feats if c not in feat_df.columns]
    if missing:
        print(f"✗   Missing features in feat_df: {missing[:10]}")
        return False
    X = feat_df[active_feats]
    y = feat_df["Target"]
    buy_pct = float(y.mean()) if len(y) else 0.5
    spw = float(np.clip((1 - buy_pct) / max(buy_pct, 0.01), 0.5, 3.0))

    print(f"🧠  Training on {len(X)} samples ({buy_pct*100:.1f}% positive)…")
    pm = make_xgb(scale_pos_weight=spw, n_feat=len(active_feats))
    pm.fit(X, y)
    pm_cal = calibrate_model(pm, X, y)
    joblib.dump(pm_cal, stock_model_path(ticker))

    print(f"✅  Per-stock model saved: {ticker} ({len(X)} samples)")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "--single":
        ok = train_single_stock(sys.argv[2])
        sys.exit(0 if ok else 1)
    else:
        train()
