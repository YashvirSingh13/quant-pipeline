"""
server/app.py — Upgraded FastAPI backend v4.

What's new:
  • Per-stock model loading (falls back to global)
  • NEUTRAL signal (0.35 < prob < 0.65 = too uncertain)
  • All new features computed in _live_features (ATR, Volume Spike, 52W,
    Market Context from Nifty, Earnings Season, Sector)
  • Label encoder fallback rebuilt from metadata if pkl missing
  • Per-stock model cache cleared on retrain
"""

import os
import sys
import json
import subprocess
import threading

import numpy as np
import joblib
import yfinance as yf
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(BASE_DIR)
DATA_DIR   = os.environ.get("DATA_DIR", os.path.join(ROOT_DIR, "data"))
MODELS_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

STOCKS_FILE = os.path.join(DATA_DIR, "known_stocks.json")
MODEL_PATH  = os.path.join(DATA_DIR, "model.pkl")
LE_PATH     = os.path.join(DATA_DIR, "label_encoder.pkl")
META_PATH   = os.path.join(DATA_DIR, "model_metadata.json")
PUBLIC_DIR  = os.path.join(ROOT_DIR,  "public")

SEED_STOCKS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]

# ── Sector map (same as train.py) ────────────────────────────────────────────────
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

# ── Sector benchmarks (approximate NSE averages) ────────────────────────────────
# Used to contextualise individual stock PE/PB vs sector
SECTOR_BENCHMARKS = {
    0: {"name": "Banking & Finance",    "pe": 18.0, "pb": 2.5,  "div_yield": 1.2},
    1: {"name": "Information Technology","pe": 28.0, "pb": 7.0,  "div_yield": 2.0},
    2: {"name": "Automobile",           "pe": 22.0, "pb": 3.5,  "div_yield": 0.8},
    3: {"name": "Energy & Oil",         "pe": 12.0, "pb": 1.8,  "div_yield": 3.5},
    4: {"name": "FMCG",                 "pe": 45.0, "pb": 12.0, "div_yield": 1.5},
    5: {"name": "Infrastructure",       "pe": 25.0, "pb": 3.0,  "div_yield": 1.0},
    6: {"name": "Pharmaceuticals",      "pe": 32.0, "pb": 5.0,  "div_yield": 0.5},
    7: {"name": "Others",               "pe": 25.0, "pb": 3.5,  "div_yield": 1.0},
}

def _compute_fundamentals(ticker: str, feats: dict) -> dict:
    """
    Fetch fundamental data from yfinance and compute a scorecard.
    Returns scorecard ratings, key metrics, and red flags.
    All values are best-effort — missing data is handled gracefully.
    """
    sector_code = SECTOR_MAP.get(ticker, 7)
    bench       = SECTOR_BENCHMARKS[sector_code]

    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}

    # ── Raw values ──────────────────────────────────────────────────────────────
    pe            = info.get("trailingPE") or info.get("forwardPE")
    pb            = info.get("priceToBook")
    div_yield_raw = info.get("dividendYield") or 0.0
    div_yield     = round(div_yield_raw * 100, 2)
    roe           = round((info.get("returnOnEquity")  or 0) * 100, 2)
    roa           = round((info.get("returnOnAssets")  or 0) * 100, 2)
    profit_margin = round((info.get("profitMargins")   or 0) * 100, 2)
    rev_growth    = round((info.get("revenueGrowth")   or 0) * 100, 2)
    earn_growth   = round((info.get("earningsGrowth")  or 0) * 100, 2)
    debt_equity   = info.get("debtToEquity") or 0.0
    current_ratio = info.get("currentRatio") or 0.0
    mkt_cap       = info.get("marketCap")
    beta          = info.get("beta")

    # ── Scorecard: Valuation ────────────────────────────────────────────────────
    if pe and bench["pe"]:
        ratio = pe / bench["pe"]
        if ratio < 0.8:   valuation, val_note = "Low",  "Trading at a discount to sector"
        elif ratio > 1.3: valuation, val_note = "High", "Overvalued vs sector average"
        else:             valuation, val_note = "Avg",  "Fairly valued vs sector"
    else:
        valuation, val_note = "N/A", "PE data unavailable"

    # ── Scorecard: Growth ───────────────────────────────────────────────────────
    growth_avg = (rev_growth + earn_growth) / 2 if (rev_growth or earn_growth) else 0
    if growth_avg > 15:   growth, g_note = "High", "Strong revenue & earnings growth"
    elif growth_avg > 5:  growth, g_note = "Avg",  "Moderate growth, in line with market"
    elif growth_avg >= 0: growth, g_note = "Low",  "Lagging behind market in growth"
    else:                 growth, g_note = "Low",  "Declining revenue or earnings"

    # ── Scorecard: Profitability ────────────────────────────────────────────────
    if roe > 15 and profit_margin > 10:
        profitability, p_note = "High", "Good profitability & efficiency"
    elif roe > 8 or profit_margin > 5:
        profitability, p_note = "Avg",  "Average profitability metrics"
    else:
        profitability, p_note = "Low",  "Below-average profitability"

    # ── Scorecard: Entry Point (uses technical features) ───────────────────────
    rsi       = feats.get("RSI", 50)
    hi52w_pct = feats.get("High52W_Pct", 0.95)
    if rsi < 45 and hi52w_pct < 0.88:
        entry, e_note = "Good",       "Underpriced, not in overbought zone"
    elif rsi > 68 or hi52w_pct > 0.97:
        entry, e_note = "Overbought", "Near 52W high or RSI elevated"
    else:
        entry, e_note = "Neutral",    "Neither cheap nor overbought"

    # ── Scorecard: Performance (1Y vs Nifty) ───────────────────────────────────
    try:
        stock_hist = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        nifty_hist = yf.download("^NSEI", period="1y", progress=False, auto_adjust=True)
        stock_ret  = float((stock_hist["Close"].iloc[-1] / stock_hist["Close"].iloc[0] - 1) * 100)
        nifty_ret  = float((nifty_hist["Close"].iloc[-1] / nifty_hist["Close"].iloc[0] - 1) * 100)
        diff       = stock_ret - nifty_ret
        if diff > 5:    performance, perf_note = "Good", f"Outperforming Nifty by {diff:.1f}%"
        elif diff < -5: performance, perf_note = "Low",  f"Underperforming Nifty by {abs(diff):.1f}%"
        else:           performance, perf_note = "Avg",  "In line with Nifty 50 returns"
        stock_1y_return = round(stock_ret, 2)
        nifty_1y_return = round(nifty_ret, 2)
    except Exception:
        performance, perf_note = "N/A", "Return data unavailable"
        stock_1y_return = nifty_1y_return = None

    # ── Red Flags ───────────────────────────────────────────────────────────────
    red_flags = []
    if debt_equity > 150:
        red_flags.append(f"High debt-to-equity ratio ({debt_equity:.0f}%)")
    if profit_margin < 0:
        red_flags.append("Negative profit margins — company running at a loss")
    if rev_growth < -5:
        red_flags.append(f"Declining revenue ({rev_growth:.1f}% YoY)")
    if earn_growth < -20:
        red_flags.append(f"Sharply falling earnings ({earn_growth:.1f}% YoY)")
    if current_ratio and current_ratio < 1.0:
        red_flags.append(f"Low current ratio ({current_ratio:.2f}) — liquidity risk")
    if pe and pe > bench["pe"] * 2:
        red_flags.append(f"PE ({pe:.1f}x) is more than 2x sector average ({bench['pe']}x)")

    rf_level = "High" if len(red_flags) >= 3 else "Medium" if len(red_flags) >= 1 else "Low"

    return {
        "scorecard": {
            "performance":    {"rating": performance, "note": perf_note},
            "valuation":      {"rating": valuation,   "note": val_note},
            "growth":         {"rating": growth,       "note": g_note},
            "profitability":  {"rating": profitability,"note": p_note},
            "entry_point":    {"rating": entry,        "note": e_note},
        },
        "key_metrics": {
            "pe_ratio":         round(pe, 2) if pe else None,
            "pb_ratio":         round(pb, 2) if pb else None,
            "div_yield":        div_yield,
            "roe":              roe,
            "roa":              roa,
            "profit_margin":    profit_margin,
            "revenue_growth":   rev_growth,
            "earnings_growth":  earn_growth,
            "debt_equity":      round(debt_equity, 1),
            "current_ratio":    round(current_ratio, 2) if current_ratio else None,
            "beta":             round(beta, 2) if beta else None,
            "stock_1y_return":  stock_1y_return,
            "nifty_1y_return":  nifty_1y_return,
        },
        "sector": {
            "name":      bench["name"],
            "pe":        bench["pe"],
            "pb":        bench["pb"],
            "div_yield": bench["div_yield"],
        },
        "red_flags":       red_flags,
        "red_flag_level":  rf_level,
    }

# ── Feature lists (must match train.py exactly) ──────────────────────────────────
STOCK_FEATURES = [
    "RSI", "MA50", "MA200", "MA_Cross", "Volatility",
    "MACD", "MACD_Signal", "MACD_Hist", "BB_Width",
    "Volume_Log", "Volume_Spike", "ATR",
    "High52W_Pct", "Low52W_Pct",
    "Market_Return", "Market_Regime", "Earnings_Season",
]
GLOBAL_FEATURES = STOCK_FEATURES + ["Ticker", "Sector"]

BUY_THRESH  = 0.65
SELL_THRESH = 0.35

# ── App ─────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Quant Pipeline API", version="4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── State ────────────────────────────────────────────────────────────────────────
_global_model   = None
_label_encoder  = None
_metadata: dict = {}
_stock_models   = {}    # ticker → per-stock XGBoost model (cached in memory)
_retraining     = False
_learning       = False
_retrain_timer  = None
DEBOUNCE_SECS   = 30

# ── Helpers: stock registry ──────────────────────────────────────────────────────
def _load_known_stocks() -> list:
    if os.path.exists(STOCKS_FILE):
        with open(STOCKS_FILE) as f: return json.load(f)
    _save_known_stocks(SEED_STOCKS); return SEED_STOCKS.copy()

def _save_known_stocks(stocks: list):
    with open(STOCKS_FILE, "w") as f: json.dump(sorted(set(stocks)), f, indent=2)

def _register_stock(ticker: str) -> bool:
    stocks = _load_known_stocks()
    if ticker in stocks: return False
    stocks.append(ticker); _save_known_stocks(stocks)
    print(f"📌 Registered new stock: {ticker} (total {len(stocks)})")
    return True

# ── Helpers: per-stock model path ───────────────────────────────────────────────
def _stock_model_path(ticker: str) -> str:
    safe = ticker.replace(".", "_").replace("^", "")
    return os.path.join(MODELS_DIR, f"{safe}.pkl")

def _get_model_for_ticker(ticker: str):
    """Return (model, feature_list, model_type) for a ticker."""
    # Try cached per-stock model
    if ticker in _stock_models:
        return _stock_models[ticker], STOCK_FEATURES, "per-stock"
    # Try loading from disk
    path = _stock_model_path(ticker)
    if os.path.exists(path):
        m = joblib.load(path)
        _stock_models[ticker] = m
        return m, STOCK_FEATURES, "per-stock"
    # Fall back to global
    if _global_model is not None:
        return _global_model, GLOBAL_FEATURES, "global"
    raise RuntimeError("No model available. Training may still be in progress.")

# ── Helpers: artefact loading ───────────────────────────────────────────────────
def _reload_artefacts():
    global _global_model, _label_encoder, _metadata, _stock_models
    _stock_models = {}  # clear per-stock cache so fresh models load

    if os.path.exists(MODEL_PATH):
        _global_model = joblib.load(MODEL_PATH)
        print("✅ Global model loaded")
    else:
        print("⚠  Global model not found")

    if os.path.exists(LE_PATH):
        _label_encoder = joblib.load(LE_PATH)
        print("✅ Label encoder loaded")
    else:
        print("⚠  Label encoder not found — will rebuild from metadata")

    if os.path.exists(META_PATH):
        with open(META_PATH) as f: _metadata = json.load(f)

    # Fallback: rebuild label encoder from metadata stocks list
    if _label_encoder is None and _metadata.get("stocks"):
        le = LabelEncoder()
        le.fit(sorted(_metadata["stocks"]))
        _label_encoder = le
        print(f"✅ Label encoder rebuilt from metadata ({len(_metadata['stocks'])} stocks)")

# ── Helpers: training runner ─────────────────────────────────────────────────────
def _run_training():
    train_script = os.path.join(ROOT_DIR, "ml", "train.py")
    env = {**os.environ, "DATA_DIR": DATA_DIR}
    subprocess.run([sys.executable, train_script],
                   check=True, cwd=ROOT_DIR, env=env)
    _reload_artefacts()

# ── Debounced auto-retrain ───────────────────────────────────────────────────────
def _schedule_auto_retrain():
    global _retrain_timer
    if _retrain_timer: _retrain_timer.cancel()
    _retrain_timer = threading.Timer(DEBOUNCE_SECS, _do_auto_retrain)
    _retrain_timer.daemon = True
    _retrain_timer.start()
    print(f"⏱  Auto-retrain in {DEBOUNCE_SECS}s …")

def _do_auto_retrain():
    global _learning, _retrain_timer
    if _retraining:
        _retrain_timer = threading.Timer(60, _do_auto_retrain)
        _retrain_timer.daemon = True
        _retrain_timer.start(); return
    _learning = True
    try:
        print("🧠 Auto-retrain (new stock) started …")
        _run_training()
        print("🧠 Auto-retrain complete ✅")
    except Exception as exc:
        print(f"❌ Auto-retrain failed: {exc}")
    finally:
        _learning = False; _retrain_timer = None

# ── Startup ──────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    if not os.path.exists(STOCKS_FILE):
        _save_known_stocks(SEED_STOCKS)
    if not os.path.exists(MODEL_PATH):
        print("🚀 First boot — training (~4 min with all improvements) …")
        try: _run_training()
        except Exception as exc: print(f"❌ Initial training failed: {exc}")
    else:
        _reload_artefacts()

# ── Indicator helpers (mirrors train.py) ─────────────────────────────────────────
def _rsi(s, p=14):
    d = s.diff(); g = d.where(d>0,0).rolling(p).mean(); l = (-d.where(d<0,0)).rolling(p).mean()
    return 100 - (100 / (1 + g/l))

def _macd(s, fast=12, slow=26, sig=9):
    ml = s.ewm(span=fast,adjust=False).mean() - s.ewm(span=slow,adjust=False).mean()
    return ml, ml.ewm(span=sig,adjust=False).mean()

def _bb_width(s, p=20):
    sma=s.rolling(p).mean(); std=s.rolling(p).std()
    return (sma+2*std-(sma-2*std))/sma

def _atr(high, low, close, p=14):
    tr = pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(p).mean()

def _earnings_season(dt) -> int:
    m, d = dt.month, dt.day
    return int(
        ((m==4 and d>=15) or m==5) or ((m==7 and d>=15) or m==8) or
        ((m==10 and d>=15) or m==11) or ((m==1 and d>=15) or m==2)
    )

# ── Live feature computation ──────────────────────────────────────────────────────
def _live_features(ticker: str) -> dict:
    # Need ~14 months for MA200 + 52W features
    df = yf.download(ticker, period="14mo", interval="1d",
                     auto_adjust=True, progress=False)
    if df.empty or len(df) < 60:
        raise ValueError(f"No data found for '{ticker}'.")

    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    vol   = df["Volume"].squeeze()

    # Nifty for market context (last 14 months to match)
    nifty = yf.download("^NSEI", period="14mo", interval="1d",
                         auto_adjust=True, progress=False)
    nifty_close  = nifty["Close"].squeeze()
    nifty_ma200  = nifty_close.rolling(200).mean()
    nifty_return = nifty_close.pct_change()

    ma50  = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1])
    ml, sl = _macd(close)

    # Market context (align to last date)
    last_date     = df.index[-1]
    mkt_ret_idx   = nifty_return.index.get_indexer([last_date], method="nearest")[0]
    mkt_ret       = float(nifty_return.iloc[mkt_ret_idx]) if mkt_ret_idx >= 0 else 0.0
    mkt_c         = float(nifty_close.iloc[mkt_ret_idx]) if mkt_ret_idx >= 0 else 0.0
    mkt_ma        = float(nifty_ma200.iloc[mkt_ret_idx]) if mkt_ret_idx >= 0 else 0.0

    # Ticker & Sector codes
    ticker_code = 0
    if _label_encoder is not None and ticker in _label_encoder.classes_:
        ticker_code = int(_label_encoder.transform([ticker])[0])
    sector_code = SECTOR_MAP.get(ticker, 7)

    feats = {
        # Core
        "RSI":            float(_rsi(close).iloc[-1]),
        "MA50":           ma50,
        "MA200":          ma200,
        "MA_Cross":       ma50 - ma200,
        "Volatility":     float(close.pct_change().rolling(10).std().iloc[-1]),
        "MACD":           float(ml.iloc[-1]),
        "MACD_Signal":    float(sl.iloc[-1]),
        "MACD_Hist":      float((ml - sl).iloc[-1]),
        "BB_Width":       float(_bb_width(close).iloc[-1]),
        "Volume_Log":     float(np.log1p(vol.iloc[-1])),
        # New
        "Volume_Spike":   float(vol.iloc[-1] / vol.rolling(20).mean().iloc[-1]),
        "ATR":            float(_atr(high, low, close).iloc[-1]),
        "High52W_Pct":    float(close.iloc[-1] / close.rolling(252).max().iloc[-1]),
        "Low52W_Pct":     float(close.iloc[-1] / close.rolling(252).min().iloc[-1]),
        "Market_Return":  mkt_ret,
        "Market_Regime":  int(mkt_c > mkt_ma),
        "Earnings_Season":_earnings_season(last_date.to_pydatetime()),
        # Identity
        "Ticker":         ticker_code,
        "Sector":         sector_code,
        # UI extras
        "_last_price":    float(close.iloc[-1]),
        "_as_of":         str(df.index[-1].date()),
    }
    return feats

# ── Prediction logic ─────────────────────────────────────────────────────────────
def _run_predict(ticker: str, feats: dict) -> dict:
    model, feature_list, model_type = _get_model_for_ticker(ticker)
    vals = [feats[f] for f in feature_list]
    prob = float(model.predict_proba([vals])[0][1])

    # Three-way signal
    if   prob > BUY_THRESH:  signal = "BUY"
    elif prob < SELL_THRESH: signal = "SELL"
    else:                    signal = "NEUTRAL"

    return {
        "signal":       signal,
        "probability":  round(prob * 100, 2),
        "buy_threshold":  BUY_THRESH,
        "sell_threshold": SELL_THRESH,
        "model_type":   model_type,
        "trained_at":   _metadata.get("trained_at", "unknown"),
        "cv_accuracy":  _metadata.get("cv_accuracy_mean"),
        "n_stocks":     _metadata.get("n_stocks", len(_metadata.get("stocks", []))),
    }

# ── Schemas ──────────────────────────────────────────────────────────────────────
class PredictBody(BaseModel):
    RSI:float; MA50:float; MA200:float; MA_Cross:float; Volatility:float
    MACD:float; MACD_Signal:float; MACD_Hist:float; BB_Width:float
    Volume_Log:float; Volume_Spike:float=1.0; ATR:float=0.0
    High52W_Pct:float=0.95; Low52W_Pct:float=1.05
    Market_Return:float=0.0; Market_Regime:int=1; Earnings_Season:int=0
    Ticker:int=0; Sector:int=7
    ticker:str = ""   # for model selection

# ── Routes ───────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok", "model_ready": _global_model is not None,
        "trained_at": _metadata.get("trained_at"),
        "n_stocks": len(_load_known_stocks()),
        "learning": _learning, "retraining": _retraining,
    }

@app.get("/stocks")
def list_stocks():
    stocks = _load_known_stocks()
    in_model = list(_label_encoder.classes_) if _label_encoder else []
    per_stock_trained = [
        f.replace("_NS.pkl","").replace("_BO.pkl","").replace("_",".")
        for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")
    ]
    return {"registered": stocks, "in_model": in_model,
            "per_stock_trained": per_stock_trained, "total": len(stocks)}

@app.post("/predict")
def predict_manual(body: PredictBody):
    feats = body.dict()
    ticker = feats.pop("ticker", "") or "UNKNOWN"
    try:
        result = _run_predict(ticker, feats)
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@app.get("/live")
def predict_live(ticker: str):
    try:
        feats = _live_features(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    try:
        result = _run_predict(ticker, feats)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    is_new = _register_stock(ticker)
    if is_new:
        _schedule_auto_retrain()

    result["features"]     = {k: v for k, v in feats.items() if not k.startswith("_")}
    result["last_price"]   = feats["_last_price"]
    result["as_of"]        = feats["_as_of"]
    result["new_stock"]    = is_new
    result["learning"]     = is_new or _learning
    result["total_stocks"] = len(_load_known_stocks())

    # Fundamental scorecard — fetched alongside technical signal
    try:
        result["fundamentals"] = _compute_fundamentals(ticker, feats)
    except Exception as exc:
        print(f"⚠  Fundamentals fetch failed for {ticker}: {exc}")
        result["fundamentals"] = None

    return result

@app.get("/metadata")
def get_metadata():
    if not _metadata:
        raise HTTPException(status_code=404, detail="No metadata yet.")
    return {**_metadata, "registered_stocks": _load_known_stocks()}

@app.get("/learning")
def learning_status():
    return {"learning": _learning, "retraining": _retraining,
            "n_stocks": len(_load_known_stocks())}

@app.post("/retrain")
def manual_retrain(background_tasks: BackgroundTasks):
    global _retraining
    if _retraining or _learning:
        raise HTTPException(status_code=409, detail="Training already in progress.")
    def _run():
        global _retraining
        _retraining = True
        try: _run_training()
        except Exception as exc: print(f"❌ Retrain failed: {exc}")
        finally: _retraining = False
    background_tasks.add_task(_run)
    return {"status": "retraining started"}

@app.get("/retrain/status")
def retrain_status():
    return {"retraining": _retraining, "learning": _learning}

if os.path.exists(PUBLIC_DIR):
    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="static")
