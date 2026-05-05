"""
server/app.py — Self-learning FastAPI backend.

New behaviour vs v2:
  • All artefacts stored in DATA_DIR (Railway Volume → persists restarts)
  • /live auto-registers new tickers → debounced background retrain
  • /stocks   → list all known stocks
  • /learning → is a background auto-retrain in progress?

Environment variables:
  DATA_DIR   path to persistent storage  (default: <root>/data)
  PORT       server port                 (set by Railway automatically)
"""

import os
import sys
import json
import subprocess
import threading

import numpy as np
import joblib
import yfinance as yf
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

DATA_DIR    = os.environ.get("DATA_DIR", os.path.join(ROOT_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

STOCKS_FILE = os.path.join(DATA_DIR, "known_stocks.json")
MODEL_PATH  = os.path.join(DATA_DIR, "model.pkl")
LE_PATH     = os.path.join(DATA_DIR, "label_encoder.pkl")
META_PATH   = os.path.join(DATA_DIR, "model_metadata.json")
PUBLIC_DIR  = os.path.join(ROOT_DIR, "public")

SEED_STOCKS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]

# ── App ─────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Quant Pipeline API", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── In-memory state ──────────────────────────────────────────────────────────────
_model          = None
_label_encoder  = None
_metadata: dict = {}
_retraining     = False   # manual retrain (user-triggered)
_learning       = False   # auto-retrain (new stock detected)
_retrain_timer  = None    # debounce timer

# ── Stock registry helpers ───────────────────────────────────────────────────────
def _load_known_stocks() -> list:
    if os.path.exists(STOCKS_FILE):
        with open(STOCKS_FILE) as f:
            return json.load(f)
    _save_known_stocks(SEED_STOCKS)
    return SEED_STOCKS.copy()

def _save_known_stocks(stocks: list):
    with open(STOCKS_FILE, "w") as f:
        json.dump(sorted(set(stocks)), f, indent=2)

def _is_known(ticker: str) -> bool:
    return ticker in _load_known_stocks()

def _register_stock(ticker: str) -> bool:
    """Add ticker to registry. Returns True if it was newly added."""
    stocks = _load_known_stocks()
    if ticker in stocks:
        return False
    stocks.append(ticker)
    _save_known_stocks(stocks)
    print(f"📌 New stock registered: {ticker} (total: {len(stocks)})")
    return True

# ── Artefact loading ─────────────────────────────────────────────────────────────
def _reload_artefacts():
    global _model, _label_encoder, _metadata
    if os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
        print("✅ model.pkl loaded")
    else:
        print("⚠  model.pkl not found")
    if os.path.exists(LE_PATH):
        _label_encoder = joblib.load(LE_PATH)
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            _metadata = json.load(f)

# ── Training runner ──────────────────────────────────────────────────────────────
def _run_training():
    """Blocking call to train.py. Call from a thread."""
    train_script = os.path.join(ROOT_DIR, "ml", "train.py")
    env = {**os.environ, "DATA_DIR": DATA_DIR}
    subprocess.run([sys.executable, train_script],
                   check=True, cwd=ROOT_DIR, env=env)
    _reload_artefacts()

# ── Debounced auto-retrain ───────────────────────────────────────────────────────
# When a new stock is searched, we wait DEBOUNCE_SECS before retraining.
# If another new stock arrives in that window, the timer resets —
# so multiple quick searches get batched into one retrain.
DEBOUNCE_SECS = 30

def _schedule_auto_retrain():
    """Cancel any pending timer and start a fresh one."""
    global _retrain_timer
    if _retrain_timer is not None:
        _retrain_timer.cancel()
    _retrain_timer = threading.Timer(DEBOUNCE_SECS, _do_auto_retrain)
    _retrain_timer.daemon = True
    _retrain_timer.start()
    print(f"⏱  Auto-retrain scheduled in {DEBOUNCE_SECS}s …")

def _do_auto_retrain():
    global _learning, _retrain_timer
    if _retraining:
        # Manual retrain already running — try again in 60s
        _retrain_timer = threading.Timer(60, _do_auto_retrain)
        _retrain_timer.daemon = True
        _retrain_timer.start()
        return
    _learning = True
    try:
        print("🧠 Auto-retrain started (new stock added) …")
        _run_training()
        print("🧠 Auto-retrain complete ✅")
    except Exception as exc:
        print(f"❌ Auto-retrain failed: {exc}")
    finally:
        _learning = False
        _retrain_timer = None

# ── Startup ──────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    # Ensure stock registry exists
    if not os.path.exists(STOCKS_FILE):
        _save_known_stocks(SEED_STOCKS)

    # First-boot training if no model exists
    if not os.path.exists(MODEL_PATH):
        print("🚀 First boot — running initial training (~3 min) …")
        try:
            _run_training()
        except Exception as exc:
            print(f"❌ Initial training failed: {exc}")
    else:
        _reload_artefacts()

# ── Technical indicator helpers ──────────────────────────────────────────────────
def _rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - (100 / (1 + g / l))

def _macd(s, fast=12, slow=26, sig=9):
    ml = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    sl = ml.ewm(span=sig, adjust=False).mean()
    return ml, sl

def _bb_width(s, p=20):
    sma = s.rolling(p).mean()
    std = s.rolling(p).std()
    return (sma + 2*std - (sma - 2*std)) / sma

def _live_features(ticker: str) -> dict:
    # Fetch enough history for MA200 (needs ~200 rows = ~10 months)
    df = yf.download(ticker, period="14mo", interval="1d",
                     auto_adjust=True, progress=False)
    if df.empty or len(df) < 60:
        raise ValueError(f"No data found for ticker '{ticker}'.")

    close = df["Close"].squeeze()
    vol   = df["Volume"].squeeze()
    ma50  = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1])
    ml, sl = _macd(close)

    ticker_code = 0
    if _label_encoder is not None and ticker in _label_encoder.classes_:
        ticker_code = int(_label_encoder.transform([ticker])[0])

    return {
        "RSI":         float(_rsi(close).iloc[-1]),
        "MA50":        ma50,
        "MA200":       ma200,
        "MA_Cross":    ma50 - ma200,
        "Volatility":  float(close.pct_change().rolling(10).std().iloc[-1]),
        "MACD":        float(ml.iloc[-1]),
        "MACD_Signal": float(sl.iloc[-1]),
        "MACD_Hist":   float((ml - sl).iloc[-1]),
        "BB_Width":    float(_bb_width(close).iloc[-1]),
        "Volume_Log":  float(np.log1p(vol.iloc[-1])),
        "Ticker":      ticker_code,
        "_last_price": float(close.iloc[-1]),
        "_as_of":      str(df.index[-1].date()),
    }

_FEATURE_ORDER = [
    "RSI", "MA50", "MA200", "MA_Cross", "Volatility",
    "MACD", "MACD_Signal", "MACD_Hist", "BB_Width",
    "Volume_Log", "Ticker"
]

def _run_predict(feature_values: list) -> dict:
    if _model is None:
        raise RuntimeError("Model not loaded yet. Please wait for training to complete.")
    threshold = _metadata.get("threshold", 0.6)
    prob      = float(_model.predict_proba([feature_values])[0][1])
    return {
        "signal":      "BUY" if prob > threshold else "SELL",
        "probability": round(prob * 100, 2),
        "threshold":   threshold,
        "trained_at":  _metadata.get("trained_at", "unknown"),
        "cv_accuracy": _metadata.get("cv_accuracy_mean"),
        "n_stocks":    _metadata.get("n_stocks", len(_metadata.get("stocks", []))),
    }

# ── Schemas ──────────────────────────────────────────────────────────────────────
class PredictBody(BaseModel):
    RSI: float;  MA50: float;  MA200: float;  MA_Cross: float
    Volatility: float;  MACD: float;  MACD_Signal: float
    MACD_Hist: float;  BB_Width: float;  Volume_Log: float
    Ticker: int = 0

# ── Routes ───────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":      "ok",
        "model_ready": _model is not None,
        "trained_at":  _metadata.get("trained_at"),
        "n_stocks":    len(_load_known_stocks()),
        "learning":    _learning,
        "retraining":  _retraining,
    }

@app.get("/stocks")
def list_stocks():
    """All stocks the model knows about."""
    stocks = _load_known_stocks()
    known_by_model = list(_label_encoder.classes_) if _label_encoder else []
    return {
        "registered": stocks,
        "in_model":   known_by_model,
        "total":      len(stocks),
    }

@app.post("/predict")
def predict_manual(body: PredictBody):
    vals = [getattr(body, f) for f in _FEATURE_ORDER]
    try:
        return _run_predict(vals)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@app.get("/live")
def predict_live(ticker: str):
    """
    Fetch live features and predict.
    If ticker is new → registers it and triggers a background retrain automatically.
    """
    # Validate ticker exists on Yahoo Finance
    try:
        feats = _live_features(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # Run prediction with current model (immediate response)
    vals = [feats[f] for f in _FEATURE_ORDER]
    try:
        result = _run_predict(vals)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # Check if this is a new stock and register it
    is_new = _register_stock(ticker)
    if is_new:
        _schedule_auto_retrain()

    result["features"]    = {k: v for k, v in feats.items() if not k.startswith("_")}
    result["last_price"]  = feats["_last_price"]
    result["as_of"]       = feats["_as_of"]
    result["new_stock"]   = is_new
    result["learning"]    = is_new or _learning
    result["total_stocks"] = len(_load_known_stocks())
    return result

@app.get("/metadata")
def get_metadata():
    if not _metadata:
        raise HTTPException(status_code=404, detail="No metadata yet.")
    return {**_metadata, "registered_stocks": _load_known_stocks()}

@app.get("/learning")
def learning_status():
    return {
        "learning":   _learning,
        "retraining": _retraining,
        "n_stocks":   len(_load_known_stocks()),
    }

@app.post("/retrain")
def manual_retrain(background_tasks: BackgroundTasks):
    """User-triggered full retrain."""
    global _retraining
    if _retraining or _learning:
        raise HTTPException(status_code=409, detail="Training already in progress.")

    def _run():
        global _retraining
        _retraining = True
        try:
            _run_training()
        except Exception as exc:
            print(f"❌ Manual retrain failed: {exc}")
        finally:
            _retraining = False

    background_tasks.add_task(_run)
    return {"status": "retraining started"}

@app.get("/retrain/status")
def retrain_status():
    return {"retraining": _retraining, "learning": _learning}

# ── Static files ─────────────────────────────────────────────────────────────────
if os.path.exists(PUBLIC_DIR):
    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="static")
