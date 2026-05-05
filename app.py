"""
server/app.py — FastAPI backend for the Quant Pipeline.

Start with:
    uvicorn server.app:app --reload --port 3000

Endpoints:
    GET  /              → health check + training date
    POST /predict       → manual feature input → BUY / SELL
    GET  /live          → ?ticker=RELIANCE.NS  → auto-fetch + predict
    GET  /metadata      → model info (accuracy, features, training date)
    POST /retrain       → kick off background retraining
    GET  /retrain/status → is a retrain in progress?
"""

import os
import sys
import json
import subprocess

import numpy as np
import joblib
import yfinance as yf
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "model.pkl")
LE_PATH    = os.path.join(ROOT_DIR, "label_encoder.pkl")
META_PATH  = os.path.join(ROOT_DIR, "model_metadata.json")
PUBLIC_DIR = os.path.join(ROOT_DIR, "public")

# ── App ─────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Quant Pipeline API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to specific origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Module-level state (loaded once at startup) ──────────────────────────────────
_model         = None
_label_encoder = None
_metadata: dict = {}
_retraining    = False

def _reload_artefacts():
    """Load / reload model, encoder, and metadata from disk."""
    global _model, _label_encoder, _metadata
    if os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
        print("✅ model.pkl loaded")
    else:
        print("⚠  model.pkl not found — run  python ml/train.py  first")
    if os.path.exists(LE_PATH):
        _label_encoder = joblib.load(LE_PATH)
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            _metadata = json.load(f)

@app.on_event("startup")
def startup():
    # Auto-train on first deploy if no model exists yet
    if not os.path.exists(MODEL_PATH):
        print("🚀 No model found — running initial training (this takes ~2 min)…")
        train_script = os.path.join(ROOT_DIR, "ml", "train.py")
        try:
            subprocess.run(
                [sys.executable, train_script],
                check=True, cwd=ROOT_DIR
            )
        except subprocess.CalledProcessError as exc:
            print(f"❌ Initial training failed: {exc}")
    _reload_artefacts()

# ── Technical indicator helpers ──────────────────────────────────────────────────
def _rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - (100 / (1 + g / l))

def _macd(s, fast=12, slow=26, sig=9):
    ml  = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    sl  = ml.ewm(span=sig, adjust=False).mean()
    return ml, sl

def _bb_width(s, p=20):
    sma = s.rolling(p).mean()
    std = s.rolling(p).std()
    return (sma + 2*std - (sma - 2*std)) / sma

# ── Live feature fetch ──────────────────────────────────────────────────────────
def _live_features(ticker: str) -> dict:
    df = yf.download(ticker, period="1y", interval="1d",
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'.")

    close = df["Close"].squeeze()
    vol   = df["Volume"].squeeze()

    ma50   = float(close.rolling(50).mean().iloc[-1])
    ma200  = float(close.rolling(200).mean().iloc[-1])
    ml, sl = _macd(close)

    ticker_code = 0
    if _label_encoder is not None and ticker in _label_encoder.classes_:
        ticker_code = int(_label_encoder.transform([ticker])[0])

    feats = {
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
        # extra context returned to UI
        "_last_price": float(close.iloc[-1]),
        "_as_of":      str(df.index[-1].date()),
    }
    return feats

_FEATURE_ORDER = [
    "RSI", "MA50", "MA200", "MA_Cross", "Volatility",
    "MACD", "MACD_Signal", "MACD_Hist", "BB_Width",
    "Volume_Log", "Ticker"
]

def _run_predict(feature_values: list) -> dict:
    if _model is None:
        raise RuntimeError("Model not loaded. Run  python ml/train.py  first.")
    threshold = _metadata.get("threshold", 0.6)
    prob      = float(_model.predict_proba([feature_values])[0][1])
    return {
        "signal":       "BUY" if prob > threshold else "SELL",
        "probability":  round(prob * 100, 2),
        "threshold":    threshold,
        "trained_at":   _metadata.get("trained_at", "unknown"),
        "cv_accuracy":  _metadata.get("cv_accuracy_mean"),
    }

# ── Pydantic schema ─────────────────────────────────────────────────────────────
class PredictBody(BaseModel):
    RSI:         float
    MA50:        float
    MA200:       float
    MA_Cross:    float
    Volatility:  float
    MACD:        float
    MACD_Signal: float
    MACD_Hist:   float
    BB_Width:    float
    Volume_Log:  float
    Ticker:      int = 0

# ── Routes ──────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":     "ok",
        "model_ready": _model is not None,
        "trained_at":  _metadata.get("trained_at"),
    }

@app.post("/predict")
def predict_manual(body: PredictBody):
    """BUY / SELL from manually supplied feature values."""
    vals = [getattr(body, f) for f in _FEATURE_ORDER]
    try:
        return _run_predict(vals)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@app.get("/live")
def predict_live(ticker: str):
    """
    Fetch live data from Yahoo Finance, compute features, and predict.
    ?ticker=RELIANCE.NS
    """
    try:
        feats = _live_features(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    vals   = [feats[f] for f in _FEATURE_ORDER]
    try:
        result = _run_predict(vals)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    result["features"]   = {k: v for k, v in feats.items() if not k.startswith("_")}
    result["last_price"] = feats["_last_price"]
    result["as_of"]      = feats["_as_of"]
    return result

@app.get("/metadata")
def get_metadata():
    if not _metadata:
        raise HTTPException(status_code=404, detail="No metadata found. Train a model first.")
    return _metadata

@app.post("/retrain")
def retrain(background_tasks: BackgroundTasks):
    """Kick off a background model retrain. Returns immediately."""
    global _retraining
    if _retraining:
        raise HTTPException(status_code=409, detail="Retraining already in progress.")

    def _run():
        global _retraining
        _retraining = True
        try:
            train_script = os.path.join(ROOT_DIR, "ml", "train.py")
            subprocess.run(
                [sys.executable, train_script],
                check=True, cwd=ROOT_DIR
            )
            _reload_artefacts()
        except Exception as exc:
            print(f"❌ Retrain failed: {exc}")
        finally:
            _retraining = False

    background_tasks.add_task(_run)
    return {"status": "retraining started"}

@app.get("/retrain/status")
def retrain_status():
    return {"retraining": _retraining}

# ── Serve static files (must be last) ───────────────────────────────────────────
if os.path.exists(PUBLIC_DIR):
    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="static")
