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

# Local modules
sys.path.insert(0, ROOT_DIR if 'ROOT_DIR' in dir() else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concurrent.futures import ThreadPoolExecutor, as_completed


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

# Full Nifty 50 universe — used to expand the registry on startup
SEED_STOCKS = [
    "HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","AXISBANK.NS","KOTAKBANK.NS","INDUSINDBK.NS",
    "BAJFINANCE.NS","BAJAJFINSV.NS","SBILIFE.NS","HDFCLIFE.NS","SHRIRAMFIN.NS",
    "TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS","LTIM.NS",
    "RELIANCE.NS","ONGC.NS","BPCL.NS","COALINDIA.NS","NTPC.NS","POWERGRID.NS",
    "HINDUNILVR.NS","ITC.NS","BRITANNIA.NS","NESTLEIND.NS","TATACONSUM.NS",
    "TATAMOTORS.NS","MARUTI.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS","EICHERMOT.NS","M&M.NS",
    "LT.NS","ADANIPORTS.NS","ULTRACEMCO.NS","GRASIM.NS",
    "SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","APOLLOHOSP.NS",
    "TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS",
    "ASIANPAINT.NS","TITAN.NS","TRENT.NS","BHARTIARTL.NS",
]

# ── Sector map (same as train.py) ────────────────────────────────────────────────
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


# ── Trade Levels (Entry, Stop Loss, Targets, Time Horizon) ──────────────────────
def _compute_trade_levels(feats: dict, consensus: dict, ml_signal: str,
                          horizon: dict) -> dict:
    """
    Always returns a trade card. Signal confidence tier determines
    how tight/wide the levels are and what action is recommended.

    Confidence tiers:
      HIGH    — consensus directional + ML agree  → tight levels, act
      MEDIUM  — one directional, other NEUTRAL    → normal levels, consider
      LOW     — both NEUTRAL                      → wide levels, observe only
      WATCH   — signals disagree (one BUY, one SELL) → no levels, wait
    """
    price = feats.get("_last_price") or feats.get("last_price") or 0
    atr   = feats.get("ATR") or 0

    if not price or price <= 0:
        price = 0
    if not atr or atr <= 0:
        atr = price * 0.015 if price > 0 else 1

    consensus_signal   = (consensus or {}).get("signal", "NEUTRAL")
    consensus_conf     = (consensus or {}).get("confidence", 0)
    agreement_label    = (consensus or {}).get("agreement_label", "—")
    votes              = (consensus or {}).get("votes", {})
    engine_breakdown   = (consensus or {}).get("engine_breakdown", [])

    # ── Determine confidence tier ─────────────────────────────────────────────
    if consensus_signal != "NEUTRAL" and ml_signal == consensus_signal:
        tier        = "HIGH"
        action_sig  = consensus_signal
        risk_mult   = 1.5
        action_text = f"Strong {action_sig} — {agreement_label} engines agree"
        action_color= "buy" if action_sig == "BUY" else "sell"

    elif consensus_signal != "NEUTRAL" and ml_signal == "NEUTRAL":
        tier        = "MEDIUM"
        action_sig  = consensus_signal
        risk_mult   = 1.8
        action_text = f"Tentative {action_sig} — engines agree, ML model neutral"
        action_color= "buy" if action_sig == "BUY" else "sell"

    elif consensus_signal == "NEUTRAL" and ml_signal != "NEUTRAL":
        tier        = "MEDIUM"
        action_sig  = ml_signal
        risk_mult   = 2.0
        action_text = f"Weak {action_sig} — ML model signal, engines mixed"
        action_color= "buy" if action_sig == "BUY" else "sell"

    elif (consensus_signal == "BUY" and ml_signal == "SELL") or          (consensus_signal == "SELL" and ml_signal == "BUY"):
        tier        = "WATCH"
        action_sig  = "WATCH"
        risk_mult   = 2.0
        action_text = "Conflicting signals — wait for clarity before entering"
        action_color= "warn"

    else:
        # Both NEUTRAL
        tier        = "LOW"
        action_sig  = "NEUTRAL"
        risk_mult   = 2.5
        action_text = "No clear signal — observe only, do not trade"
        action_color= "dim"

    # ── Compute price levels ──────────────────────────────────────────────────
    risk = atr * risk_mult

    period_map = {"Intraday":"Intraday","Short-term":"3–5 Days",
                  "Swing":"1–3 Weeks","Long-term":"Months+"}
    primary    = (horizon or {}).get("primary", "Short-term")
    time_label = period_map.get(primary, "3–5 Days")

    if action_sig in ("BUY",):
        entry_low  = round(price * 0.997, 2)
        entry_high = round(price * 1.003, 2)
        stop_loss  = round(price - risk, 2)
        target1    = round(price + risk,     2)
        target2    = round(price + risk * 2, 2)
        stop_pct   = round((stop_loss - price) / price * 100, 2) if price else 0
        t1_pct     = round((target1   - price) / price * 100, 2) if price else 0
        t2_pct     = round((target2   - price) / price * 100, 2) if price else 0

    elif action_sig in ("SELL",):
        entry_low  = round(price * 0.997, 2)
        entry_high = round(price * 1.003, 2)
        stop_loss  = round(price + risk, 2)
        target1    = round(price - risk,     2)
        target2    = round(price - risk * 2, 2)
        stop_pct   = round((stop_loss - price) / price * 100, 2) if price else 0
        t1_pct     = round((target1   - price) / price * 100, 2) if price else 0
        t2_pct     = round((target2   - price) / price * 100, 2) if price else 0

    else:
        # NEUTRAL / WATCH — show observation range, no directional targets
        entry_low  = round(price * 0.990, 2)
        entry_high = round(price * 1.010, 2)
        stop_loss  = None
        target1    = None
        target2    = None
        stop_pct   = None
        t1_pct     = None
        t2_pct     = None

    # ── Engine-by-engine recommendation summary ───────────────────────────────
    engine_recs = []
    ENGINE_LABELS = {
        "xgboost":"XGBoost ML","multi_timeframe":"Multi-Timeframe",
        "mean_reversion":"Mean Reversion","sentiment":"Sentiment",
        "fundamental_rank":"Fundamental",
    }
    for bd in engine_breakdown:
        engine_recs.append({
            "name":   ENGINE_LABELS.get(bd.get("engine",""), bd.get("engine","")),
            "signal": bd.get("signal","NEUTRAL"),
            "weight": bd.get("weight", 0),
            "detail": bd.get("detail",""),
        })

    # ── Tier-based guidance text ──────────────────────────────────────────────
    guidance_map = {
        "HIGH":   "High confidence — consider entering with defined risk.",
        "MEDIUM": "Moderate confidence — use smaller position size, keep stop tight.",
        "LOW":    "Low confidence — wait for more signals to align before trading.",
        "WATCH":  "Conflicting signals — do not enter. Wait for consensus.",
    }

    # ── Position Sizing ──────────────────────────────────────────────────────
    # Formula: position_pct = base × (confidence/100) / vol_norm × regime_mult
    # base = 2% of capital; vol_norm = ATR/price (normalised 0-1 via clip)
    BASE_SIZE    = 2.0     # % of capital as starting point
    conf_ratio   = consensus_conf / 100 if consensus_conf else 0.5
    vol_norm     = min(1.0, (atr / price) / 0.03) if price > 0 else 1.0  # 3% ATR = 1.0
    regime_mult  = {
        "HIGH":   1.20,
        "MEDIUM": 1.00,
        "LOW":    0.60,
        "WATCH":  0.30,
    }.get(tier, 0.80)

    raw_size   = BASE_SIZE * conf_ratio / max(vol_norm, 0.1) * regime_mult
    pos_pct    = round(min(5.0, max(0.25, raw_size)), 2)   # cap at 5%, floor at 0.25%

    if action_sig in ("NEUTRAL", "WATCH") or tier in ("LOW", "WATCH"):
        pos_guidance = "Do not open a position — wait for confirmation"
        pos_pct      = 0.0
    elif pos_pct >= 3.5:
        pos_guidance = f"Full size: high confidence + low volatility"
    elif pos_pct >= 2.0:
        pos_guidance = f"Normal size: moderate confidence"
    else:
        pos_guidance = f"Half size: lower confidence or elevated volatility"

    return {
        "signal":         action_sig,
        "tier":           tier,
        "action_text":    action_text,
        "action_color":   action_color,
        "guidance":       guidance_map[tier],
        "entry_low":      entry_low,
        "entry_high":     entry_high,
        "entry_mid":      round(price, 2),
        "stop_loss":      stop_loss,
        "stop_pct":       stop_pct,
        "target1":        target1,
        "target1_pct":    t1_pct,
        "target2":        target2,
        "target2_pct":    t2_pct,
        "time_horizon":   time_label if action_sig not in ("NEUTRAL","WATCH") else "Wait for signal",
        "risk_reward":    "1:1 & 2:1" if action_sig in ("BUY","SELL") else "—",
        "risk_mult":      risk_mult,
        "atr_used":       round(atr, 2),
        "position_pct":   pos_pct,
        "position_guidance": pos_guidance,
        "engine_recs":    engine_recs,
        "votes":          votes,
        "consensus_conf": round(consensus_conf, 1),
        "ml_signal":      ml_signal,
        "consensus_signal": consensus_signal,
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
        with open(STOCKS_FILE) as f:
            existing = json.load(f)
        # Merge any new SEED_STOCKS into existing registry automatically
        new_seeds = [s for s in SEED_STOCKS if s not in existing]
        if new_seeds:
            merged = sorted(set(existing + new_seeds))
            _save_known_stocks(merged)
            print(f"📋 Registry expanded {len(existing)}→{len(merged)} stocks")
            return merged
        return existing
    _save_known_stocks(SEED_STOCKS)
    return SEED_STOCKS.copy()

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
def _strip_model_feature_names(model):
    """
    Strip whitespace from XGBoost booster feature names — fixes legacy models.

    Note: feature_names_in_ is read-only in newer sklearn/XGBoost versions,
    so we only strip the booster's internal list. Since _run_predict always
    passes numpy arrays (.values), XGBoost skips name validation entirely —
    this strip is just a belt-and-suspenders cleanup for logging clarity.
    """
    try:
        booster = model.get_booster()
        if booster.feature_names:
            booster.feature_names = [f.strip() for f in booster.feature_names]
    except Exception as e:
        print(f"⚠  Could not strip booster feature names: {e}")

def _reload_artefacts():
    global _global_model, _label_encoder, _metadata, _stock_models
    _stock_models = {}  # clear per-stock cache so fresh models load

    if os.path.exists(MODEL_PATH):
        _global_model = joblib.load(MODEL_PATH)
        _strip_model_feature_names(_global_model)
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
        # Strip whitespace from stored feature lists too
        for key in ("global_features","stock_features"):
            if key in _metadata:
                _metadata[key] = [f.strip() for f in _metadata[key]]

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
def _live_features(ticker: str, df=None) -> dict:
    # Need ~14 months for MA200 + 52W features
    if df is None:
        df = yf.download(ticker, period="14mo", interval="1d",
                         auto_adjust=True, progress=False)
    if df.empty or len(df) < 60:
        raise ValueError(f"No data found for '{ticker}'.")

    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    vol   = df["Volume"].squeeze()

    # All global data downloaded in parallel
    import concurrent.futures as _cf
    _GLOBAL_SYMS = {
        "nifty":    "^NSEI",
        "usdinr":   "USDINR=X",
        "crude":    "BZ=F",
        "sp500":    "^GSPC",
        "nasdaq":   "^IXIC",
        "vix_us":   "^VIX",
        "vix_in":   "^INDIAVIX",
        "us10y":    "^TNX",
        "copper":   "HG=F",
        "shanghai": "000001.SS",
    }
    def _dl_live(sym):
        try:
            df = yf.download(sym, period="14mo", interval="1d",
                             auto_adjust=True, progress=False)
            return df["Close"].squeeze() if not df.empty else pd.Series(dtype=float)
        except Exception:
            return pd.Series(dtype=float)

    with _cf.ThreadPoolExecutor(max_workers=10) as _ex:
        _futs = {k: _ex.submit(_dl_live, v) for k, v in _GLOBAL_SYMS.items()}
        _gd   = {k: f.result(timeout=20) for k, f in _futs.items()}

    nifty_close    = _gd["nifty"]
    nifty_ma200    = nifty_close.rolling(200).mean()
    nifty_return   = nifty_close.pct_change()
    usdinr_close   = _gd["usdinr"]
    crude_close    = _gd["crude"]
    sp500_close    = _gd["sp500"]
    nasdaq_close   = _gd["nasdaq"]
    vix_us_close   = _gd["vix_us"]
    vix_in_close   = _gd["vix_in"]
    us10y_close    = _gd["us10y"]
    copper_close   = _gd["copper"]
    shanghai_close = _gd["shanghai"]

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
        # Phase 1: Momentum features
        "Return_1d":      float(close.pct_change().iloc[-1]),
        "Return_5d_lag":  float(close.pct_change(5).iloc[-1]),
        "Return_20d":     float(close.pct_change(20).iloc[-1]),
        # Phase 1: Beta vs Nifty (60d rolling)
        # Fix: .iloc[-1] must be inside float(), not outside
        "Beta_60d": float((
            close.pct_change().rolling(60).cov(nifty_return.reindex(close.index).fillna(0)) /
            (nifty_return.reindex(close.index).fillna(0).rolling(60).var() + 1e-9)
        ).iloc[-1]) if len(close) > 60 else 1.0,
        "Rel_Strength":   float(
            close.pct_change(20).iloc[-1] - nifty_close.reindex(close.index).ffill().pct_change(20).iloc[-1]
        ) if len(close) > 20 else 0.0,
        # Phase 4: Market structure features
        "Dist_MA20":   float((close.iloc[-1] - close.rolling(20).mean().iloc[-1]) /
                              (close.rolling(20).mean().iloc[-1] + 1e-9)),
        "Dist_MA50":   float((close.iloc[-1] - close.rolling(50).mean().iloc[-1]) /
                              (close.rolling(50).mean().iloc[-1] + 1e-9)),
        "MA20_Slope":  float(close.rolling(20).mean().pct_change(5).iloc[-1]),
        "MA50_Slope":  float(close.rolling(50).mean().pct_change(10).iloc[-1]),
        "BB_Position": float(
            ((close.iloc[-1] - (close.rolling(20).mean().iloc[-1] - 2*close.rolling(20).std().iloc[-1])) /
             (4 * close.rolling(20).std().iloc[-1] + 1e-9)).clip(0, 1)
        ),
        # Phase 5B: Macro — Dollar, Crude, Seasonal
        "USDINR_Return":   float(usdinr_close.reindex(close.index).ffill()
                                 .pct_change().iloc[-1]) if len(usdinr_close) > 1 else 0.0,
        "USDINR_20d_Mom":  float(usdinr_close.reindex(close.index).ffill()
                                 .pct_change(20).iloc[-1]) if len(usdinr_close) > 20 else 0.0,
        "Crude_Return":    float(crude_close.reindex(close.index).ffill()
                                 .pct_change().iloc[-1]) if len(crude_close) > 1 else 0.0,
        "Crude_20d_Mom":   float(crude_close.reindex(close.index).ffill()
                                 .pct_change(20).iloc[-1]) if len(crude_close) > 20 else 0.0,
        "Month_Sin":       float(np.sin(2 * np.pi * last_date.month / 12)),
        "Month_Cos":       float(np.cos(2 * np.pi * last_date.month / 12)),
        "Is_Budget_Month": int(last_date.month == 2),
        "Is_Monsoon":      int(last_date.month in [6, 7, 8, 9]),
        # Phase 6: Global macro
        "SP500_Return":    float(sp500_close.reindex(close.index).ffill().pct_change().iloc[-1])
                           if len(sp500_close) > 1 else 0.0,
        "SP500_5d":        float(sp500_close.reindex(close.index).ffill().pct_change(5).iloc[-1])
                           if len(sp500_close) > 5 else 0.0,
        "VIX_US_Level":    float(vix_us_close.reindex(close.index).ffill().iloc[-1] / 100)
                           if len(vix_us_close) > 1 else 0.20,
        "VIX_IN_ROC5":     float(vix_in_close.reindex(close.index).ffill().pct_change(5).iloc[-1])
                           if len(vix_in_close) > 5 else 0.0,
        "VIX_IN_Pct":      float(vix_in_close.reindex(close.index).ffill()
                                 .rolling(252, min_periods=30).rank(pct=True).iloc[-1])
                           if len(vix_in_close) > 30 else 0.5,
        "US10Y_Level":     float(us10y_close.reindex(close.index).ffill().iloc[-1] / 100)
                           if len(us10y_close) > 1 else 0.04,
        "US10Y_Chg":       float(us10y_close.reindex(close.index).ffill().diff().iloc[-1])
                           if len(us10y_close) > 1 else 0.0,
        "FII_Proxy":       float(
            nifty_return.reindex(close.index).fillna(0).iloc[-1] -
            sp500_close.reindex(close.index).ffill().pct_change().fillna(0).iloc[-1]
        ) if len(sp500_close) > 1 else 0.0,
        "Copper_Return":   float(copper_close.reindex(close.index).ffill().pct_change().iloc[-1])
                           if len(copper_close) > 1 else 0.0,
        "Shanghai_Return": float(shanghai_close.reindex(close.index).ffill().pct_change().iloc[-1])
                           if len(shanghai_close) > 1 else 0.0,

        # Phase 7: NSE official data (live values from NSE APIs)
        # PCR from option chain — will be filled after NSE fetch below
        "PCR":          1.0,        # placeholder — updated below
        "PCR_Signal":   0.0,
        "FII_Net_Norm": 0.0,
        "DII_Net_Norm": 0.0,
        "Breadth_Pct":  50.0,
        "AdvDec_Ratio": 50.0,
        # Phase 6: Sector-conditional
        "NASDAQ_IT":       float(nasdaq_close.reindex(close.index).ffill().pct_change().iloc[-1]
                                  if len(nasdaq_close) > 1 else 0.0) * int(sector_code == 2),
        "USD_Export":      float(usdinr_close.reindex(close.index).ffill().pct_change().iloc[-1]
                                  if len(usdinr_close) > 1 else 0.0) * int(sector_code in [2, 7]),
        "Crude_Sector":    float(crude_close.reindex(close.index).ffill().pct_change().iloc[-1]
                                  if len(crude_close) > 1 else 0.0) * int(sector_code == 4),
        "Copper_Sector":   float(copper_close.reindex(close.index).ffill().pct_change().iloc[-1]
                                  if len(copper_close) > 1 else 0.0) * int(sector_code == 8),
        "Shanghai_Sector": float(shanghai_close.reindex(close.index).ffill().pct_change().iloc[-1]
                                  if len(shanghai_close) > 1 else 0.0) * int(sector_code in [6, 8]),
        "Yield_Banking":   float(us10y_close.reindex(close.index).ffill().diff().iloc[-1]
                                  if len(us10y_close) > 1 else 0.0) * int(sector_code in [0, 1]),
        "Monsoon_FMCG":    int(last_date.month in [6, 7, 8, 9]) * int(sector_code == 5),
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
    # Use booster's own feature names (strips trailing spaces from legacy models)
    try:
        feature_list = [f.strip() for f in model.get_booster().feature_names]
    except Exception:
        feature_list = [f.strip() for f in feature_list]
    clean_feats = {k.strip(): v for k, v in feats.items()}
    vals = [clean_feats.get(f, 0) for f in feature_list]
    # numpy array bypasses XGBoost feature name validation entirely
    prob = float(model.predict_proba(np.array([vals]))[0][1])

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
    """
    Accepts all 54 model features.
    extra='allow' means any additional feature sent from the frontend
    (Phase 1/4/5B/6) is passed through automatically — no need to list them all.
    """
    model_config = {"extra": "allow"}

    # Core 19 (always required from form)
    RSI:float=50.0; MA50:float=0.0; MA200:float=0.0
    MA_Cross:float=0.0; Volatility:float=0.02
    MACD:float=0.0; MACD_Signal:float=0.0; MACD_Hist:float=0.0
    BB_Width:float=0.05; Volume_Log:float=14.0
    Volume_Spike:float=1.0; ATR:float=0.0
    High52W_Pct:float=0.95; Low52W_Pct:float=1.05
    Market_Return:float=0.0; Market_Regime:int=1; Earnings_Season:int=0
    Ticker:int=0; Sector:int=7
    ticker:str=""   # for model selection

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
    feats  = body.dict()
    ticker = feats.pop("ticker", "") or "UNKNOWN"

    # ── Fetch live macro context so manual matches live prediction ──────────
    # Without this, Phase 5B/6 features default to 0, diverging from live.
    try:
        from concurrent.futures import ThreadPoolExecutor as _TPE
        _MSYMS = {
            "nifty": "^NSEI", "usdinr": "USDINR=X", "crude": "BZ=F",
            "sp500": "^GSPC", "nasdaq": "^IXIC", "vix_us": "^VIX",
            "vix_in": "^INDIAVIX", "us10y": "^TNX",
            "copper": "HG=F", "shanghai": "000001.SS",
        }
        def _qdl(sym):
            try:
                df = yf.download(sym, period="2mo", interval="1d",
                                 auto_adjust=True, progress=False)
                return df["Close"].squeeze() if not df.empty else None
            except Exception: return None

        with _TPE(max_workers=10) as _ex:
            _macro = {k: _ex.submit(_qdl, v).result(timeout=12)
                      for k, v in _MSYMS.items()}

        def _lr(s, p=1):
            if s is None or len(s) < p+1: return 0.0
            v = s.pct_change(p).iloc[-1]
            return float(v) if not np.isnan(v) else 0.0

        def _lv(s, default=0.0):
            if s is None or s.empty: return default
            v = s.iloc[-1]; return float(v) if not np.isnan(v) else default

        _now = pd.Timestamp.now()
        _sc  = feats.get("Sector", 7)
        _sp5r = _lr(_macro.get("sp500"))
        _nifr = _lr(_macro.get("nifty"))
        _usd  = _lr(_macro.get("usdinr"))
        _crd  = _lr(_macro.get("crude"))
        _copr = _lr(_macro.get("copper"))
        _shan = _lr(_macro.get("shanghai"))
        _nsdq = _lr(_macro.get("nasdaq"))

        _us10y_s = _macro.get("us10y")
        _us10y_chg = float(_us10y_s.diff().iloc[-1])             if _us10y_s is not None and len(_us10y_s) > 1 else 0.0

        _vixin = _macro.get("vix_in")
        _vix_roc = float(_vixin.pct_change(5).iloc[-1])             if _vixin is not None and len(_vixin) > 5 else 0.0

        # Phase 1 — momentum (no historical data for manual, use neutral)
        feats.setdefault("Return_1d",     0.0)
        feats.setdefault("Return_5d_lag", 0.0)
        feats.setdefault("Return_20d",    0.0)
        feats.setdefault("Beta_60d",      1.0)   # market-neutral default
        feats.setdefault("Rel_Strength",  0.0)

        # Phase 4 — derive from MA values user entered
        _ma50 = feats.get("MA50", 0) or 0
        _ma20 = _ma50 * 0.99   # approximate
        _price = _ma50 * 1.01
        _bb_w  = feats.get("BB_Width", 0.05) or 0.05
        feats.setdefault("Dist_MA20",   (_price - _ma20) / (_ma20 + 1e-9) if _ma20 else 0.0)
        feats.setdefault("Dist_MA50",   (_price - _ma50) / (_ma50 + 1e-9) if _ma50 else 0.0)
        feats.setdefault("MA20_Slope",  0.0)
        feats.setdefault("MA50_Slope",  0.0)
        feats.setdefault("BB_Position", 0.5)

        # Phase 5B — live macro
        feats.setdefault("USDINR_Return",   _usd)
        feats.setdefault("USDINR_20d_Mom",  _lr(_macro.get("usdinr"), 20))
        feats.setdefault("Crude_Return",    _crd)
        feats.setdefault("Crude_20d_Mom",   _lr(_macro.get("crude"), 20))
        feats.setdefault("Month_Sin",       float(np.sin(2*np.pi*_now.month/12)))
        feats.setdefault("Month_Cos",       float(np.cos(2*np.pi*_now.month/12)))
        feats.setdefault("Is_Budget_Month", int(_now.month == 2))
        feats.setdefault("Is_Monsoon",      int(_now.month in [6,7,8,9]))

        # Phase 6 — global macro (live)
        feats.setdefault("SP500_Return",    _sp5r)
        feats.setdefault("SP500_5d",        _lr(_macro.get("sp500"), 5))
        feats.setdefault("VIX_US_Level",    _lv(_macro.get("vix_us"), 20.0) / 100)
        feats.setdefault("VIX_IN_ROC5",     _vix_roc)
        feats.setdefault("VIX_IN_Pct",      0.5)
        feats.setdefault("US10Y_Level",     _lv(_us10y_s, 4.0) / 100)
        feats.setdefault("US10Y_Chg",       _us10y_chg)
        feats.setdefault("FII_Proxy",       _nifr - _sp5r)
        feats.setdefault("Copper_Return",   _copr)
        feats.setdefault("Shanghai_Return", _shan)

        # Phase 6 — sector-conditional (live)
        feats.setdefault("NASDAQ_IT",       _nsdq * int(_sc == 2))
        feats.setdefault("USD_Export",      _usd  * int(_sc in [2, 7]))
        feats.setdefault("Crude_Sector",    _crd  * int(_sc == 4))
        feats.setdefault("Copper_Sector",   _copr * int(_sc == 8))
        feats.setdefault("Shanghai_Sector", _shan * int(_sc in [6, 8]))
        feats.setdefault("Yield_Banking",   _us10y_chg * int(_sc in [0, 1]))
        feats.setdefault("Monsoon_FMCG",    feats.get("Is_Monsoon", 0) * int(_sc == 5))

    except Exception as _me:
        print(f"⚠  Macro fetch for manual predict failed: {_me}")
        # Fallback: fill everything with neutral defaults so model still runs
        _now = pd.Timestamp.now()
        for _k, _v in {
            "Return_1d":0.0, "Return_5d_lag":0.0, "Return_20d":0.0,
            "Beta_60d":1.0, "Rel_Strength":0.0,
            "Dist_MA20":0.0, "Dist_MA50":0.0, "MA20_Slope":0.0,
            "MA50_Slope":0.0, "BB_Position":0.5,
            "USDINR_Return":0.0, "USDINR_20d_Mom":0.0,
            "Crude_Return":0.0, "Crude_20d_Mom":0.0,
            "Month_Sin":float(np.sin(2*np.pi*_now.month/12)),
            "Month_Cos":float(np.cos(2*np.pi*_now.month/12)),
            "Is_Budget_Month":int(_now.month==2), "Is_Monsoon":int(_now.month in [6,7,8,9]),
            "SP500_Return":0.0, "SP500_5d":0.0, "VIX_US_Level":0.20,
            "VIX_IN_ROC5":0.0, "VIX_IN_Pct":0.5,
            "US10Y_Level":0.04, "US10Y_Chg":0.0, "FII_Proxy":0.0,
            "Copper_Return":0.0, "Shanghai_Return":0.0,
            "NASDAQ_IT":0.0, "USD_Export":0.0, "Crude_Sector":0.0,
            "Copper_Sector":0.0, "Shanghai_Sector":0.0,
            "Yield_Banking":0.0, "Monsoon_FMCG":0.0,
        }.items():
            feats.setdefault(_k, _v)

    try:
        result = _run_predict(ticker, feats)
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# ── Trading Horizon Recommendation ──────────────────────────────────────────────
def _compute_trading_horizon(feats: dict, mtf_result: dict,
                              fun_result: dict, vix_result: dict) -> dict:
    """
    Determines which trading horizons this stock is currently suited for,
    based on technical features, timeframe alignment, fundamentals and VIX.

    IMPORTANT: The model is trained on daily data. Intraday is flagged but
    marked as 'low confidence' since tick-level data was not used in training.
    """
    rsi        = feats.get("RSI", 50)
    vol_spike  = feats.get("Volume_Spike", 1.0)
    atr        = feats.get("ATR", 0)
    macd_hist  = feats.get("MACD_Hist", 0)
    ma_cross   = feats.get("MA_Cross", 0)
    bb_width   = feats.get("BB_Width", 0.05)
    hi52w      = feats.get("High52W_Pct", 0.95)
    regime     = feats.get("Market_Regime", 1)
    earn_szn   = feats.get("Earnings_Season", 0)

    daily_trend  = mtf_result.get("daily",   "UNKNOWN")
    weekly_trend = mtf_result.get("weekly",  "UNKNOWN")
    monthly_trend= mtf_result.get("monthly", "UNKNOWN")
    mom_1m       = mtf_result.get("mom_1m",  0)
    mom_3m       = mtf_result.get("mom_3m",  0)

    fund_score   = fun_result.get("score",  0.5)
    vix          = vix_result.get("vix",    18)
    vix_regime   = vix_result.get("regime", "NORMAL")

    horizons = []

    # ── Intraday ──────────────────────────────────────────────────────────────
    # High ATR + volume spike = intraday volatility present
    # BUT: model is daily — always low confidence for intraday
    intraday_signals = []
    if vol_spike > 2.0:  intraday_signals.append(f"Volume spike {vol_spike:.1f}x")
    if rsi < 32:         intraday_signals.append(f"RSI oversold ({rsi:.0f})")
    if rsi > 68:         intraday_signals.append(f"RSI overbought ({rsi:.0f})")
    if bb_width > 0.12:  intraday_signals.append("Wide Bollinger bands")
    if vix > 22:         intraday_signals.append(f"Elevated VIX ({vix:.1f})")

    horizons.append({
        "horizon":     "Intraday",
        "icon":        "⚡",
        "suitable":    len(intraday_signals) >= 2,
        "confidence":  "Low",
        "reasons":     intraday_signals[:2] if intraday_signals else ["Low intraday volatility indicators"],
        "disclaimer":  "Model trained on daily data — intraday precision limited",
        "color":       "warn",
    })

    # ── Short-term: 3–5 days ──────────────────────────────────────────────────
    short_score = 0
    short_reasons = []
    if macd_hist > 0:
        short_score += 2; short_reasons.append("MACD histogram positive")
    if "UP" in daily_trend:
        short_score += 2; short_reasons.append(f"Daily trend: {daily_trend}")
    if vol_spike > 1.3:
        short_score += 1; short_reasons.append(f"Above-avg volume ({vol_spike:.1f}x)")
    if 40 < rsi < 65:
        short_score += 1; short_reasons.append(f"RSI in momentum zone ({rsi:.0f})")
    if earn_szn:
        short_score += 1; short_reasons.append("Earnings season — catalyst potential")

    horizons.append({
        "horizon":     "Short-term",
        "icon":        "📈",
        "period":      "3–5 days",
        "suitable":    short_score >= 4,
        "confidence":  "High" if short_score >= 5 else "Medium" if short_score >= 3 else "Low",
        "reasons":     short_reasons[:3],
        "color":       "accent",
    })

    # ── Swing: 1–3 weeks ─────────────────────────────────────────────────────
    swing_score = 0
    swing_reasons = []
    if "UP" in weekly_trend:
        swing_score += 3; swing_reasons.append(f"Weekly trend: {weekly_trend}")
    if ma_cross > 0:
        swing_score += 2; swing_reasons.append("MA50 above MA200 (golden cross zone)")
    if mom_3m > 5:
        swing_score += 1; swing_reasons.append(f"3M momentum +{mom_3m:.1f}%")
    if hi52w < 0.90:
        swing_score += 1; swing_reasons.append("Below 52W high — room to run")
    if vix_regime in ("NORMAL", "LOW_VOLATILITY"):
        swing_score += 1; swing_reasons.append("Calm market regime")

    horizons.append({
        "horizon":     "Swing",
        "icon":        "🔄",
        "period":      "1–3 weeks",
        "suitable":    swing_score >= 4,
        "confidence":  "High" if swing_score >= 6 else "Medium" if swing_score >= 3 else "Low",
        "reasons":     swing_reasons[:3],
        "color":       "buy" if swing_score >= 4 else "dim",
    })

    # ── Long-term: months+ ───────────────────────────────────────────────────
    long_score = 0
    long_reasons = []
    if "UP" in monthly_trend:
        long_score += 3; long_reasons.append(f"Monthly trend: {monthly_trend}")
    if regime == 1:
        long_score += 2; long_reasons.append("Bull market regime (Nifty above 200MA)")
    if fund_score > 0.60:
        long_score += 2; long_reasons.append(f"Strong fundamentals (score {fund_score:.2f})")
    if ma_cross > 0 and mom_3m > 0:
        long_score += 1; long_reasons.append("Positive trend + momentum")
    if vix_regime == "LOW_VOLATILITY":
        long_score += 1; long_reasons.append("Low volatility — stable environment")

    horizons.append({
        "horizon":     "Long-term",
        "icon":        "🏦",
        "period":      "Months+",
        "suitable":    long_score >= 4,
        "confidence":  "High" if long_score >= 7 else "Medium" if long_score >= 4 else "Low",
        "reasons":     long_reasons[:3],
        "color":       "buy" if long_score >= 4 else "dim",
    })

    # ── Best recommendation ───────────────────────────────────────────────────
    suitable = [h for h in horizons if h["suitable"]]
    best     = max(horizons, key=lambda h: (h["suitable"], h.get("confidence","Low") == "High"))

    return {
        "horizons":    horizons,
        "recommended": [h["horizon"] for h in suitable],
        "primary":     best["horizon"] if suitable else "Short-term",
        "summary":     f"Best suited for {', '.join(h['horizon'] for h in suitable)}" if suitable
                       else "Mixed signals — no strong horizon bias currently",
    }

@app.get("/live")
def predict_live(ticker: str):
    # ── Step 1: Fetch stock data ONCE — shared across all engines ─────────────
    df = yf.download(ticker, period="14mo", interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or df.empty or len(df) < 60:
        raise HTTPException(status_code=404, detail=f"No data found for '{ticker}'.")

    # ── Step 2: Compute technical features ───────────────────────────────────
    try:
        feats = _live_features(ticker, df=df)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # ── Step 3: XGBoost prediction ────────────────────────────────────────────
    try:
        ml_result = _run_predict(ticker, feats)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # ── Step 4: Run all parallel engines concurrently ─────────────────────────
    from engines import mean_reversion, multi_timeframe, sentiment
    from engines import volatility_regime, fundamental_rank, fusion
    from engines import hmm_regime, sector_correlation, leader_lagger
    from engines import nse_data

    def _safe(fn, *args, **kwargs):
        try:    return fn(*args, **kwargs)
        except Exception as e:
            name = getattr(fn, "__module__", "unknown").split(".")[-1]
            print(f"⚠  Engine {name} failed: {e}")
            return {"engine": name, "signal": "NEUTRAL", "score": 0.5,
                    "detail": str(e)}

    with ThreadPoolExecutor(max_workers=8) as ex:
        fut_mr   = ex.submit(_safe, mean_reversion.run, ticker, df)
        fut_mtf  = ex.submit(_safe, multi_timeframe.run, ticker, df)
        fut_sen  = ex.submit(_safe, sentiment.run, ticker)
        fut_vix  = ex.submit(_safe, volatility_regime.run)
        fut_fun  = ex.submit(_safe, fundamental_rank.run, ticker,
                             SECTOR_MAP, SECTOR_BENCHMARKS)
        fut_hmm  = ex.submit(_safe, hmm_regime.run, df, ticker)
        fut_sec  = ex.submit(_safe, sector_correlation.run, ticker,
                             df, SECTOR_MAP)
        fut_ll   = ex.submit(_safe, leader_lagger.run, ticker)
        fut_nse  = ex.submit(_safe, nse_data.fetch_all)

        mr_res   = fut_mr.result(timeout=15)
        mtf_res  = fut_mtf.result(timeout=15)
        sen_res  = fut_sen.result(timeout=15)
        vix_res  = fut_vix.result(timeout=15)
        fun_res  = fut_fun.result(timeout=15)
        hmm_res  = fut_hmm.result(timeout=20)
        sec_res  = fut_sec.result(timeout=20)
        ll_res   = fut_ll.result(timeout=20)
        nse_res  = fut_nse.result(timeout=20)

    # Attach engine name for fusion
    xgb_for_fusion = {
        "engine": "xgboost",
        "signal": ml_result["signal"],
        "score":  ml_result["probability"] / 100,
        "detail": f"XGBoost prob {ml_result['probability']}%",
    }
    # Tag sector_correlation engine name
    sec_res["engine"] = "sector_corr"
    ll_res["engine"]  = "leader_lagger"

    # ── Step 5: Fuse all signals ──────────────────────────────────────────────
    vix_multiplier  = vix_res.get("confidence_multiplier", 1.0)
    current_regime  = hmm_res.get("regime", "UNKNOWN")
    consensus = fusion.fuse(
        [xgb_for_fusion, mr_res, mtf_res, sen_res, fun_res, sec_res, ll_res],
        vix_multiplier    = vix_multiplier,
        hmm_regime_result = hmm_res,
        current_regime    = current_regime,
    )
    consensus["vix_engine"] = vix_res
    consensus["hmm_regime"] = hmm_res

    # ── Step 5c: Inject live NSE data into feats ───────────────────────────
    try:
        pcr_data    = nse_res.get("pcr", {})
        fii_data    = nse_res.get("fii_dii", {})
        brd_data    = nse_res.get("breadth", {})

        pcr_val  = float(pcr_data.get("pcr", 1.0))
        # PCR z-score vs recent (use cache history if available)
        pcr_sig  = float(np.clip((pcr_val - 1.0) / 0.25, -2, 2))  # normalised signal

        feats["PCR"]          = pcr_val
        feats["PCR_Signal"]   = pcr_sig
        feats["FII_Net_Norm"] = float(fii_data.get("fii_normalised", 0.0))
        feats["DII_Net_Norm"] = float(fii_data.get("dii_normalised", 0.0))
        feats["Breadth_Pct"]  = float(brd_data.get("breadth_pct", 50.0))
        feats["AdvDec_Ratio"] = float(brd_data.get("adv_pct", 50.0))
    except Exception as _ne:
        print(f"⚠  NSE feature injection failed: {_ne}")

    # ── Step 5b: Record signals for performance tracking ──────────────────────
    try:
        from engines.performance_tracker import record_signals
        record_signals(
            ticker          = ticker,
            engine_results  = [xgb_for_fusion, mr_res, mtf_res,
                                sen_res, fun_res, sec_res, ll_res],
            regime          = current_regime,
            price           = feats.get("_last_price", 0),
        )
    except Exception as _pe:
        print(f"⚠  Performance tracking failed: {_pe}")

    # ── Step 6: Register stock + auto-retrain if new ──────────────────────────
    is_new = _register_stock(ticker)
    if is_new:
        _schedule_auto_retrain()

    # ── Step 7: SHAP explanation ──────────────────────────────────────────────
    shap_result = []
    try:
        from ml.explain import explain_prediction
        model_used, feat_list, _ = _get_model_for_ticker(ticker)
        clean_feats = {k: v for k, v in feats.items() if not k.startswith("_")}
        shap_result = explain_prediction(clean_feats, model_used, feat_list)
    except Exception as exc:
        print(f"⚠  SHAP failed: {exc}")

    # ── Step 8: Trading Horizon ──────────────────────────────────────────────
    try:
        horizon = _compute_trading_horizon(feats, mtf_res, fun_res, vix_res)
    except Exception as exc:
        print(f"⚠  Horizon failed: {exc}")
        horizon = {"horizons": [], "recommended": [], "primary": "Short-term",
                   "summary": "Could not compute"}

    # ── Step 9: Trade levels (entry, stop, targets) ─────────────────────────
    try:
        trade_levels = _compute_trade_levels(
            feats, consensus, ml_result.get("signal","NEUTRAL"), horizon
        )
    except Exception as exc:
        print(f"⚠  Trade levels failed: {exc}")
        trade_levels = None

    # ── Step 10: Fundamentals (re-use yf.info already fetched in fundamental engine) ─
    fundamentals = None
    try:
        fundamentals = _compute_fundamentals(ticker, feats)
    except Exception as exc:
        print(f"⚠  Fundamentals failed: {exc}")

    # ── Build response ────────────────────────────────────────────────────────
    return {
        # ML signal (from XGBoost alone)
        **ml_result,
        # Fusion signal (all engines combined) — this is the primary signal
        "consensus":      consensus,
        # Data
        "features":       {k: v for k, v in feats.items() if not k.startswith("_")},
        "last_price":     feats["_last_price"],
        "as_of":          feats["_as_of"],
        # Individual engine results (for display)
        "engines": {
            "mean_reversion":    mr_res,
            "multi_timeframe":   mtf_res,
            "sentiment":         sen_res,
            "volatility_regime": vix_res,
            "fundamental_rank":  fun_res,
            "hmm_regime":        hmm_res,
            "sector_corr":       sec_res,
            "leader_lagger":     ll_res,
        },
        "nse_data": nse_res,
        # Extras
        "shap":           shap_result,
        "trade_levels":   trade_levels,
        "fundamentals":   fundamentals,
        "horizon":        horizon,
        "new_stock":      is_new,
        "learning":       is_new or _learning,
        "total_stocks":   len(_load_known_stocks()),
    }



@app.get("/status")
def server_status():
    """Real-time server state for the notification system."""
    return {
        "ready":       _model_ready,
        "learning":    _learning,
        "retraining":  _retraining,
        "n_stocks":    len(_load_known_stocks()),
        "model_type":  "per-stock" if _stock_models else "global",
        "trained_at":  _metadata.get("trained_at","—") if _metadata else "—",
    }


@app.get("/predictions")
def prediction_history(limit: int = 100):
    """Return recent prediction history with outcomes for the prediction log panel."""
    try:
        from engines.performance_tracker import _load
        data     = _load()
        signals  = data.get("signals", [])
        # Return most recent first
        recent   = list(reversed(signals[-limit:]))
        # Summarise for UI
        total    = len(signals)
        resolved = [s for s in signals if s.get("outcome") in ("CORRECT","WRONG")]
        correct  = sum(1 for s in resolved if s["outcome"]=="CORRECT")
        accuracy = round(correct/len(resolved)*100,1) if resolved else None
        return {
            "predictions":  recent,
            "total":        total,
            "resolved":     len(resolved),
            "correct":      correct,
            "accuracy":     accuracy,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/performance")
def engine_performance():
    """Return rolling accuracy stats for all engines."""
    try:
        from engines.performance_tracker import get_performance_summary, resolve_outcomes
        # Try to resolve any pending outcomes first
        def _price_fetcher(t):
            try:
                df = yf.download(t, period="5d", interval="1d",
                                 auto_adjust=True, progress=False)
                return float(df["Close"].iloc[-1]) if not df.empty else None
            except Exception:
                return None
        resolve_outcomes(_price_fetcher)
        return get_performance_summary()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/backtest")
def backtest_ticker(ticker: str, period: str = "3y"):
    """
    Run a historical backtest for a ticker.
    period: 1y | 2y | 3y | 5y
    """
    if period not in ("1y","2y","3y","5y"):
        raise HTTPException(status_code=400, detail="period must be 1y, 2y, 3y, or 5y")
    if _global_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        from backtest.engine import run_backtest
        result = run_backtest(
            ticker       = ticker,
            model        = _global_model,
            label_encoder= _label_encoder,
            metadata     = _metadata,
            sector_map   = SECTOR_MAP,
            period       = period,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Backtest error: {exc}")

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
