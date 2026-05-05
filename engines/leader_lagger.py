"""
engines/leader_lagger.py — Leader-Lagger Relationship Engine.

Pre-computes a leader matrix during retrain: for every stock pair,
calculates the rolling correlation between stock A's return today
and stock B's return tomorrow (lead-lag lag=1).

Leader matrix stored at: DATA_DIR/leader_matrix.json
{
  "WIPRO.NS": {
    "leaders": ["TCS.NS", "INFY.NS"],     # these stocks lead WIPRO
    "followers": ["MPHASIS.NS"],           # WIPRO leads these
    "computed_at": "2026-01-01"
  }
}

Live signal logic:
  1. Look up leaders for the target ticker
  2. Fetch today's returns for each leader
  3. If leaders are strongly up/down → generate early BUY/SELL signal
  4. Weight by the strength of the historical lead-lag correlation
"""

import os
import json
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

ENGINE_NAME = "leader_lagger"

DATA_DIR    = os.environ.get("DATA_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
MATRIX_FILE = os.path.join(DATA_DIR, "leader_matrix.json")

CORR_THRESHOLD  = 0.20    # minimum lead-lag correlation to consider a leader
SIGNAL_THRESH   = 0.005   # 0.5% move in leader to count as signal
BUY_SCORE_THRESH  = 0.60
SELL_SCORE_THRESH = 0.40

# Simple in-memory cache for leader returns (60 min)
_leader_cache: dict = {}
CACHE_TTL = 3600


# ── Matrix I/O ────────────────────────────────────────────────────────────────
def load_matrix() -> dict:
    if os.path.exists(MATRIX_FILE):
        try:
            with open(MATRIX_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_matrix(matrix: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MATRIX_FILE, "w") as f:
        json.dump(matrix, f, indent=2)


# ── Build leader matrix (called during retrain) ────────────────────────────────
def build_leader_matrix(stocks: list, period: str = "2y") -> dict:
    """
    Download returns for all stocks and compute lead-lag correlations.
    Called from ml/train.py after model training.

    Returns the matrix dict (also saves to disk).
    """
    print(f"📊 Building leader-lagger matrix for {len(stocks)} stocks…")

    # Download all at once
    try:
        raw = yf.download(stocks, period=period, interval="1d",
                          auto_adjust=True, progress=False)
        if raw.empty:
            print("⚠  No data for leader matrix")
            return {}
        prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    except Exception as e:
        print(f"⚠  Leader matrix download failed: {e}")
        return {}

    returns = prices.pct_change().dropna()

    matrix = {}
    tickers = [c for c in returns.columns if c in stocks]

    for i, ticker in enumerate(tickers):
        if ticker not in returns.columns:
            continue

        target      = returns[ticker]
        target_next = target.shift(-1)   # target's NEXT day return

        leaders   = []
        followers = []

        for other in tickers:
            if other == ticker or other not in returns.columns:
                continue

            other_ret = returns[other]
            aligned   = pd.concat([other_ret, target_next], axis=1).dropna()
            if len(aligned) < 60:
                continue

            # Lead-lag: does OTHER today predict TARGET tomorrow?
            corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))

            if not np.isnan(corr) and corr >= CORR_THRESHOLD:
                leaders.append({"ticker": other, "corr": round(corr, 3)})

            # Does TARGET today predict OTHER tomorrow?
            aligned2  = pd.concat([target, returns[other].shift(-1)], axis=1).dropna()
            if len(aligned2) >= 60:
                corr2 = float(aligned2.iloc[:, 0].corr(aligned2.iloc[:, 1]))
                if not np.isnan(corr2) and corr2 >= CORR_THRESHOLD:
                    followers.append({"ticker": other, "corr": round(corr2, 3)})

        # Keep top 5 leaders and top 5 followers by correlation strength
        leaders.sort(key=lambda x: x["corr"], reverse=True)
        followers.sort(key=lambda x: x["corr"], reverse=True)

        matrix[ticker] = {
            "leaders":      leaders[:5],
            "followers":    followers[:5],
            "computed_at":  datetime.now().strftime("%Y-%m-%d"),
        }

        if i % 10 == 0:
            print(f"   {i+1}/{len(tickers)} processed…")

    save_matrix(matrix)
    print(f"✅ Leader matrix saved — {len(matrix)} stocks mapped")
    return matrix


# ── Live signal ────────────────────────────────────────────────────────────────
def _fetch_leader_returns(leaders: list) -> dict:
    """Fetch today's return for each leader stock."""
    results = {}
    now = time.time()

    for l in leaders:
        ticker = l["ticker"]
        cache_key = f"ll_{ticker}"

        if cache_key in _leader_cache and (now - _leader_cache[cache_key]["ts"]) < CACHE_TTL:
            results[ticker] = _leader_cache[cache_key]["ret"]
            continue

        try:
            df = yf.download(ticker, period="5d", interval="1d",
                             auto_adjust=True, progress=False)
            if len(df) >= 2:
                closes = df["Close"].squeeze()
                ret = float((closes.iloc[-1] / closes.iloc[-2]) - 1)
                _leader_cache[cache_key] = {"ts": now, "ret": ret}
                results[ticker] = ret
        except Exception:
            results[ticker] = 0.0

    return results


def run(ticker: str) -> dict:
    """
    Generate lead-lag signal for a ticker based on its leaders' current moves.

    Returns dict:
        engine, signal, score, leaders_up, leaders_down, weighted_signal,
        leader_data, detail
    """
    try:
        matrix = load_matrix()

        if ticker not in matrix:
            return {
                "engine":  ENGINE_NAME,
                "signal":  "NEUTRAL",
                "score":   0.5,
                "detail":  f"No leader data for {ticker} — run retrain to build matrix",
                "leaders": [],
            }

        entry   = matrix[ticker]
        leaders = entry.get("leaders", [])

        if not leaders:
            return {
                "engine":  ENGINE_NAME,
                "signal":  "NEUTRAL",
                "score":   0.5,
                "detail":  f"No leaders identified for {ticker}",
                "leaders": [],
            }

        # Fetch today's returns for each leader
        leader_returns = _fetch_leader_returns(leaders)

        # Compute weighted signal
        total_weight   = 0.0
        weighted_signal = 0.0
        leaders_up     = []
        leaders_down   = []
        leader_data    = []

        for l in leaders:
            lticker = l["ticker"]
            corr    = l["corr"]
            ret     = leader_returns.get(lticker, 0.0)

            # Weighted contribution: correlation × leader's return today
            contrib = corr * ret
            weighted_signal += contrib
            total_weight    += corr

            if ret > SIGNAL_THRESH:
                leaders_up.append({"ticker": lticker, "return_pct": round(ret*100, 2), "corr": corr})
            elif ret < -SIGNAL_THRESH:
                leaders_down.append({"ticker": lticker, "return_pct": round(ret*100, 2), "corr": corr})

            leader_data.append({
                "ticker":     lticker,
                "return_pct": round(ret * 100, 2),
                "corr":       corr,
                "direction":  "UP" if ret > 0 else "DOWN",
            })

        # Normalise: weighted_signal / total_weight gives average corr-weighted return
        norm_signal = weighted_signal / total_weight if total_weight > 0 else 0.0

        # Convert to 0–1 score
        # norm_signal of +0.01 (leaders up 1%) → score ~0.70
        # norm_signal of -0.01 (leaders down 1%) → score ~0.30
        score = 0.5 + norm_signal * 20
        score = max(0.05, min(0.95, score))

        if score > BUY_SCORE_THRESH:
            signal = "BUY"
            n_up   = len(leaders_up)
            detail = (f"{n_up}/{len(leaders)} leaders rising today "
                      f"(avg corr {total_weight/len(leaders):.2f}) "
                      f"— early BUY signal for {ticker}")
        elif score < SELL_SCORE_THRESH:
            signal = "SELL"
            n_dn   = len(leaders_down)
            detail = (f"{n_dn}/{len(leaders)} leaders falling today "
                      f"(avg corr {total_weight/len(leaders):.2f}) "
                      f"— early SELL signal for {ticker}")
        else:
            signal = "NEUTRAL"
            detail = (f"Leaders mixed: {len(leaders_up)} up, {len(leaders_down)} down "
                      f"— no strong lead-lag signal")

        return {
            "engine":          ENGINE_NAME,
            "signal":          signal,
            "score":           round(score, 3),
            "leaders_up":      leaders_up,
            "leaders_down":    leaders_down,
            "weighted_signal": round(norm_signal * 100, 3),
            "leader_data":     leader_data,
            "n_leaders":       len(leaders),
            "detail":          detail,
        }

    except Exception as exc:
        return {
            "engine":  ENGINE_NAME,
            "signal":  "NEUTRAL",
            "score":   0.5,
            "detail":  f"Error: {exc}",
            "leaders": [],
        }
