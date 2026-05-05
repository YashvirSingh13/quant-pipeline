"""
engines/volatility_regime.py — Volatility Regime Engine.

Uses India VIX (^INDIAVIX) to classify the current market regime.
High VIX = fear/uncertainty → suppress buy signals.
Low VIX  = calm/risk-on   → amplify buy signals.

VIX thresholds (calibrated for Indian market):
  < 13   : Low volatility   — risk-on,  BUY bias
  13–18  : Normal           — neutral
  18–24  : Elevated         — cautious, NEUTRAL
  > 24   : High volatility  — risk-off, SELL bias
  > 30   : Crisis           — strong SELL

confidence_multiplier: applied to all other engine scores in fusion.
  Low VIX  → 1.10 (amplify signals)
  Normal   → 1.00
  Elevated → 0.80 (dampen signals)
  High     → 0.55
  Crisis   → 0.30 (almost suppress all signals)
"""

import time
import yfinance as yf

ENGINE_NAME = "volatility_regime"

# ── Simple in-memory cache (VIX doesn't change minute-to-minute) ─────────────
_vix_cache: dict = {"ts": 0, "data": None}
VIX_CACHE_TTL = 4 * 3600   # 4 hours


def _fetch_vix():
    global _vix_cache
    now = time.time()
    if _vix_cache["data"] is not None and (now - _vix_cache["ts"]) < VIX_CACHE_TTL:
        return _vix_cache["data"]

    vix = yf.download("^INDIAVIX", period="3mo", interval="1d",
                      auto_adjust=True, progress=False)
    if vix.empty:
        return None

    close = vix["Close"].squeeze()
    data  = {
        "current":   float(close.iloc[-1]),
        "ma20":      float(close.rolling(20).mean().iloc[-1]),
        "ma5":       float(close.rolling(5).mean().iloc[-1]),
        "pct_chg":   float(close.pct_change().iloc[-1] * 100),
        "52w_high":  float(close.max()),
        "52w_low":   float(close.min()),
    }
    _vix_cache = {"ts": now, "data": data}
    return data


def run() -> dict:
    """
    Returns dict:
        engine, signal, score, vix, vix_ma20, regime,
        confidence_multiplier, detail
    """
    try:
        vix_data = _fetch_vix()

        if vix_data is None:
            return {
                "engine": ENGINE_NAME, "signal": "NEUTRAL", "score": 0.5,
                "regime": "unknown", "confidence_multiplier": 1.0,
                "detail": "India VIX data unavailable",
            }

        vix     = vix_data["current"]
        vix_ma  = vix_data["ma20"]
        vix_roc = vix_data["pct_chg"]
        rising  = vix > vix_ma   # VIX rising above its own MA = fear increasing

        # ── Regime classification ────────────────────────────────────────────────
        if vix > 30:
            regime, signal, score, cm = "CRISIS",          "SELL",    0.10, 0.30
            detail = f"India VIX={vix:.1f} — crisis level, suppress all signals"
        elif vix > 24:
            regime, signal, score, cm = "HIGH_VOLATILITY",  "SELL",   0.25, 0.55
            detail = f"India VIX={vix:.1f} — elevated fear, risk-off"
        elif vix > 18:
            if rising:
                regime, signal, score, cm = "ELEVATED_RISING", "NEUTRAL", 0.42, 0.80
                detail = f"India VIX={vix:.1f} rising above MA — increasing caution"
            else:
                regime, signal, score, cm = "ELEVATED",       "NEUTRAL", 0.47, 0.85
                detail = f"India VIX={vix:.1f} — mildly elevated, stay cautious"
        elif vix > 13:
            regime, signal, score, cm = "NORMAL",           "NEUTRAL", 0.52, 1.00
            detail = f"India VIX={vix:.1f} — normal range, no regime bias"
        else:
            regime, signal, score, cm = "LOW_VOLATILITY",   "BUY",    0.70, 1.10
            detail = f"India VIX={vix:.1f} — low fear, risk-on environment"

        return {
            "engine":               ENGINE_NAME,
            "signal":               signal,
            "score":                round(score, 3),
            "vix":                  round(vix,    2),
            "vix_ma20":             round(vix_ma, 2),
            "vix_pct_chg":         round(vix_roc, 2),
            "vix_52w_high":         round(vix_data["52w_high"], 2),
            "vix_52w_low":          round(vix_data["52w_low"],  2),
            "regime":               regime,
            "confidence_multiplier": cm,
            "detail":               detail,
        }

    except Exception as exc:
        return {
            "engine": ENGINE_NAME, "signal": "NEUTRAL", "score": 0.5,
            "regime": "unknown", "confidence_multiplier": 1.0,
            "detail": f"Error: {exc}",
        }
