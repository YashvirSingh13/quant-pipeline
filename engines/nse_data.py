"""
engines/nse_data.py — NSE Official Data Engine.

Fetches from NSE's official APIs (free, legal for personal use):
  1. Put/Call Ratio (PCR) from Nifty option chain JSON
  2. FII/DII net flows from NSE daily report
  3. Nifty advance/decline ratio
  4. Market breadth — % of Nifty 50 stocks above 50-day MA

All values are cached to avoid rate-limiting.
Falls back to neutral defaults if NSE is unavailable.

NSE session note: NSE requires a browser-like session with cookies.
We hit the homepage first to get cookies, then call the API endpoints.
"""

import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime

ENGINE_NAME = "nse_data"

# ── Cache ─────────────────────────────────────────────────────────────────────
_cache: dict = {}
PCR_TTL       = 30 * 60     # 30 min — option chain changes intraday
FII_TTL       = 4  * 3600   # 4 h — FII data is daily
BREADTH_TTL   = 1  * 3600   # 1 h — breadth changes slowly

# ── NSE Session ───────────────────────────────────────────────────────────────
_NSE_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer":         "https://www.nseindia.com/",
    "Origin":          "https://www.nseindia.com",
    "Connection":      "keep-alive",
}


def _nse_session() -> requests.Session:
    """Create a requests session with NSE cookies."""
    s = requests.Session()
    s.headers.update(_NSE_HEADERS)
    try:
        # Hit homepage to get session cookies (required by NSE)
        s.get("https://www.nseindia.com/", timeout=8)
        time.sleep(0.5)
    except Exception:
        pass
    return s


# ── Put/Call Ratio ─────────────────────────────────────────────────────────────
def get_pcr() -> dict:
    """
    Fetch Nifty Put/Call Ratio from NSE option chain.

    PCR > 1.2 = bearish sentiment (too many puts)
    PCR < 0.7 = bullish sentiment (too many calls)
    PCR 0.7–1.2 = neutral

    Returns: {pcr, call_oi, put_oi, signal, detail}
    """
    now = time.time()
    if "pcr" in _cache and (now - _cache["pcr"]["ts"]) < PCR_TTL:
        return _cache["pcr"]["data"]

    try:
        s = _nse_session()
        r = s.get(
            "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY",
            timeout=10
        )
        data = r.json()

        records   = data.get("records", {}).get("data", [])
        call_oi   = sum(rec.get("CE", {}).get("openInterest", 0) for rec in records
                        if "CE" in rec)
        put_oi    = sum(rec.get("PE", {}).get("openInterest", 0) for rec in records
                        if "PE" in rec)
        pcr       = round(put_oi / call_oi, 3) if call_oi > 0 else 1.0

        # IV of ATM call (first available)
        atm_iv = None
        spot = data.get("records", {}).get("underlyingValue", 0)
        if spot:
            atm_strike = round(spot / 50) * 50
            for rec in records:
                if rec.get("strikePrice") == atm_strike and "CE" in rec:
                    atm_iv = rec["CE"].get("impliedVolatility")
                    break

        if pcr > 1.3:
            signal = "SELL"; detail = f"PCR={pcr} — extreme bearish sentiment"
        elif pcr > 1.1:
            signal = "SELL"; detail = f"PCR={pcr} — bearish sentiment"
        elif pcr < 0.65:
            signal = "BUY";  detail = f"PCR={pcr} — bullish sentiment (oversold puts)"
        elif pcr < 0.80:
            signal = "BUY";  detail = f"PCR={pcr} — mildly bullish sentiment"
        else:
            signal = "NEUTRAL"; detail = f"PCR={pcr} — neutral sentiment"

        result = {
            "pcr":      pcr,
            "call_oi":  call_oi,
            "put_oi":   put_oi,
            "atm_iv":   atm_iv,
            "nifty_spot": spot,
            "signal":   signal,
            "detail":   detail,
        }
        _cache["pcr"] = {"ts": now, "data": result}
        return result

    except Exception as e:
        fallback = {"pcr": 1.0, "call_oi": 0, "put_oi": 0,
                    "atm_iv": None, "nifty_spot": 0,
                    "signal": "NEUTRAL", "detail": f"NSE unavailable: {e}"}
        _cache["pcr"] = {"ts": now - PCR_TTL + 60, "data": fallback}  # retry in 1 min
        return fallback


# ── FII / DII Net Flows ────────────────────────────────────────────────────────
def get_fii_dii() -> dict:
    """
    Fetch today's FII and DII net buy/sell from NSE.

    FII net positive = foreign money flowing in = bullish
    FII net negative = foreign money leaving = bearish

    Returns: {fii_net, dii_net, fii_signal, detail}
    """
    now = time.time()
    if "fii" in _cache and (now - _cache["fii"]["ts"]) < FII_TTL:
        return _cache["fii"]["data"]

    try:
        s = _nse_session()
        r = s.get(
            "https://www.nseindia.com/api/fiidiiTradeReact",
            timeout=10
        )
        data = r.json()

        # Data is a list of records; most recent first
        if isinstance(data, list) and len(data) > 0:
            latest = data[0]
            fii_net = float(latest.get("fii_net_turnover", 0) or 0)
            dii_net = float(latest.get("dii_net_turnover", 0) or 0)
            date_str= latest.get("date", "")
        else:
            fii_net = dii_net = 0
            date_str = ""

        # Convert crores to numeric (already in crores from NSE)
        if fii_net > 2000:
            fii_signal = "BUY"; fii_detail = f"Strong FII buying ₹{fii_net:.0f} Cr"
        elif fii_net > 500:
            fii_signal = "BUY"; fii_detail = f"Mild FII buying ₹{fii_net:.0f} Cr"
        elif fii_net < -2000:
            fii_signal = "SELL"; fii_detail = f"Strong FII selling ₹{abs(fii_net):.0f} Cr"
        elif fii_net < -500:
            fii_signal = "SELL"; fii_detail = f"Mild FII selling ₹{abs(fii_net):.0f} Cr"
        else:
            fii_signal = "NEUTRAL"; fii_detail = f"FII neutral ₹{fii_net:.0f} Cr"

        # Normalise for use as model feature (-1 to +1 scale based on ±5000 Cr range)
        fii_norm = float(np.clip(fii_net / 5000, -1, 1))
        dii_norm = float(np.clip(dii_net / 5000, -1, 1))

        result = {
            "fii_net":      round(fii_net, 1),
            "dii_net":      round(dii_net, 1),
            "fii_normalised": round(fii_norm, 4),
            "dii_normalised": round(dii_norm, 4),
            "date":         date_str,
            "signal":       fii_signal,
            "detail":       fii_detail,
        }
        _cache["fii"] = {"ts": now, "data": result}
        return result

    except Exception as e:
        fallback = {"fii_net": 0, "dii_net": 0, "fii_normalised": 0,
                    "dii_normalised": 0, "date": "",
                    "signal": "NEUTRAL", "detail": f"FII data unavailable: {e}"}
        _cache["fii"] = {"ts": now - FII_TTL + 300, "data": fallback}
        return fallback


# ── Market Breadth ─────────────────────────────────────────────────────────────
def get_market_breadth(nifty50_tickers: list = None) -> dict:
    """
    Compute market breadth: % of Nifty 50 stocks above their 50-day MA.

    >70% = broad bull market
    50–70% = mixed
    <50% = broad weakness
    <30% = deeply oversold market — contrarian buy zone

    Returns: {breadth_pct, above_50ma, total, adv_decline, signal, detail}
    """
    now = time.time()
    if "breadth" in _cache and (now - _cache["breadth"]["ts"]) < BREADTH_TTL:
        return _cache["breadth"]["data"]

    if not nifty50_tickers:
        nifty50_tickers = [
            "HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","AXISBANK.NS","KOTAKBANK.NS",
            "RELIANCE.NS","TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS",
            "TATAMOTORS.NS","MARUTI.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS","M&M.NS",
            "HINDUNILVR.NS","ITC.NS","BRITANNIA.NS","NESTLEIND.NS","TATACONSUM.NS",
            "SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","APOLLOHOSP.NS",
            "LT.NS","ADANIPORTS.NS","NTPC.NS","POWERGRID.NS","ULTRACEMCO.NS",
            "TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS",
            "BAJFINANCE.NS","BAJAJFINSV.NS","SBILIFE.NS","HDFCLIFE.NS",
            "ASIANPAINT.NS","TITAN.NS","BHARTIARTL.NS","TECHM.NS","LTIM.NS",
        ]

    try:
        import yfinance as yf
        raw = yf.download(
            nifty50_tickers, period="3mo", interval="1d",
            auto_adjust=True, progress=False
        )
        if raw.empty:
            raise ValueError("No data")

        prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        above_50  = 0
        below_50  = 0
        advancing = 0  # up today
        declining = 0  # down today

        for col in prices.columns:
            s = prices[col].dropna()
            if len(s) < 50:
                continue
            ma50 = s.rolling(50).mean()
            if s.iloc[-1] > ma50.iloc[-1]:
                above_50 += 1
            else:
                below_50 += 1
            # Advance/decline
            if len(s) >= 2:
                if s.iloc[-1] > s.iloc[-2]:
                    advancing += 1
                else:
                    declining += 1

        total      = above_50 + below_50
        breadth    = round(above_50 / total * 100, 1) if total > 0 else 50
        adv_dec    = round(advancing / (advancing + declining) * 100, 1) \
                     if (advancing + declining) > 0 else 50

        if breadth >= 70:
            signal = "BUY";  detail = f"{breadth}% stocks above 50MA — broad bull market"
        elif breadth >= 50:
            signal = "BUY";  detail = f"{breadth}% stocks above 50MA — majority bullish"
        elif breadth >= 35:
            signal = "SELL"; detail = f"{breadth}% stocks above 50MA — majority bearish"
        else:
            signal = "SELL"; detail = f"{breadth}% stocks above 50MA — deeply oversold"

        result = {
            "breadth_pct":  breadth,
            "above_50ma":   above_50,
            "below_50ma":   below_50,
            "total":        total,
            "adv_pct":      adv_dec,
            "advancing":    advancing,
            "declining":    declining,
            "signal":       signal,
            "detail":       detail,
        }
        _cache["breadth"] = {"ts": now, "data": result}
        return result

    except Exception as e:
        fallback = {"breadth_pct": 50, "above_50ma": 0, "below_50ma": 0,
                    "total": 0, "adv_pct": 50, "advancing": 0, "declining": 0,
                    "signal": "NEUTRAL", "detail": f"Breadth unavailable: {e}"}
        _cache["breadth"] = {"ts": now - BREADTH_TTL + 120, "data": fallback}
        return fallback


# ── Combined fetch ─────────────────────────────────────────────────────────────
def fetch_all() -> dict:
    """Fetch all NSE data in one call. Returns combined dict."""
    pcr     = get_pcr()
    fii     = get_fii_dii()
    breadth = get_market_breadth()
    return {
        "pcr":     pcr,
        "fii_dii": fii,
        "breadth": breadth,
    }
