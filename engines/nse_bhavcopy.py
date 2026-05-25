"""
engines/nse_bhavcopy.py — NSE daily Bhavcopy reader.

Fetches the official "Securities Bhavcopy" from NSE archives, which contains
real DELIVERY DATA (DELIV_QTY, DELIV_PER) — the percentage of traded volume
that was settled by actual delivery rather than intraday speculation.

Why it matters:
  • Real delivery % is a strong signal of institutional / long-term interest.
  • Currently the system uses a volume/ATR-based proxy (Delivery_Pct_Proxy).
    Real Bhavcopy data replaces guesswork with actual settlement data.

Approach:
  • Tries today's bhavcopy first; falls back to most recent trading day.
  • Caches each day's CSV on disk in DATA_DIR/bhavcopy_cache/.
  • Falls back gracefully if NSE blocks or returns errors — caller can detect
    None and use the proxy instead.

Reference URL format (current):
  https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_DDMMYYYY.csv
"""

import os
import time
import requests
from io import StringIO
from datetime import datetime, timedelta
import pandas as pd

ENGINE_NAME = "nse_bhavcopy"

DATA_DIR  = os.environ.get("DATA_DIR",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
CACHE_DIR = os.path.join(DATA_DIR, "bhavcopy_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# In-memory cache (one Bhavcopy is ~1MB CSV — keep latest 3 days)
_mem_cache: dict = {}   # date_str -> pd.DataFrame
_lookup_failed: set = set()  # date_strs that already failed — don't retry within session

# Network throttle: don't hammer NSE
_last_fetch_ts = 0
MIN_FETCH_INTERVAL = 2.0   # seconds between fetches

_NSE_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
    "Accept":          "text/csv, application/json, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/all-reports",
}


def _nse_session() -> requests.Session:
    """Create an NSE session with required cookies."""
    s = requests.Session()
    s.headers.update(_NSE_HEADERS)
    try:
        s.get("https://www.nseindia.com/", timeout=8)
        time.sleep(0.3)
    except Exception:
        pass
    return s


def _cache_path(date_str: str) -> str:
    """date_str format: DDMMYYYY (e.g. '15052026')"""
    return os.path.join(CACHE_DIR, f"sec_bhavdata_full_{date_str}.csv")


def _load_cached(date_str: str) -> pd.DataFrame | None:
    """Load Bhavcopy from disk cache, return None if not present or invalid."""
    if date_str in _mem_cache:
        return _mem_cache[date_str]
    path = _cache_path(date_str)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        # Clean column names (NSE sometimes adds leading whitespace)
        df.columns = [c.strip() for c in df.columns]
        _mem_cache[date_str] = df
        return df
    except Exception:
        return None


def _fetch_bhavcopy(date_str: str) -> pd.DataFrame | None:
    """Download bhavcopy for given date (DDMMYYYY). Returns DataFrame or None."""
    global _last_fetch_ts
    if date_str in _lookup_failed:
        return None

    # Disk cache check first
    cached = _load_cached(date_str)
    if cached is not None:
        return cached

    # Throttle
    wait = (_last_fetch_ts + MIN_FETCH_INTERVAL) - time.time()
    if wait > 0:
        time.sleep(wait)

    url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date_str}.csv"
    try:
        s = _nse_session()
        r = s.get(url, timeout=15)
        _last_fetch_ts = time.time()
        if r.status_code != 200 or not r.text or "SYMBOL" not in r.text[:200]:
            _lookup_failed.add(date_str)
            return None

        # Persist to disk cache
        with open(_cache_path(date_str), "w") as f:
            f.write(r.text)

        df = pd.read_csv(StringIO(r.text))
        df.columns = [c.strip() for c in df.columns]
        _mem_cache[date_str] = df
        print(f"📥 Bhavcopy {date_str}: {len(df)} rows cached")
        return df
    except Exception as e:
        print(f"⚠  Bhavcopy fetch failed for {date_str}: {e}")
        _lookup_failed.add(date_str)
        return None


def _latest_trading_date(max_lookback: int = 7) -> pd.DataFrame | None:
    """Return Bhavcopy for the most recent trading day available.

    Tries today, then yesterday, etc. Stops at first success or after
    `max_lookback` attempts.
    """
    today = datetime.now()
    for back in range(max_lookback):
        d = today - timedelta(days=back)
        # Skip weekends (Sat=5, Sun=6)
        if d.weekday() >= 5:
            continue
        date_str = d.strftime("%d%m%Y")
        df = _fetch_bhavcopy(date_str)
        if df is not None:
            return df
    return None


# ── Public API ────────────────────────────────────────────────────────────────
def get_delivery_data(ticker: str) -> dict | None:
    """Return delivery data for a ticker from the most recent Bhavcopy.

    `ticker` can be e.g. "RELIANCE.NS" or "RELIANCE". Returns dict:
      {
        "symbol":            "RELIANCE",
        "close":             1432.50,
        "delivery_qty":      1234567,
        "total_qty":         3456789,
        "delivery_pct":      35.71,     # raw %
        "delivery_pct_norm": 0.71,      # normalised to 0-2 (50% → 1.0)
        "date":              "15052026",
      }

    Returns None if Bhavcopy is unreachable or symbol not found.
    Callers should fall back to the proxy in that case.
    """
    if not ticker:
        return None

    # Strip exchange suffix — Bhavcopy uses bare symbols
    sym = ticker.split(".")[0].upper().strip()

    df = _latest_trading_date()
    if df is None:
        return None

    # Lookup the symbol (only EQ series for cash equity)
    try:
        if "SERIES" in df.columns:
            row = df[(df["SYMBOL"].str.strip() == sym) &
                     (df["SERIES"].str.strip() == "EQ")]
        else:
            row = df[df["SYMBOL"].str.strip() == sym]
        if row.empty:
            return None
        r = row.iloc[0]

        total_qty    = float(r.get("TTL_TRD_QNTY", 0) or 0)
        deliv_qty    = float(r.get("DELIV_QTY", 0) or 0)
        deliv_pct    = float(r.get("DELIV_PER", 0) or 0)
        if total_qty > 0 and deliv_qty > 0 and deliv_pct == 0:
            deliv_pct = (deliv_qty / total_qty) * 100.0

        # Normalise so 50% delivery → 1.0 (matches proxy scale 0-2)
        norm = max(0.0, min(2.0, deliv_pct / 50.0))

        date_str = ""
        if "DATE1" in df.columns:
            date_str = str(r.get("DATE1", "")).strip()

        return {
            "symbol":            sym,
            "close":             float(r.get("CLOSE_PRICE", 0) or 0),
            "delivery_qty":      deliv_qty,
            "total_qty":         total_qty,
            "delivery_pct":      round(deliv_pct, 2),
            "delivery_pct_norm": round(norm, 4),
            "date":              date_str,
        }
    except Exception as e:
        print(f"⚠  Bhavcopy parse error for {sym}: {e}")
        return None


def get_delivery_signal(ticker: str, recent_return: float = 0.0) -> dict:
    """Convert delivery data + recent return into a BUY/SELL/NEUTRAL signal.

    Uses settled-volume interpretation:
      • High delivery % (>60%) + price up   → BUY  (institutional accumulation)
      • High delivery % (>60%) + price down → SELL (institutional distribution)
      • Low delivery % (<30%) + price up    → NEUTRAL (speculative pop, no conviction)
      • Low delivery % (<30%) + price down  → NEUTRAL (panic, may bounce)
      • Mid-range                            → NEUTRAL

    Returns dict matching other engines' shape:
      {engine, signal, score, detail, display_name}
    """
    data = get_delivery_data(ticker)
    if data is None:
        return {
            "engine":       ENGINE_NAME,
            "display_name": "Delivery %",
            "signal":       "NEUTRAL",
            "score":        0.5,
            "detail":       "Bhavcopy unavailable",
        }

    pct = data["delivery_pct"]

    if pct >= 60 and recent_return > 0:
        signal = "BUY";  score = 0.70
        detail = f"Strong delivery {pct:.0f}% + price up — institutional accumulation"
    elif pct >= 60 and recent_return < 0:
        signal = "SELL"; score = 0.30
        detail = f"Strong delivery {pct:.0f}% + price down — institutional distribution"
    elif pct < 30:
        signal = "NEUTRAL"; score = 0.50
        detail = f"Weak delivery {pct:.0f}% — speculative move, low conviction"
    else:
        signal = "NEUTRAL"; score = 0.50
        detail = f"Delivery {pct:.0f}% — normal range"

    return {
        "engine":       ENGINE_NAME,
        "display_name": "Delivery %",
        "signal":       signal,
        "score":        score,
        "detail":       detail,
        "delivery_pct": pct,
    }
