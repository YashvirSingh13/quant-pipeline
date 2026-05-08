"""
engines/eps_data.py — EPS Surprise & Promoter Holdings Engine.

Scrapes Screener.in for:
  1. Quarterly EPS data → computes EPS growth QoQ (surprise proxy)
  2. Shareholding pattern → promoter holding % change QoQ

Both signals are slow-moving (quarterly) but highly predictive:
  - EPS growing QoQ = improving business momentum
  - Promoter increasing stake = management confidence = bullish
  - FII increasing holding = institutional accumulation = bullish

Usage: scraped once per day per ticker and cached.
Falls back to neutral (0) if Screener.in is unavailable.
"""

import time
import re
import requests
import numpy as np

ENGINE_NAME = "eps_data"

_cache: dict = {}
CACHE_TTL = 24 * 3600   # 24 hours — quarterly data rarely changes daily

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept":          "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.screener.in/",
}


def _ticker_to_screener(ticker: str) -> str:
    """Convert NSE ticker to screener.in company slug."""
    sym = ticker.replace(".NS", "").replace(".BO", "").upper()
    # Handle common name differences
    mapping = {
        "M&M":      "M-M",
        "BAJAJ-AUTO": "BAJAJ-AUTO",
        "HDFCBANK":   "HDFC-Bank",
    }
    return mapping.get(sym, sym)


def fetch_eps_data(ticker: str) -> dict:
    """
    Fetch EPS and shareholding data from Screener.in.

    Returns:
        eps_growth:       QoQ EPS growth % (latest vs previous quarter)
        eps_surprise:     Same as eps_growth — proxy for surprise
        promoter_chg:     Promoter holding change QoQ (percentage points)
        fii_holding_chg:  FII holding change QoQ (percentage points)
        signal:           BUY / SELL / NEUTRAL
        detail:           Human-readable summary
    """
    now = time.time()
    cache_key = f"eps_{ticker}"
    if cache_key in _cache and (now - _cache[cache_key]["ts"]) < CACHE_TTL:
        return _cache[cache_key]["data"]

    slug = _ticker_to_screener(ticker)

    try:
        session = requests.Session()
        session.headers.update(_HEADERS)

        url = f"https://www.screener.in/company/{slug}/consolidated/"
        resp = session.get(url, timeout=12)

        if resp.status_code == 404:
            # Try standalone (non-consolidated)
            url = f"https://www.screener.in/company/{slug}/"
            resp = session.get(url, timeout=12)

        if resp.status_code != 200:
            raise ValueError(f"HTTP {resp.status_code}")

        html = resp.text

        # ── Extract quarterly EPS ─────────────────────────────────────────────
        # Look for "Earnings per share" row in quarterly results table
        eps_match = re.findall(
            r'Earnings per share[^<]*</td>\s*(?:<td[^>]*>([0-9.\-]+)</td>\s*){2,6}',
            html, re.IGNORECASE
        )
        eps_values = re.findall(
            r'<td[^>]*>\s*([0-9.\-]+)\s*</td>',
            html[html.find("Earnings per share"):html.find("Earnings per share")+500]
        ) if "Earnings per share" in html else []

        eps_growth = 0.0
        if len(eps_values) >= 2:
            try:
                latest  = float(eps_values[0])
                prev    = float(eps_values[1])
                if abs(prev) > 0.01:
                    eps_growth = round((latest - prev) / abs(prev) * 100, 2)
            except (ValueError, ZeroDivisionError):
                pass

        # ── Extract shareholding ─────────────────────────────────────────────
        promoter_chg = 0.0
        fii_chg      = 0.0

        if "Promoters" in html:
            # Find promoter holding percentages (last 2 quarters)
            prom_section = html[html.find("Promoters"):html.find("Promoters")+800]
            prom_vals = re.findall(r'(\d+\.?\d*)\s*%?', prom_section)
            if len(prom_vals) >= 2:
                try:
                    promoter_chg = round(float(prom_vals[0]) - float(prom_vals[1]), 2)
                except (ValueError, IndexError):
                    pass

        if "FII" in html or "Foreign" in html:
            fii_section_start = max(html.find("FII"), html.find("Foreign"))
            fii_section = html[fii_section_start:fii_section_start+600]
            fii_vals = re.findall(r'(\d+\.?\d*)\s*%?', fii_section)
            if len(fii_vals) >= 2:
                try:
                    fii_chg = round(float(fii_vals[0]) - float(fii_vals[1]), 2)
                except (ValueError, IndexError):
                    pass

        # ── Signal ────────────────────────────────────────────────────────────
        score = 0
        reasons = []

        if eps_growth > 20:
            score += 2; reasons.append(f"EPS +{eps_growth:.1f}% QoQ")
        elif eps_growth > 5:
            score += 1; reasons.append(f"EPS +{eps_growth:.1f}% QoQ")
        elif eps_growth < -20:
            score -= 2; reasons.append(f"EPS {eps_growth:.1f}% QoQ")
        elif eps_growth < -5:
            score -= 1; reasons.append(f"EPS {eps_growth:.1f}% QoQ")

        if promoter_chg > 0.5:
            score += 1; reasons.append(f"Promoter buying +{promoter_chg:.1f}pp")
        elif promoter_chg < -0.5:
            score -= 1; reasons.append(f"Promoter selling {promoter_chg:.1f}pp")

        if fii_chg > 0.5:
            score += 1; reasons.append(f"FII adding +{fii_chg:.1f}pp")
        elif fii_chg < -0.5:
            score -= 1; reasons.append(f"FII reducing {fii_chg:.1f}pp")

        if score >= 2:
            signal = "BUY"
        elif score <= -2:
            signal = "SELL"
        else:
            signal = "NEUTRAL"

        detail = ", ".join(reasons) if reasons else "No significant fundamental change"

        result = {
            "eps_growth":      eps_growth,
            "eps_surprise":    eps_growth,  # proxy — growth vs prior quarter
            "promoter_chg":    promoter_chg,
            "fii_holding_chg": fii_chg,
            "signal":          signal,
            "detail":          detail,
            "source":          "screener.in",
        }
        _cache[cache_key] = {"ts": now, "data": result}
        return result

    except Exception as e:
        fallback = {
            "eps_growth": 0.0, "eps_surprise": 0.0,
            "promoter_chg": 0.0, "fii_holding_chg": 0.0,
            "signal": "NEUTRAL", "detail": f"Screener unavailable: {e}",
            "source": "fallback",
        }
        _cache[cache_key] = {"ts": now - CACHE_TTL + 3600, "data": fallback}
        return fallback
