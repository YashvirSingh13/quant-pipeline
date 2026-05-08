"""
engines/sector_rotation.py — Sector Rotation Detection Engine.

Downloads NSE sector indices and computes:
  1. 20-day momentum rank for each sector (which sectors are leading)
  2. Sector relative performance vs Nifty 50
  3. Risk-on/Risk-off classification (cyclicals vs defensives)
  4. Signal for target stock based on its sector's rotation position

Sector indices via Yahoo Finance:
  Banking:  ^NSEBANK    IT:      ^CNXIT
  Pharma:   ^CNXPHARMA  Auto:    ^CNXAUTO
  FMCG:     ^CNXFMCG    Metal:   ^CNXMETAL
  Energy:   ^CNXENERGY  Infra:   ^CNXINFRA

Cached 2 hours — sector rotation is a daily/weekly phenomenon.
"""

import time
import numpy as np
import pandas as pd
import yfinance as yf

ENGINE_NAME = "sector_rotation"

SECTOR_INDICES = {
    0: "^NSEBANK",     # Banking
    1: "^NSEBANK",     # Finance (use Bank as proxy)
    2: "^CNXIT",       # IT
    3: "^CNXAUTO",     # Auto
    4: "^CNXENERGY",   # Energy
    5: "^CNXFMCG",     # FMCG
    6: "^CNXINFRA",    # Infrastructure
    7: "^CNXPHARMA",   # Pharma
    8: "^CNXMETAL",    # Metals
    9: "^CNXFMCG",     # Other (use FMCG as proxy)
}

SECTOR_NAMES = {
    0: "Banking", 1: "Finance", 2: "IT", 3: "Auto",
    4: "Energy",  5: "FMCG",   6: "Infra", 7: "Pharma",
    8: "Metals",  9: "Other",
}

# Cyclical vs defensive classification
CYCLICAL_SECTORS    = {3, 4, 8}   # Auto, Energy, Metals
DEFENSIVE_SECTORS   = {5, 7}      # FMCG, Pharma
GROWTH_SECTORS      = {2, 1}      # IT, Finance

_cache: dict = {}
CACHE_TTL = 2 * 3600  # 2 hours


def _fetch_sector_data() -> dict:
    """Download all sector index data. Cached."""
    now = time.time()
    if "sectors" in _cache and (now - _cache["sectors"]["ts"]) < CACHE_TTL:
        return _cache["sectors"]["data"]

    unique_syms = list(set(SECTOR_INDICES.values())) + ["^NSEI"]

    try:
        raw = yf.download(unique_syms, period="3mo", interval="1d",
                          auto_adjust=True, progress=False)
        if raw.empty:
            return {}

        prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        data = {}
        for sym in unique_syms:
            if sym in prices.columns:
                data[sym] = prices[sym].dropna()

        _cache["sectors"] = {"ts": now, "data": data}
        return data
    except Exception:
        return {}


def run(ticker: str, sector_code: int) -> dict:
    """
    Args:
        ticker      : e.g. "WIPRO.NS"
        sector_code : 0-9 sector code

    Returns dict:
        engine, signal, score, sector_rank, sector_momentum,
        sector_rel_perf, regime_type, detail
    """
    try:
        sector_data = _fetch_sector_data()
        if not sector_data:
            return _neutral(sector_code, "Sector data unavailable")

        nifty_sym = "^NSEI"
        nifty     = sector_data.get(nifty_sym, pd.Series(dtype=float))

        # ── Compute 20-day momentum for all sectors ───────────────────────────
        sector_moms = {}
        for sc, sym in SECTOR_INDICES.items():
            s = sector_data.get(sym, pd.Series(dtype=float))
            if len(s) >= 21:
                mom = float(s.iloc[-1] / s.iloc[-21] - 1) * 100
                sector_moms[sc] = mom

        if not sector_moms:
            return _neutral(sector_code, "Insufficient sector data")

        # ── Rank sectors by momentum (0 = worst, 1 = best) ───────────────────
        unique_sectors = sorted(set(SECTOR_INDICES.keys()))
        vals = [sector_moms.get(sc, 0) for sc in unique_sectors]
        ranks = pd.Series(vals).rank(pct=True)
        rank_map = {sc: round(float(ranks.iloc[i]), 3)
                    for i, sc in enumerate(unique_sectors)}

        target_rank = rank_map.get(sector_code, 0.5)
        target_mom  = sector_moms.get(sector_code, 0)

        # ── Sector relative performance vs Nifty ─────────────────────────────
        target_sym  = SECTOR_INDICES.get(sector_code)
        sector_s    = sector_data.get(target_sym, pd.Series(dtype=float))
        rel_perf    = 0.0
        if len(sector_s) >= 21 and len(nifty) >= 21:
            sec_ret  = float(sector_s.iloc[-1] / sector_s.iloc[-21] - 1) * 100
            nif_ret  = float(nifty.iloc[-1] / nifty.iloc[-21] - 1) * 100
            rel_perf = round(sec_ret - nif_ret, 2)

        # ── Risk-on / Risk-off regime ─────────────────────────────────────────
        cyclical_avg  = np.mean([sector_moms.get(s, 0) for s in CYCLICAL_SECTORS
                                  if s in sector_moms])
        defensive_avg = np.mean([sector_moms.get(s, 0) for s in DEFENSIVE_SECTORS
                                  if s in sector_moms])
        if cyclical_avg > defensive_avg + 2:
            regime_type = "RISK_ON"
        elif defensive_avg > cyclical_avg + 2:
            regime_type = "RISK_OFF"
        else:
            regime_type = "NEUTRAL"

        # ── Top / bottom sectors ──────────────────────────────────────────────
        sorted_sectors = sorted(sector_moms.items(), key=lambda x: x[1], reverse=True)
        top3    = [SECTOR_NAMES.get(sc, sc) for sc, _ in sorted_sectors[:3]]
        bottom3 = [SECTOR_NAMES.get(sc, sc) for sc, _ in sorted_sectors[-3:]]

        # ── Signal ────────────────────────────────────────────────────────────
        if target_rank >= 0.75:
            signal = "BUY"
            score  = 0.65 + (target_rank - 0.75) * 0.8
            detail = (f"{SECTOR_NAMES.get(sector_code,'?')} sector top quartile "
                      f"({target_mom:+.1f}% 20d, rank {target_rank:.0%})")
        elif target_rank <= 0.25:
            signal = "SELL"
            score  = 0.35 - (0.25 - target_rank) * 0.8
            detail = (f"{SECTOR_NAMES.get(sector_code,'?')} sector bottom quartile "
                      f"({target_mom:+.1f}% 20d, rank {target_rank:.0%})")
        else:
            signal = "NEUTRAL"
            score  = 0.5
            detail = (f"{SECTOR_NAMES.get(sector_code,'?')} sector mid-range "
                      f"({target_mom:+.1f}% 20d)")

        return {
            "engine":           ENGINE_NAME,
            "signal":           signal,
            "score":            round(float(np.clip(score, 0.05, 0.95)), 3),
            "sector_rank":      target_rank,
            "sector_momentum":  round(target_mom, 2),
            "sector_rel_perf":  rel_perf,
            "regime_type":      regime_type,
            "top_sectors":      top3,
            "bottom_sectors":   bottom3,
            "detail":           detail,
        }

    except Exception as exc:
        return _neutral(sector_code, f"Error: {exc}")


def _neutral(sector_code, reason):
    return {
        "engine":          ENGINE_NAME,
        "signal":          "NEUTRAL",
        "score":           0.5,
        "sector_rank":     0.5,
        "sector_momentum": 0.0,
        "sector_rel_perf": 0.0,
        "regime_type":     "NEUTRAL",
        "top_sectors":     [],
        "bottom_sectors":  [],
        "detail":          reason,
    }
