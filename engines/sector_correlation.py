"""
engines/sector_correlation.py — Sector Lead-Lag Correlation Engine.

Detects whether sector peers are moving ahead of the target stock,
giving an early directional signal.

Logic:
  1. Download recent 60-day returns for sector peers
  2. Compute rolling correlation between peer returns and target's
     NEXT-day return (lead-lag with lag=1)
  3. If strong peers are rising and historically lead this stock → BUY signal
  4. If strong peers are falling → SELL signal

Also computes:
  - Sector momentum: is the overall sector trending up/down?
  - Relative strength: is target outperforming its sector peers?
  - Peer divergence: is the target diverging from its sector?
"""

import numpy as np
import pandas as pd
import yfinance as yf
import time

ENGINE_NAME = "sector_correlation"

# Sector peer groups (compact list — faster download)
SECTOR_PEERS = {
    0: ["HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","AXISBANK.NS","KOTAKBANK.NS"],  # Banking
    1: ["BAJFINANCE.NS","BAJAJFINSV.NS","SBILIFE.NS","HDFCLIFE.NS"],           # Finance
    2: ["TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS"],               # IT
    3: ["TATAMOTORS.NS","MARUTI.NS","BAJAJ-AUTO.NS","M&M.NS"],                # Auto
    4: ["RELIANCE.NS","ONGC.NS","BPCL.NS","NTPC.NS","POWERGRID.NS"],         # Energy
    5: ["HINDUNILVR.NS","ITC.NS","BRITANNIA.NS","NESTLEIND.NS"],              # FMCG
    6: ["LT.NS","ADANIPORTS.NS","ULTRACEMCO.NS","GRASIM.NS"],                 # Infra
    7: ["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS"],                # Pharma
    8: ["TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS"],                          # Metals
    9: ["ASIANPAINT.NS","TITAN.NS","BHARTIARTL.NS"],                          # Other
}

# Cache peer data — refreshed every 4 hours
_cache: dict = {}
CACHE_TTL = 4 * 3600

BUY_THRESH  = 0.60
SELL_THRESH = 0.40


def _fetch_peer_returns(peers: list, period: str = "3mo") -> pd.DataFrame:
    """Download and return a DataFrame of daily returns for peers."""
    cache_key = "_".join(sorted(peers))
    now = time.time()
    if cache_key in _cache and (now - _cache[cache_key]["ts"]) < CACHE_TTL:
        return _cache[cache_key]["data"]

    try:
        raw = yf.download(peers, period=period, interval="1d",
                          auto_adjust=True, progress=False)
        if raw.empty:
            return pd.DataFrame()
        prices  = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        returns = prices.pct_change().dropna()
        _cache[cache_key] = {"ts": now, "data": returns}
        return returns
    except Exception:
        return pd.DataFrame()


def run(ticker: str, df: pd.DataFrame, sector_map: dict) -> dict:
    """
    Args:
        ticker    : e.g. "WIPRO.NS"
        df        : pre-fetched OHLCV DataFrame for target stock
        sector_map: {ticker: sector_code}

    Returns dict:
        engine, signal, score, sector_momentum, rel_strength,
        peer_divergence, lead_lag_score, detail
    """
    try:
        sector_code = sector_map.get(ticker, -1)
        peers_all   = SECTOR_PEERS.get(sector_code, [])
        # Exclude the target stock itself from peer list
        peers       = [p for p in peers_all if p != ticker]

        if not peers:
            return {
                "engine": ENGINE_NAME, "signal": "NEUTRAL", "score": 0.5,
                "detail": f"No sector peers defined for {ticker}",
            }

        # ── Get peer returns ─────────────────────────────────────────────────
        peer_rets = _fetch_peer_returns(peers)
        if peer_rets.empty or len(peer_rets) < 20:
            return {
                "engine": ENGINE_NAME, "signal": "NEUTRAL", "score": 0.5,
                "detail": "Could not fetch peer data",
            }

        target_close = df["Close"].squeeze().dropna()
        target_ret   = target_close.pct_change().dropna()

        # Align indices
        common = peer_rets.index.intersection(target_ret.index)
        if len(common) < 20:
            return {
                "engine": ENGINE_NAME, "signal": "NEUTRAL", "score": 0.5,
                "detail": "Insufficient aligned data",
            }

        peer_rets_aligned = peer_rets.reindex(common)
        target_aligned    = target_ret.reindex(common)

        # ── Sector momentum: equal-weighted average peer return (20d) ────────
        peer_20d_avg = peer_rets_aligned.iloc[-20:].mean().mean()
        sector_momentum = float(peer_20d_avg * 100)

        # ── Relative strength: target vs sector (20d) ────────────────────────
        target_20d  = float(target_aligned.iloc[-20:].sum() * 100)
        peer_20d    = float(peer_rets_aligned.iloc[-20:].mean(axis=1).sum() * 100)
        rel_strength = round(target_20d - peer_20d, 2)

        # ── Lead-lag: do peers predict target's next-day move? ───────────────
        # Compute average correlation between peer_t and target_t+1
        target_next = target_aligned.shift(-1).dropna()
        lead_scores = []
        for col in peer_rets_aligned.columns:
            peer_col = peer_rets_aligned[col].reindex(target_next.index).dropna()
            tgt_col  = target_next.reindex(peer_col.index)
            if len(peer_col) >= 20:
                corr = float(peer_col.corr(tgt_col))
                if not np.isnan(corr):
                    lead_scores.append(corr)

        avg_lead_corr = float(np.mean(lead_scores)) if lead_scores else 0.0

        # Recent peer direction (last 5 days)
        recent_peer_ret = float(peer_rets_aligned.iloc[-5:].mean().mean())

        # ── Peer divergence: is target moving differently from peers? ─────────
        target_5d = float(target_aligned.iloc[-5:].mean())
        peer_5d   = float(peer_rets_aligned.iloc[-5:].mean().mean())
        divergence = target_5d - peer_5d   # positive = outperforming

        # ── Signal logic ─────────────────────────────────────────────────────
        # Lead-lag weighted by sector momentum and recent peer direction
        lead_lag_score = avg_lead_corr * recent_peer_ret * 100

        score = 0.50   # neutral baseline

        if avg_lead_corr > 0.15 and recent_peer_ret > 0.003:
            # Peers are rising and historically predict target rising
            score += 0.12

        if avg_lead_corr < -0.15 and recent_peer_ret < -0.003:
            # Peers are falling and historically predict target falling
            score -= 0.12

        if sector_momentum > 1.5:
            score += 0.08   # sector tailwind
        elif sector_momentum < -1.5:
            score -= 0.08   # sector headwind

        if rel_strength > 2.0:
            score += 0.05   # outperforming sector → momentum
        elif rel_strength < -2.0:
            score -= 0.05   # underperforming → weakness

        score = round(max(0.05, min(0.95, score)), 3)

        if score > BUY_THRESH:
            signal = "BUY"
            detail = (f"Sector tailwind: peers +{sector_momentum:.1f}% (20d), "
                      f"target outperforming by {rel_strength:+.1f}%")
        elif score < SELL_THRESH:
            signal = "SELL"
            detail = (f"Sector headwind: peers {sector_momentum:.1f}% (20d), "
                      f"target {rel_strength:+.1f}% vs peers")
        else:
            signal = "NEUTRAL"
            detail = (f"Mixed sector signals: momentum {sector_momentum:.1f}%, "
                      f"rel-strength {rel_strength:+.1f}%")

        return {
            "engine":           ENGINE_NAME,
            "signal":           signal,
            "score":            score,
            "sector_momentum":  round(sector_momentum, 2),
            "rel_strength":     rel_strength,
            "peer_divergence":  round(divergence * 100, 2),
            "lead_lag_corr":    round(avg_lead_corr, 3),
            "lead_lag_score":   round(lead_lag_score, 3),
            "n_peers":          len(peers),
            "detail":           detail,
        }

    except Exception as exc:
        return {
            "engine":  ENGINE_NAME,
            "signal":  "NEUTRAL",
            "score":   0.5,
            "detail":  f"Error: {exc}",
        }
