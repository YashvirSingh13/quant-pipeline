"""
engines/mean_reversion.py — Mean Reversion Engine.

Logic: Stocks that deviate significantly from their statistical mean
tend to revert. Uses Z-score + RSI + Bollinger Band position.

Works best in: sideways/range-bound markets, overbought/oversold stocks.
Complements XGBoost which is trend-following.

Returns score 0.0–1.0 where 1.0 = strong BUY, 0.0 = strong SELL.
"""

import numpy as np
import pandas as pd

ENGINE_NAME = "mean_reversion"


def _rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - (100 / (1 + g / l))


def run(ticker: str, df: pd.DataFrame) -> dict:
    """
    Args:
        ticker : e.g. "WIPRO.NS"
        df     : pre-fetched OHLCV DataFrame (at least 60 rows)

    Returns dict:
        engine, signal, score, zscore_20, zscore_50, rsi,
        bb_position, detail
    """
    try:
        close = df["Close"].squeeze()
        if len(close) < 30:
            raise ValueError("Insufficient data")

        # ── Z-scores ────────────────────────────────────────────────────────────
        ma20  = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        z20   = float(((close - ma20) / std20).iloc[-1])

        ma50  = close.rolling(50).mean()
        std50 = close.rolling(50).std()
        z50   = float(((close - ma50) / std50).iloc[-1])

        # ── RSI ──────────────────────────────────────────────────────────────────
        rsi = float(_rsi(close).iloc[-1])

        # ── Bollinger Band position (0 = lower band, 1 = upper band) ─────────────
        bb_upper = ma20 + 2 * std20
        bb_lower = ma20 - 2 * std20
        bb_range = (bb_upper - bb_lower).iloc[-1]
        bb_pos   = float((close.iloc[-1] - bb_lower.iloc[-1]) / bb_range) if bb_range > 0 else 0.5

        # ── Momentum: 5-day rate of change ────────────────────────────────────────
        roc5 = float(close.pct_change(5).iloc[-1] * 100)

        # ── Signal logic ─────────────────────────────────────────────────────────
        # Strong BUY: very oversold on multiple measures
        if z20 < -2.0 and rsi < 33 and bb_pos < 0.1:
            signal, score = "BUY",  min(0.95, 0.75 + abs(z20) * 0.05)
            detail = f"Deeply oversold: Z={z20:.2f}, RSI={rsi:.1f}"

        elif z20 < -1.5 and rsi < 42:
            signal, score = "BUY",  0.68
            detail = f"Oversold: Z={z20:.2f}, RSI={rsi:.1f}"

        elif z20 < -1.0 and rsi < 50:
            signal, score = "BUY",  0.60
            detail = f"Mildly oversold: Z={z20:.2f}"

        # Strong SELL: very overbought
        elif z20 > 2.0 and rsi > 67 and bb_pos > 0.9:
            signal, score = "SELL", max(0.05, 0.25 - abs(z20) * 0.05)
            detail = f"Deeply overbought: Z={z20:.2f}, RSI={rsi:.1f}"

        elif z20 > 1.5 and rsi > 58:
            signal, score = "SELL", 0.32
            detail = f"Overbought: Z={z20:.2f}, RSI={rsi:.1f}"

        elif z20 > 1.0 and rsi > 55:
            signal, score = "SELL", 0.40
            detail = f"Mildly overbought: Z={z20:.2f}"

        else:
            signal, score = "NEUTRAL", 0.50
            detail = f"Within normal range: Z={z20:.2f}, RSI={rsi:.1f}"

        return {
            "engine":      ENGINE_NAME,
            "signal":      signal,
            "score":       round(score, 3),
            "zscore_20":   round(z20,   3),
            "zscore_50":   round(z50,   3),
            "rsi":         round(rsi,   1),
            "bb_position": round(bb_pos, 3),
            "roc_5d":      round(roc5,  2),
            "detail":      detail,
        }

    except Exception as exc:
        return {
            "engine": ENGINE_NAME, "signal": "NEUTRAL", "score": 0.5,
            "detail": f"Error: {exc}",
        }
