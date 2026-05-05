"""
engines/multi_timeframe.py — Multi-Timeframe Alignment Engine.

Logic: Only act when multiple timeframes agree on direction.
- Daily  (short-term momentum)
- Weekly (medium-term trend)
- Monthly (long-term direction)

Strong signal when all 3 align. Weak/NEUTRAL when mixed.
Uses pandas resample on the pre-fetched daily df — no extra downloads.
"""

import numpy as np
import pandas as pd

ENGINE_NAME = "multi_timeframe"


def _trend(series: pd.Series, ma_period: int) -> str:
    """Returns 'UP' or 'DOWN' based on price vs its moving average."""
    if len(series) < ma_period + 2:
        return "UNKNOWN"
    ma = series.rolling(ma_period).mean()
    # Also check slope: is MA itself rising?
    ma_slope = ma.iloc[-1] - ma.iloc[-5] if len(ma) >= 5 else 0
    above_ma  = series.iloc[-1] > ma.iloc[-1]
    rising_ma = ma_slope > 0
    if above_ma and rising_ma:   return "UP_STRONG"
    if above_ma and not rising_ma: return "UP_WEAK"
    if not above_ma and not rising_ma: return "DOWN_STRONG"
    return "DOWN_WEAK"


def _momentum(series: pd.Series, period: int) -> float:
    """Rate of change over period."""
    if len(series) < period + 1:
        return 0.0
    return float((series.iloc[-1] / series.iloc[-period] - 1) * 100)


def run(ticker: str, df: pd.DataFrame) -> dict:
    """
    Args:
        ticker : e.g. "WIPRO.NS"
        df     : pre-fetched daily OHLCV DataFrame (at least 90 rows)
    """
    try:
        close = df["Close"].squeeze().dropna()
        if len(close) < 50:
            raise ValueError("Insufficient data for multi-timeframe analysis")

        # ── Resample to weekly and monthly ──────────────────────────────────────
        close.index = pd.DatetimeIndex(close.index)
        weekly  = close.resample("W").last().dropna()
        monthly = close.resample("ME").last().dropna()

        # ── Trend per timeframe ─────────────────────────────────────────────────
        d_trend = _trend(close,   50)    # 50-day MA
        w_trend = _trend(weekly,  20)    # 20-week MA (~1 year)
        m_trend = _trend(monthly, 6)     # 6-month MA

        # ── Momentum ────────────────────────────────────────────────────────────
        d_mom = _momentum(close,   20)   # 1-month
        w_mom = _momentum(weekly,  12)   # 3-month
        m_mom = _momentum(monthly,  3)   # 3-month on monthly

        # ── Count bullish / bearish timeframes ──────────────────────────────────
        is_bullish = lambda t: t.startswith("UP")
        is_bearish = lambda t: t.startswith("DOWN")

        bulls  = sum(is_bullish(t) for t in [d_trend, w_trend, m_trend])
        bears  = sum(is_bearish(t) for t in [d_trend, w_trend, m_trend])
        strong_bulls = sum(t == "UP_STRONG"   for t in [d_trend, w_trend, m_trend])
        strong_bears = sum(t == "DOWN_STRONG" for t in [d_trend, w_trend, m_trend])

        alignment = f"{max(bulls, bears)}/3"

        # ── Signal ───────────────────────────────────────────────────────────────
        if bulls == 3 and strong_bulls >= 2:
            signal, score = "BUY",  0.90
            detail = "All 3 timeframes bullish — strong alignment"
        elif bulls == 3:
            signal, score = "BUY",  0.75
            detail = "All 3 timeframes bullish"
        elif bulls == 2 and strong_bulls >= 1:
            signal, score = "BUY",  0.63
            detail = f"2/3 timeframes bullish ({alignment})"
        elif bears == 3 and strong_bears >= 2:
            signal, score = "SELL", 0.10
            detail = "All 3 timeframes bearish — strong alignment"
        elif bears == 3:
            signal, score = "SELL", 0.25
            detail = "All 3 timeframes bearish"
        elif bears == 2 and strong_bears >= 1:
            signal, score = "SELL", 0.37
            detail = f"2/3 timeframes bearish ({alignment})"
        else:
            signal, score = "NEUTRAL", 0.50
            detail = f"Mixed timeframes: {alignment} agree"

        return {
            "engine":     ENGINE_NAME,
            "signal":     signal,
            "score":      round(score, 3),
            "daily":      d_trend,
            "weekly":     w_trend,
            "monthly":    m_trend,
            "alignment":  alignment,
            "mom_1m":     round(d_mom, 2),
            "mom_3m":     round(w_mom, 2),
            "detail":     detail,
        }

    except Exception as exc:
        return {
            "engine": ENGINE_NAME, "signal": "NEUTRAL", "score": 0.5,
            "detail": f"Error: {exc}",
        }
