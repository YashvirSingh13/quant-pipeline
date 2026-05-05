"""
engines/fundamental_rank.py — Fundamental Quality Engine.

Scores a stock on four fundamental pillars:
  1. Valuation  — PE vs sector average (cheaper = better score)
  2. Profitability — ROE + profit margin
  3. Growth     — revenue + earnings YoY growth
  4. Health     — debt/equity + current ratio

Composite score 0–1 → BUY / NEUTRAL / SELL signal.

Note: fundamentals change quarterly, not daily.
This engine adds slow-moving directional bias to the consensus.
"""

import yfinance as yf

ENGINE_NAME = "fundamental_rank"

BUY_THRESH  = 0.62
SELL_THRESH = 0.38


def run(ticker: str, sector_map: dict, sector_benchmarks: dict,
        prefetched_info: dict | None = None) -> dict:
    """
    Args:
        ticker            : e.g. "WIPRO.NS"
        sector_map        : {ticker: sector_code}
        sector_benchmarks : {sector_code: {pe, pb, div_yield, name}}
        prefetched_info   : yf.Ticker.info dict if already fetched (avoids duplicate call)
    """
    try:
        info = prefetched_info or yf.Ticker(ticker).info or {}

        sector_code = sector_map.get(ticker, 7)
        bench       = sector_benchmarks[sector_code]

        scores = {}
        details = {}

        # ── Valuation ────────────────────────────────────────────────────────────
        pe = info.get("trailingPE") or info.get("forwardPE")
        if pe and bench["pe"] and pe > 0:
            ratio  = pe / bench["pe"]
            # Discount curve: 0.5x sector PE → 0.90 score, 2x → 0.10 score
            val_sc = max(0.05, min(0.95, 1.05 - ratio * 0.45))
            scores["valuation"] = val_sc
            details["valuation"] = f"PE {pe:.1f} vs sector {bench['pe']} ({ratio:.2f}x)"
        else:
            scores["valuation"] = 0.50
            details["valuation"] = "PE unavailable"

        # ── Profitability ────────────────────────────────────────────────────────
        roe    = (info.get("returnOnEquity")  or 0) * 100
        margin = (info.get("profitMargins")   or 0) * 100
        roa    = (info.get("returnOnAssets")  or 0) * 100
        # ROE > 20% excellent, > 12% good, < 5% poor
        roe_sc  = max(0.0, min(1.0, roe / 25))
        marg_sc = max(0.0, min(1.0, margin / 20))
        prof_sc = (roe_sc * 0.6 + marg_sc * 0.4)
        scores["profitability"] = prof_sc
        details["profitability"] = f"ROE {roe:.1f}%, Margin {margin:.1f}%"

        # ── Growth ───────────────────────────────────────────────────────────────
        rev_g  = (info.get("revenueGrowth")  or 0) * 100
        earn_g = (info.get("earningsGrowth") or 0) * 100
        # 20% growth → score 1.0, 0% → 0.5, -20% → 0.0
        rev_sc  = max(0.0, min(1.0, (rev_g  + 20) / 40))
        earn_sc = max(0.0, min(1.0, (earn_g + 20) / 40))
        grow_sc = (rev_sc * 0.5 + earn_sc * 0.5)
        scores["growth"] = grow_sc
        details["growth"] = f"Rev {rev_g:+.1f}%, Earn {earn_g:+.1f}% YoY"

        # ── Financial Health ─────────────────────────────────────────────────────
        debt_eq  = info.get("debtToEquity") or 0
        curr_rat = info.get("currentRatio") or 1.5
        # Low debt + high current ratio = good
        debt_sc  = max(0.0, min(1.0, 1.0 - debt_eq / 300))
        curr_sc  = max(0.0, min(1.0, (curr_rat - 0.5) / 2.0))
        health_sc = (debt_sc * 0.5 + curr_sc * 0.5)
        scores["health"] = health_sc
        details["health"] = f"D/E {debt_eq:.0f}%, CR {curr_rat:.2f}"

        # ── Composite ────────────────────────────────────────────────────────────
        weights   = {"valuation": 0.30, "profitability": 0.30,
                     "growth": 0.25,    "health": 0.15}
        composite = sum(scores[k] * weights[k] for k in weights)

        if composite >= BUY_THRESH:
            signal = "BUY"
            detail = f"Strong fundamentals (score {composite:.2f})"
        elif composite <= SELL_THRESH:
            signal = "SELL"
            detail = f"Weak fundamentals (score {composite:.2f})"
        else:
            signal = "NEUTRAL"
            detail = f"Average fundamentals (score {composite:.2f})"

        return {
            "engine":    ENGINE_NAME,
            "signal":    signal,
            "score":     round(composite, 3),
            "pillar_scores": {k: round(v, 3) for k, v in scores.items()},
            "pillar_details": details,
            "sector":    bench["name"],
            "detail":    detail,
        }

    except Exception as exc:
        return {
            "engine":  ENGINE_NAME,
            "signal":  "NEUTRAL",
            "score":   0.50,
            "detail":  f"Error: {exc}",
        }
