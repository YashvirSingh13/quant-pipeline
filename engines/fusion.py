"""
engines/fusion.py — Signal Fusion Layer.

Combines signals from all parallel engines into one final verdict.

Engine weights (must sum to 1.0):
  xgboost          0.35  — most sophisticated, trained on 10Y of data
  multi_timeframe  0.25  — strong trend confirmation across timeframes
  sentiment        0.18  — leads price, catches news-driven moves
  mean_reversion   0.14  — catches different market regime (range-bound)
  fundamental_rank 0.08  — slow-moving directional quality bias

Volatility regime acts as a confidence MULTIPLIER (not a vote):
  Pulls all signals toward NEUTRAL when India VIX is high.

Final thresholds:
  > 0.63 → BUY
  < 0.37 → SELL
  else   → NEUTRAL

Agreement level: what % of engines voted the same direction.
"""

ENGINE_NAME = "fusion"

# Voting weights per engine
WEIGHTS = {
    "xgboost":          0.35,
    "multi_timeframe":  0.25,
    "sentiment":        0.18,
    "mean_reversion":   0.14,
    "fundamental_rank": 0.08,
}

BUY_THRESH  = 0.63
SELL_THRESH = 0.37


def _to_score(signal: str, engine_score: float) -> float:
    """Convert signal + engine score to a 0–1 fusion score."""
    if signal == "BUY":
        return engine_score                    # already 0.5–1.0 for BUY
    elif signal == "SELL":
        return 1.0 - engine_score             # already 0–0.5 for SELL
    else:
        return 0.50                            # NEUTRAL = no opinion


def fuse(engine_results: list, vix_multiplier: float = 1.0) -> dict:
    """
    Args:
        engine_results  : list of result dicts from each engine
                          must include 'engine', 'signal', 'score'
        vix_multiplier  : from volatility_regime engine (default 1.0)

    Returns dict:
        signal, confidence (0–100), agreement (0–100),
        votes {BUY, SELL, NEUTRAL}, engine_breakdown,
        raw_score, buy_threshold, sell_threshold
    """
    weighted_sum  = 0.0
    total_weight  = 0.0
    votes         = {"BUY": 0, "SELL": 0, "NEUTRAL": 0}
    breakdown     = []

    for result in engine_results:
        name   = result.get("engine", "unknown")
        signal = result.get("signal", "NEUTRAL")
        score  = float(result.get("score", 0.5))
        weight = WEIGHTS.get(name, 0.05)

        fusion_score  = _to_score(signal, score)
        weighted_sum += fusion_score * weight
        total_weight += weight
        votes[signal] = votes.get(signal, 0) + 1

        breakdown.append({
            "engine":       name,
            "signal":       signal,
            "score":        round(score, 3),
            "fusion_score": round(fusion_score, 3),
            "weight":       weight,
            "detail":       result.get("detail", ""),
        })

    if total_weight == 0:
        return _neutral_result(votes, breakdown)

    # Raw weighted average
    raw = weighted_sum / total_weight

    # Apply VIX confidence multiplier
    # Pulls score toward 0.5 (NEUTRAL) when volatility is high
    multiplier = max(0.20, min(1.15, vix_multiplier))
    adjusted   = 0.5 + (raw - 0.5) * multiplier

    # Final signal
    if adjusted > BUY_THRESH:
        final_signal = "BUY"
    elif adjusted < SELL_THRESH:
        final_signal = "SELL"
    else:
        final_signal = "NEUTRAL"

    # Agreement: % of engines that voted for the winning direction
    total_engines  = len(engine_results)
    winning_votes  = votes.get(final_signal, 0)
    agreement      = round(winning_votes / total_engines * 100, 1) if total_engines > 0 else 0

    # Confidence: how far from the neutral 0.5 midpoint, scaled to 0–100
    confidence = round(abs(adjusted - 0.5) * 200, 1)

    return {
        "signal":          final_signal,
        "confidence":      confidence,
        "agreement":       agreement,
        "agreement_label": f"{winning_votes}/{total_engines} engines",
        "votes":           votes,
        "raw_score":       round(raw, 4),
        "adjusted_score":  round(adjusted, 4),
        "vix_multiplier":  round(multiplier, 2),
        "engine_breakdown": breakdown,
        "buy_threshold":   BUY_THRESH,
        "sell_threshold":  SELL_THRESH,
    }


def _neutral_result(votes, breakdown):
    return {
        "signal": "NEUTRAL", "confidence": 0, "agreement": 0,
        "agreement_label": "0/0", "votes": votes, "raw_score": 0.5,
        "adjusted_score": 0.5, "vix_multiplier": 1.0,
        "engine_breakdown": breakdown,
        "buy_threshold": BUY_THRESH, "sell_threshold": SELL_THRESH,
    }
