"""
engines/fusion.py — Signal Fusion Layer v3.

Upgrades:
  1. Performance-based dynamic weighting
     → Each engine's weight adjusted by its rolling regime-conditional accuracy
     → Stored in engine_performance.json, updated after each resolved trade
  2. Continuous regime weighting (from HMM regime vector)
     → No discrete mode switches — proportional blending
  3. Leader-lagger included as a voting engine
  4. VIX multiplier on top

Final weight per engine:
  effective_weight = base_weight × perf_modifier × regime_modifier

Renormalised to sum=1 before fusion.
"""

ENGINE_NAME = "fusion"

# ── Base weights (starting point before adjustments) ─────────────────────────
BASE_WEIGHTS = {
    "xgboost":          0.25,
    "multi_timeframe":  0.17,
    "mean_reversion":   0.11,
    "sentiment":        0.12,
    "fundamental_rank": 0.06,
    "sector_corr":      0.09,
    "leader_lagger":    0.07,
    "sector_rotation":  0.08,   # Phase 9
    "eps_fundamental":  0.05,   # Phase 9
}

BUY_THRESH  = 0.62
SELL_THRESH = 0.38


def _to_score(signal: str, engine_score: float) -> float:
    if signal == "BUY":   return float(engine_score)
    if signal == "SELL":  return 1.0 - float(engine_score)
    return 0.50


def fuse(engine_results: list,
         vix_multiplier: float = 1.0,
         hmm_regime_result: dict | None = None,
         current_regime: str = "UNKNOWN") -> dict:
    """
    Args:
        engine_results    : list of engine dicts
        vix_multiplier    : from volatility_regime engine
        hmm_regime_result : from hmm_regime engine
        current_regime    : regime label for performance tracker lookup

    Returns comprehensive fusion dict.
    """
    # ── Step 1: Performance-based weight modifiers ────────────────────────────
    try:
        from engines.performance_tracker import get_dynamic_weights
        perf_mods = get_dynamic_weights(current_regime)
    except Exception:
        perf_mods = {e: 1.0 for e in BASE_WEIGHTS}

    # ── Step 2: Regime-based weight modifiers (continuous) ────────────────────
    regime_mods  = {}
    regime_conf  = 1.0
    regime_vector = {}
    regime        = "UNKNOWN"

    if hmm_regime_result:
        regime       = hmm_regime_result.get("regime", "UNKNOWN")
        regime_mods  = hmm_regime_result.get("engine_weight_modifiers", {})
        regime_conf  = hmm_regime_result.get("confidence_mult", 1.0)
        regime_vector= hmm_regime_result.get("regime_vector", {})

    # ── Step 3: Compute effective weights ─────────────────────────────────────
    effective_weights = {}
    for name, base_w in BASE_WEIGHTS.items():
        perf_mod   = perf_mods.get(name, 1.0)
        regime_mod = regime_mods.get(name, 1.0)
        effective_weights[name] = base_w * perf_mod * regime_mod

    # Renormalise to sum = 1
    total = sum(effective_weights.values())
    if total > 0:
        effective_weights = {k: v / total for k, v in effective_weights.items()}

    # ── Step 4: Weighted fusion ───────────────────────────────────────────────
    weighted_sum = 0.0
    total_weight = 0.0
    votes        = {"BUY": 0, "SELL": 0, "NEUTRAL": 0}
    breakdown    = []

    for result in engine_results:
        name   = result.get("engine", "unknown")
        signal = result.get("signal", "NEUTRAL")
        score  = float(result.get("score", 0.5))
        weight = effective_weights.get(name, 0.03)

        fusion_score  = _to_score(signal, score)
        weighted_sum += fusion_score * weight
        total_weight += weight
        votes[signal] = votes.get(signal, 0) + 1

        breakdown.append({
            "engine":        name,
            "signal":        signal,
            "score":         round(score, 3),
            "fusion_score":  round(fusion_score, 3),
            "weight":        round(weight, 4),
            "base_weight":   BASE_WEIGHTS.get(name, 0.03),
            "perf_mod":      round(perf_mods.get(name, 1.0), 3),
            "regime_mod":    round(regime_mods.get(name, 1.0), 3),
            "detail":        result.get("detail", ""),
        })

    if total_weight == 0:
        return _neutral_result(votes, breakdown, regime, regime_vector)

    raw = weighted_sum / total_weight

    # ── Step 5: Apply multipliers (regime × VIX) ──────────────────────────────
    combined_mult = min(1.20, max(0.20, regime_conf * vix_multiplier))
    adjusted      = 0.5 + (raw - 0.5) * combined_mult

    # ── Step 6: Final signal ──────────────────────────────────────────────────
    if   adjusted > BUY_THRESH:  final_signal = "BUY"
    elif adjusted < SELL_THRESH: final_signal = "SELL"
    else:                        final_signal = "NEUTRAL"

    # ── Step 7: Agreement + confidence ───────────────────────────────────────
    total_engines = len(engine_results)
    winning_votes = votes.get(final_signal, 0)
    agreement     = round(winning_votes / total_engines * 100, 1) if total_engines else 0
    confidence    = round(abs(adjusted - 0.5) * 200, 1)

    # ── Step 8: Tier ──────────────────────────────────────────────────────────
    buy_v  = votes.get("BUY",  0)
    sell_v = votes.get("SELL", 0)
    if final_signal != "NEUTRAL" and winning_votes >= 5 and confidence > 40:
        tier = "HIGH"
    elif final_signal != "NEUTRAL" and winning_votes >= 3:
        tier = "MEDIUM"
    elif buy_v > 0 and sell_v > 0 and abs(buy_v - sell_v) <= 1:
        tier = "WATCH"
    else:
        tier = "LOW"

    return {
        "signal":              final_signal,
        "confidence":          confidence,
        "tier":                tier,
        "agreement":           agreement,
        "agreement_label":     f"{winning_votes}/{total_engines} engines",
        "votes":               votes,
        "raw_score":           round(raw, 4),
        "adjusted_score":      round(adjusted, 4),
        "vix_multiplier":      round(vix_multiplier, 2),
        "regime":              regime,
        "regime_vector":       regime_vector,
        "regime_confidence":   round(regime_conf, 2),
        "combined_multiplier": round(combined_mult, 2),
        "perf_modifiers":      {k: round(v, 3) for k, v in perf_mods.items()},
        "engine_breakdown":    breakdown,
        "effective_weights":   {k: round(v, 4) for k, v in effective_weights.items()},
        "buy_threshold":       BUY_THRESH,
        "sell_threshold":      SELL_THRESH,
    }


def _neutral_result(votes, breakdown, regime, regime_vector):
    return {
        "signal": "NEUTRAL", "confidence": 0, "tier": "LOW",
        "agreement": 0, "agreement_label": "0/0", "votes": votes,
        "raw_score": 0.5, "adjusted_score": 0.5,
        "vix_multiplier": 1.0, "regime": regime,
        "regime_vector": regime_vector,
        "regime_confidence": 1.0, "combined_multiplier": 1.0,
        "perf_modifiers": {e: 1.0 for e in BASE_WEIGHTS},
        "engine_breakdown": breakdown,
        "effective_weights": BASE_WEIGHTS,
        "buy_threshold": BUY_THRESH, "sell_threshold": SELL_THRESH,
    }
