"""
engines/hmm_regime.py — HMM Regime Detection with Continuous Strength Vector.

Instead of just "TRENDING / SIDEWAYS / VOLATILE" labels, now outputs:
  regime_vector = {
    "trend_strength":          0.72,  # 0-1
    "volatility_level":        0.45,  # 0-1
    "mean_reversion_pull":     0.30,  # 0-1 (how stretched from mean)
    "momentum_consistency":    0.65,  # 0-1 (directional persistence)
  }

The fusion layer uses these continuous values for proportional weight
adjustment instead of discrete mode-switching.

Falls back to rule-based detection if hmmlearn unavailable.
"""

import numpy as np
import pandas as pd
import time

ENGINE_NAME = "hmm_regime"

_cache: dict = {}
CACHE_TTL = 6 * 3600

BASE_REGIME_WEIGHTS = {
    "TRENDING": {
        "xgboost": 1.10, "multi_timeframe": 1.30, "mean_reversion": 0.50,
        "sentiment": 1.00, "fundamental_rank": 0.90, "sector_corr": 1.10,
        "leader_lagger": 1.20, "confidence_mult": 1.05,
    },
    "SIDEWAYS": {
        "xgboost": 0.90, "multi_timeframe": 0.65, "mean_reversion": 1.60,
        "sentiment": 1.10, "fundamental_rank": 1.10, "sector_corr": 0.90,
        "leader_lagger": 0.80, "confidence_mult": 0.95,
    },
    "VOLATILE": {
        "xgboost": 0.85, "multi_timeframe": 0.70, "mean_reversion": 0.75,
        "sentiment": 0.80, "fundamental_rank": 1.00, "sector_corr": 0.70,
        "leader_lagger": 0.60, "confidence_mult": 0.60,
    },
    "UNKNOWN": {
        "xgboost": 1.0, "multi_timeframe": 1.0, "mean_reversion": 1.0,
        "sentiment": 1.0, "fundamental_rank": 1.0, "sector_corr": 1.0,
        "leader_lagger": 1.0, "confidence_mult": 1.0,
    },
}


def _compute_regime_vector(returns: pd.Series, close: pd.Series,
                            high: pd.Series, low_: pd.Series) -> dict:
    """
    Compute continuous regime strength metrics (all normalised 0-1).
    """
    if len(returns) < 30:
        return {"trend_strength": 0.5, "volatility_level": 0.5,
                "mean_reversion_pull": 0.5, "momentum_consistency": 0.5}

    # ── Trend strength (ADX-based, normalised) ────────────────────────────────
    tr       = pd.concat([high - low_,
                           (high - close.shift()).abs(),
                           (low_ - close.shift()).abs()], axis=1).max(axis=1)
    atr_14   = tr.rolling(14).mean()
    dm_plus  = (high.diff()).where(high.diff() > 0, 0).rolling(14).mean()
    dm_minus = (-low_.diff()).where(low_.diff() < 0, 0).rolling(14).mean()
    dip      = (dm_plus  / (atr_14 + 1e-9)).fillna(0)
    dim      = (dm_minus / (atr_14 + 1e-9)).fillna(0)
    raw_adx  = (abs(dip - dim) / (dip + dim + 1e-9)).rolling(14).mean().fillna(0)
    trend_strength = float(raw_adx.iloc[-1].clip(0, 1))

    # ── Volatility level (normalised vs 1-year history) ───────────────────────
    vol_20   = returns.rolling(20).std()
    vol_252  = returns.rolling(252).std()
    if float(vol_252.iloc[-1]) > 0:
        vol_ratio = float(vol_20.iloc[-1] / vol_252.iloc[-1])
    else:
        vol_ratio = 1.0
    volatility_level = float(min(1.0, vol_ratio / 2))   # normalise: ratio of 2 = 1.0

    # ── Mean reversion pull (how stretched price is from 20-day mean) ─────────
    ma20  = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    z_score = float(abs((close.iloc[-1] - ma20.iloc[-1]) / (std20.iloc[-1] + 1e-9)))
    mean_reversion_pull = float(min(1.0, z_score / 3))  # z=3 → fully stretched

    # ── Momentum consistency (% of last 10 days in same direction) ────────────
    recent_rets  = returns.iloc[-10:]
    pos_days     = (recent_rets > 0).sum()
    momentum_consistency = float(max(pos_days, 10 - pos_days) / 10)

    return {
        "trend_strength":       round(trend_strength,       3),
        "volatility_level":     round(volatility_level,     3),
        "mean_reversion_pull":  round(mean_reversion_pull,  3),
        "momentum_consistency": round(momentum_consistency,  3),
    }


def _continuous_weight_modifiers(rv: dict) -> dict:
    """
    Convert continuous regime vector to continuous weight modifiers.
    Uses proportional blending between TRENDING and SIDEWAYS based on
    trend_strength and mean_reversion_pull.
    """
    ts  = rv["trend_strength"]
    vl  = rv["volatility_level"]
    mrp = rv["mean_reversion_pull"]
    mc  = rv["momentum_consistency"]

    # Blend between TRENDING (ts=1) and SIDEWAYS (ts=0) weights
    engines = ["xgboost", "multi_timeframe", "mean_reversion",
               "sentiment", "fundamental_rank", "sector_corr", "leader_lagger"]
    mods = {}
    for eng in engines:
        w_trend   = BASE_REGIME_WEIGHTS["TRENDING"].get(eng, 1.0)
        w_sideways= BASE_REGIME_WEIGHTS["SIDEWAYS"].get(eng, 1.0)
        w_volatile= BASE_REGIME_WEIGHTS["VOLATILE"].get(eng, 1.0)

        # Blend trending/sideways by trend_strength
        base = ts * w_trend + (1 - ts) * w_sideways
        # Blend in volatile component by volatility_level
        blended = (1 - vl) * base + vl * w_volatile
        mods[eng] = round(float(blended), 3)

    # Confidence multiplier: reduced by volatility, boosted by consistency
    conf_trend = BASE_REGIME_WEIGHTS["TRENDING"]["confidence_mult"]
    conf_side  = BASE_REGIME_WEIGHTS["SIDEWAYS"]["confidence_mult"]
    conf_vol   = BASE_REGIME_WEIGHTS["VOLATILE"]["confidence_mult"]
    conf_base  = ts * conf_trend + (1 - ts) * conf_side
    conf_final = (1 - vl) * conf_base + vl * conf_vol
    # Boost slightly for high momentum consistency
    conf_final *= (0.9 + 0.2 * mc)
    mods["confidence_mult"] = round(float(min(1.20, max(0.25, conf_final))), 3)

    return mods


def _classify_regime(rv: dict) -> str:
    """Map continuous regime vector to a discrete label for display."""
    ts  = rv["trend_strength"]
    vl  = rv["volatility_level"]
    if vl > 0.65:
        return "VOLATILE"
    if ts > 0.55:
        return "TRENDING"
    return "SIDEWAYS"


def _rule_based(returns, close, high, low_):
    rv     = _compute_regime_vector(returns, close, high, low_)
    regime = _classify_regime(rv)
    return regime, rv


def _fit_hmm(observations: np.ndarray, returns, close, high, low_):
    from hmmlearn.hmm import GaussianHMM
    model = GaussianHMM(n_components=3, covariance_type="full",
                        n_iter=100, random_state=42)
    model.fit(observations)
    states     = model.predict(observations)
    last_state = int(states[-1])

    means    = model.means_[:, 0]
    variances= model.covars_[:, 0, 0]

    vol_state    = int(np.argmax(variances))
    trend_states = [i for i in range(3) if i != vol_state]
    trend_state  = int(trend_states[np.argmax([abs(means[i]) for i in trend_states])])
    side_state   = int([i for i in range(3) if i not in [vol_state, trend_state]][0])

    state_map = {vol_state: "VOLATILE", trend_state: "TRENDING", side_state: "SIDEWAYS"}
    regime    = state_map.get(last_state, "UNKNOWN")
    rv        = _compute_regime_vector(returns, close, high, low_)
    return regime, rv


def run(df: pd.DataFrame, ticker: str = "") -> dict:
    cache_key = f"hmm_{ticker}"
    now = time.time()
    if cache_key in _cache and (now - _cache[cache_key]["ts"]) < CACHE_TTL:
        return _cache[cache_key]["data"]

    try:
        close = df["Close"].squeeze().dropna()
        high  = df["High"].squeeze().reindex(close.index)
        low_  = df["Low"].squeeze().reindex(close.index)
        vol   = df["Volume"].squeeze().dropna()

        if len(close) < 60:
            raise ValueError("Insufficient data")

        returns  = close.pct_change().dropna()
        vol_20   = returns.rolling(20).std().dropna()
        vol_chg  = vol.pct_change().rolling(5).mean().reindex(returns.index).fillna(0)
        tr       = pd.concat([high - low_,
                               (high - close.shift()).abs(),
                               (low_ - close.shift()).abs()], axis=1).max(axis=1)
        dm_plus  = (high.diff()).where(high.diff() > 0, 0).rolling(14).mean()
        dm_minus = (-low_.diff()).where(low_.diff() < 0, 0).rolling(14).mean()
        atr14    = tr.rolling(14).mean()
        dip      = (dm_plus / (atr14 + 1e-9)).fillna(0)
        dim      = (dm_minus / (atr14 + 1e-9)).fillna(0)
        adx_proxy= (abs(dip - dim) / (dip + dim + 1e-9)).rolling(14).mean().fillna(0)

        common_idx = (returns.index
                      .intersection(vol_20.index)
                      .intersection(vol_chg.index)
                      .intersection(adx_proxy.index))
        obs = np.column_stack([
            returns.reindex(common_idx).values,
            vol_20.reindex(common_idx).values,
            vol_chg.reindex(common_idx).fillna(0).values,
            adx_proxy.reindex(common_idx).values,
        ])
        obs = obs[~np.isnan(obs).any(axis=1)]

        try:
            regime, rv = _fit_hmm(obs, returns, close, high, low_)
        except Exception:
            regime, rv = _rule_based(returns, close, high, low_)

        mods = _continuous_weight_modifiers(rv)

        regime_signals = {
            "TRENDING":  ("BUY",     0.65),
            "SIDEWAYS":  ("NEUTRAL", 0.50),
            "VOLATILE":  ("NEUTRAL", 0.45),
        }
        sig, score = regime_signals.get(regime, ("NEUTRAL", 0.50))

        # Consistency: how many of last 5 obs classify to same regime
        consistency = 0.6
        try:
            from hmmlearn.hmm import GaussianHMM
            m2 = GaussianHMM(n_components=3, covariance_type="full",
                             n_iter=50, random_state=42)
            m2.fit(obs)
            recent = m2.predict(obs[-5:])
            consistency = sum(1 for s in recent if s == recent[-1]) / 5
        except Exception:
            pass

        regime_descriptions = {
            "TRENDING": f"Trending regime (strength {rv['trend_strength']:.2f}) — momentum strategies favoured",
            "SIDEWAYS": f"Sideways regime (MR pull {rv['mean_reversion_pull']:.2f}) — mean reversion favoured",
            "VOLATILE": f"Volatile regime (vol level {rv['volatility_level']:.2f}) — reduce size, widen stops",
        }

        result = {
            "engine":                  ENGINE_NAME,
            "regime":                  regime,
            "signal":                  sig,
            "score":                   round(score, 3),
            "consistency":             round(consistency, 2),
            "regime_vector":           rv,
            "confidence_mult":         mods["confidence_mult"],
            "engine_weight_modifiers": {k: v for k, v in mods.items()
                                        if k != "confidence_mult"},
            "detail":                  regime_descriptions.get(regime, "Unknown regime"),
            "obs_count":               len(obs),
        }

        _cache[cache_key] = {"ts": now, "data": result}
        return result

    except Exception as exc:
        result = {
            "engine":  ENGINE_NAME, "regime": "UNKNOWN",
            "signal":  "NEUTRAL",  "score":   0.5,
            "consistency": 0,
            "regime_vector": {"trend_strength": 0.5, "volatility_level": 0.5,
                              "mean_reversion_pull": 0.5, "momentum_consistency": 0.5},
            "confidence_mult": 1.0,
            "engine_weight_modifiers": {e: 1.0 for e in
                ["xgboost","multi_timeframe","mean_reversion",
                 "sentiment","fundamental_rank","sector_corr","leader_lagger"]},
            "detail": f"HMM error: {exc}",
        }
        _cache[cache_key] = {"ts": now, "data": result}
        return result
