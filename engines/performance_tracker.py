"""
engines/performance_tracker.py — Dynamic Engine Weight System.

Tracks each engine's predictions vs actual outcomes over rolling window.
Computes regime-conditional accuracy → feeds into fusion as dynamic weights.

Storage: DATA_DIR/engine_performance.json
Schema:
{
  "signals": [
    {
      "ts":        ISO timestamp,
      "ticker":    "WIPRO.NS",
      "engine":    "xgboost",
      "signal":    "BUY",
      "score":     0.72,
      "regime":    "TRENDING",
      "price":     489.5,
      "resolve_date": ISO date (5 days later),
      "outcome":   null | "CORRECT" | "WRONG" | "NEUTRAL_SKIP"
    }, ...
  ]
}

Weight computation:
  For each engine × regime combination:
    accuracy = correct / (correct + wrong) over last N_WINDOW resolved signals
    weight_modifier = 0.5 + accuracy  (range: 0.5–1.5)
    → clipped to [0.40, 1.60]

If fewer than MIN_SIGNALS resolved → use default weight modifier of 1.0.
"""

import os
import json
import math
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

DATA_DIR    = os.environ.get("DATA_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
PERF_FILE   = os.path.join(DATA_DIR, "engine_performance.json")

N_WINDOW    = 40    # rolling window of resolved signals per engine per regime
MIN_SIGNALS = 8     # minimum resolved signals before adjusting weights

ENGINES = [
    "xgboost", "multi_timeframe", "mean_reversion",
    "sentiment", "fundamental_rank", "sector_corr",
]
REGIMES = ["TRENDING", "SIDEWAYS", "VOLATILE", "UNKNOWN"]


# ── I/O ────────────────────────────────────────────────────────────────────────
def _load() -> dict:
    if os.path.exists(PERF_FILE):
        try:
            with open(PERF_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"signals": []}


def _save(data: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PERF_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── Record a signal ────────────────────────────────────────────────────────────
def record_signals(ticker: str, engine_results: list,
                   regime: str, price: float):
    """
    Call this after every live prediction to log each engine's signal.
    Outcome is filled in later by resolve_outcomes().

    engine_results: list of dicts from each engine with 'engine', 'signal', 'score'
    """
    data  = _load()
    now   = datetime.utcnow()
    resolve = (now + timedelta(days=6)).strftime("%Y-%m-%d")  # 5 trading days ~= 6 calendar

    new_entries = []
    for res in engine_results:
        eng_name = res.get("engine", "unknown")
        signal   = res.get("signal", "NEUTRAL")
        if signal == "NEUTRAL":
            continue   # don't track NEUTRAL signals — no outcome to measure

        new_entries.append({
            "ts":           now.isoformat(),
            "ticker":       ticker,
            "engine":       eng_name,
            "signal":       signal,
            "score":        round(float(res.get("score", 0.5)), 4),
            "regime":       regime or "UNKNOWN",
            "price":        round(float(price), 4),
            "resolve_date": resolve,
            "outcome":      None,
        })

    data["signals"].extend(new_entries)

    # Keep only last 2000 entries to prevent file bloat
    data["signals"] = data["signals"][-2000:]
    _save(data)


# ── Resolve outcomes ───────────────────────────────────────────────────────────
def resolve_outcomes(price_fetcher):
    """
    Check all pending signals whose resolve_date has passed.
    price_fetcher: callable(ticker) → current price (float) or None

    Marks outcome as CORRECT / WRONG based on:
      BUY signal  → correct if current_price > entry_price × 1.003 (0.3% threshold)
      SELL signal → correct if current_price < entry_price × 0.997
    """
    data    = _load()
    today   = datetime.utcnow().strftime("%Y-%m-%d")
    updated = 0

    for entry in data["signals"]:
        if entry.get("outcome") is not None:
            continue
        if entry.get("resolve_date", "9999") > today:
            continue

        try:
            current = price_fetcher(entry["ticker"])
            if current is None:
                continue

            entry_price = entry["price"]
            signal      = entry["signal"]

            if signal == "BUY":
                entry["outcome"] = "CORRECT" if current > entry_price * 1.003 else "WRONG"
            elif signal == "SELL":
                entry["outcome"] = "CORRECT" if current < entry_price * 0.997 else "WRONG"
            else:
                entry["outcome"] = "NEUTRAL_SKIP"

            updated += 1
        except Exception:
            continue

    if updated:
        _save(data)

    return updated


# ── Compute dynamic weights ────────────────────────────────────────────────────
def get_dynamic_weights(current_regime: str = "UNKNOWN") -> dict:
    """
    Compute weight modifiers for each engine based on their rolling accuracy
    in the current regime (and overall).

    Returns:
        {engine_name: weight_modifier}   e.g. {"xgboost": 1.15, "sentiment": 0.82, ...}
    """
    data    = _load()
    signals = data.get("signals", [])
    resolved = [s for s in signals if s.get("outcome") in ("CORRECT", "WRONG")]

    modifiers = {}

    for engine in ENGINES:
        # Filter: this engine, resolved, in current regime
        regime_sigs = [s for s in resolved
                       if s["engine"] == engine and s["regime"] == current_regime]
        overall_sigs = [s for s in resolved if s["engine"] == engine]

        # Use regime-specific if enough data, else fall back to overall
        target_sigs = regime_sigs[-N_WINDOW:] if len(regime_sigs) >= MIN_SIGNALS \
                      else overall_sigs[-N_WINDOW:]

        if len(target_sigs) < MIN_SIGNALS:
            modifiers[engine] = 1.0   # not enough data → neutral
            continue

        correct = sum(1 for s in target_sigs if s["outcome"] == "CORRECT")
        total   = len(target_sigs)
        accuracy = correct / total

        # Weight modifier: accuracy 50% → 1.0, 65% → 1.30, 40% → 0.80
        # Formula: 0.4 + 1.2 × accuracy  (range: 0.4–1.6)
        modifier = 0.4 + 1.2 * accuracy
        modifier = max(0.40, min(1.60, modifier))
        modifiers[engine] = round(modifier, 3)

    return modifiers


# ── Summary stats ──────────────────────────────────────────────────────────────
def get_performance_summary() -> dict:
    """
    Returns a summary dict for display in the UI.
    {engine: {accuracy, n_signals, n_resolved, regime_breakdown}}
    """
    data    = _load()
    signals = data.get("signals", [])
    summary = {}

    for engine in ENGINES:
        eng_sigs  = [s for s in signals if s["engine"] == engine]
        resolved  = [s for s in eng_sigs if s.get("outcome") in ("CORRECT","WRONG")]
        correct   = sum(1 for s in resolved if s["outcome"] == "CORRECT")
        accuracy  = correct / len(resolved) if resolved else None

        regime_stats = {}
        for regime in REGIMES:
            r_sigs    = [s for s in resolved if s["regime"] == regime]
            r_correct = sum(1 for s in r_sigs if s["outcome"] == "CORRECT")
            regime_stats[regime] = {
                "n":        len(r_sigs),
                "accuracy": round(r_correct / len(r_sigs), 3) if r_sigs else None,
            }

        summary[engine] = {
            "n_signals":  len(eng_sigs),
            "n_resolved": len(resolved),
            "accuracy":   round(accuracy, 3) if accuracy is not None else None,
            "regimes":    regime_stats,
        }

    return summary
