"""
predict.py — CLI entry-point for single-row inference.

Usage (all features positional, in order):
    python ml/predict.py <RSI> <MA50> <MA200> <MA_Cross> <Volatility> \
                         <MACD> <MACD_Signal> <MACD_Hist> <BB_Width>  \
                         <Volume_Log> <Ticker>

Feature order must match the FEATURES list in train.py.
"""

import os
import sys
import json
import joblib

# ── Paths (work regardless of cwd) ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "model.pkl")
META_PATH  = os.path.join(ROOT_DIR, "model_metadata.json")

FEATURES = [
    "RSI", "MA50", "MA200", "MA_Cross", "Volatility",
    "MACD", "MACD_Signal", "MACD_Hist", "BB_Width",
    "Volume_Log", "Ticker"
]

# ── Helpers ─────────────────────────────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"model.pkl not found at {MODEL_PATH}. "
            "Run  python ml/train.py  first."
        )
    return joblib.load(MODEL_PATH)

def load_metadata() -> dict:
    if not os.path.exists(META_PATH):
        return {"threshold": 0.6, "features": FEATURES}
    with open(META_PATH) as f:
        return json.load(f)

def predict(feature_values: list, threshold: float | None = None) -> dict:
    """
    Run inference on a pre-ordered feature list.

    Returns a dict with signal, probability, threshold_used, and metadata.
    """
    model    = load_model()
    meta     = load_metadata()
    threshold = threshold if threshold is not None else meta.get("threshold", 0.6)

    expected_feats = meta.get("features", FEATURES)
    if len(feature_values) != len(expected_feats):
        raise ValueError(
            f"Expected {len(expected_feats)} features "
            f"({', '.join(expected_feats)}), got {len(feature_values)}."
        )

    prob   = float(model.predict_proba([feature_values])[0][1])
    signal = "BUY" if prob > threshold else "SELL"

    return {
        "signal":        signal,
        "probability":   round(prob * 100, 2),
        "threshold_used": threshold,
        "trained_at":    meta.get("trained_at", "unknown"),
        "cv_accuracy":   meta.get("cv_accuracy_mean"),
    }

# ── CLI ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        print(f"\nFeatures ({len(FEATURES)}):")
        for i, f in enumerate(FEATURES, 1):
            print(f"  {i:>2}. {f}")
        sys.exit(0)

    # Parse values
    try:
        vals = list(map(float, args))
    except ValueError as exc:
        print(f"❌  Input error: all values must be numeric. ({exc})")
        sys.exit(1)

    # Run
    try:
        result = predict(vals)
    except (FileNotFoundError, ValueError) as exc:
        print(f"❌  {exc}")
        sys.exit(1)

    icon = "🟢" if result["signal"] == "BUY" else "🔴"
    print(
        f"{icon}  {result['signal']}  |  "
        f"Prob: {result['probability']}%  |  "
        f"Threshold: {result['threshold_used']}  |  "
        f"CV Acc: {result['cv_accuracy']:.2%}  |  "
        f"Trained: {result['trained_at']}"
    )
