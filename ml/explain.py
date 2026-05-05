"""
ml/explain.py — SHAP feature importance for model decisions.

Uses TreeExplainer (fast for XGBoost — typically <200ms).
Returns the top features ranked by absolute SHAP value,
along with their direction (bullish = pushed toward BUY,
bearish = pushed toward SELL).
"""

from __future__ import annotations

# Human-readable names for the feature display
FEATURE_LABELS = {
    "RSI":             "RSI (14)",
    "MA50":            "MA 50-day",
    "MA200":           "MA 200-day",
    "MA_Cross":        "MA Cross (50-200)",
    "Volatility":      "Volatility (10d)",
    "MACD":            "MACD Line",
    "MACD_Signal":     "MACD Signal",
    "MACD_Hist":       "MACD Histogram",
    "BB_Width":        "Bollinger Width",
    "Volume_Log":      "Volume (log)",
    "Volume_Spike":    "Volume Spike",
    "ATR":             "ATR (14d)",
    "High52W_Pct":     "52W High Position",
    "Low52W_Pct":      "52W Low Position",
    "Market_Return":   "Nifty Daily Return",
    "Market_Regime":   "Market Regime",
    "Earnings_Season": "Earnings Season",
    "Ticker":          "Stock Identity",
    "Sector":          "Sector",
}

def explain_prediction(feats_dict: dict, model, feature_list: list,
                       top_n: int = 10) -> list[dict]:
    """
    Compute SHAP values for a single live prediction.

    Args:
        feats_dict   : feature dict from _live_features (same keys as feature_list)
        model        : trained XGBoost model
        feature_list : ordered list of feature names the model expects
        top_n        : how many top features to return

    Returns:
        List of dicts sorted by |shap_value| descending:
        [{"feature": "RSI", "label": "RSI (14)", "shap_value": 0.142,
          "feature_value": 54.3, "direction": "bullish"}, ...]
    """
    try:
        import shap
        import pandas as pd

        row = pd.DataFrame([[feats_dict[f] for f in feature_list]], columns=feature_list)
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(row)

        results = []
        for feat, sv, fv in zip(feature_list, shap_vals[0], row.values[0]):
            results.append({
                "feature":       feat,
                "label":         FEATURE_LABELS.get(feat, feat),
                "shap_value":    round(float(sv), 4),
                "feature_value": round(float(fv), 4),
                "direction":     "bullish" if sv > 0 else "bearish",
            })

        results.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return results[:top_n]

    except ImportError:
        print("⚠  shap not installed — pip install shap")
        return []
    except Exception as exc:
        print(f"⚠  SHAP error: {exc}")
        return []
