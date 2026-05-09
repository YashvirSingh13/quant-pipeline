# QP-a11d4d92-825 2026-05-09 07:07:11
"""
ml/calibration.py — Shared calibration class.
Importable by both train.py and server/app.py so joblib.load works.

IMPORTANT: No __getattr__ — it causes infinite recursion during unpickling
because Python calls __getattr__ to find 'base_model' before it's set.
"""
import numpy as np


class CalibratedModel:
    """
    Manual isotonic calibration wrapper around XGBoost.
    Maps raw probabilities → true calibrated probabilities.
    Safe for joblib.dump / joblib.load.
    """

    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    # ── Explicit pickle support (avoids __getattr__ recursion) ───────────────
    def __getstate__(self):
        return {"base_model": self.base_model, "calibrator": self.calibrator}

    def __setstate__(self, state):
        self.base_model = state["base_model"]
        self.calibrator = state["calibrator"]

    # ── Core interface ────────────────────────────────────────────────────────
    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = np.clip(self.calibrator.predict(np.clip(raw, 0, 1)), 0, 1)
        return np.column_stack([1 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_booster(self):
        """Expose XGBoost booster for feature name extraction in backtest."""
        return self.base_model.get_booster()

    def get_params(self, deep=True):
        return self.base_model.get_params(deep=deep)

    # NO __getattr__ — it breaks pickle/unpickle
