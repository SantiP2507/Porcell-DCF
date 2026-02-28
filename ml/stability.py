"""
ml/stability.py — Valuation Stability Detection Model.

Detects whether a DCF valuation is reliable (stable) or highly sensitive
to assumption changes (unstable). Unstable valuations should be treated
with much more skepticism even if they show attractive upside.

Model: Gradient Boosted Trees (GradientBoostingClassifier)
  - Better than Random Forest at capturing threshold effects
    (e.g. "TV% above 75% is a cliff, not a slope")
  - Still interpretable via feature importances

Output:
  - is_stable: bool
  - confidence: 0.0–1.0 (how confident the model is)
  - reason: human-readable explanation of what's driving instability

Key stability signals the model learns:
  - Terminal value % (high TV% = model depends on unknowable future)
  - FCF historical consistency (volatile FCF = unreliable base)
  - Bear-bull spread (wide spread = assumptions matter a lot)
  - FCF growth rate (high assumed growth = more sensitive)
"""

import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict

from ml.bootstrap import generate_stability_data, FEATURE_NAMES
from config import BASE_DIR as ROOT_DIR

logger = logging.getLogger(__name__)

MODEL_PATH = ROOT_DIR / "ml" / "models" / "stability.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

MIN_REAL_SAMPLES_FOR_BLEND = 50
MIN_REAL_SAMPLES_ONLY = 500

# Features most predictive of instability (for rule-based fallback explanation)
INSTABILITY_FEATURES = {
    "terminal_value_pct": (0.75, "Terminal value exceeds 75% of EV"),
    "fcf_stability":      (0.40, "Historical FCF is inconsistent"),
    "bear_base_spread":   (0.80, "Wide gap between bear and bull scenarios"),
    "leverage_ratio":     (7.0,  "High leverage amplifies assumption sensitivity"),
}


def train(real_X=None, real_y=None) -> "StabilityModel":
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score

    syn_X, syn_y = generate_stability_data(n=2000)

    if real_X is not None and len(real_X) >= MIN_REAL_SAMPLES_FOR_BLEND:
        if len(real_X) >= MIN_REAL_SAMPLES_ONLY:
            X, y = real_X, real_y
        else:
            real_weight = 3
            X = np.vstack([syn_X, np.tile(real_X, (real_weight, 1))])
            y = np.concatenate([syn_y, np.tile(real_y, real_weight)])
    else:
        X, y = syn_X, syn_y

    logger.info(f"Stability model: training on {len(X)} samples.")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ])
    pipeline.fit(X, y)

    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")
    logger.info(f"Stability CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    importances = pipeline.named_steps["clf"].feature_importances_
    feat_importance = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
    logger.info("Stability feature importances:")
    for name, imp in feat_importance[:6]:
        logger.info(f"  {name:<30} {imp:.3f}")

    model = StabilityModel(pipeline=pipeline, cv_auc=cv_scores.mean())
    model.save()
    return model


def load() -> Optional["StabilityModel"]:
    if not MODEL_PATH.exists():
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"Could not load stability model: {e}")
        return None


def load_or_train(real_X=None, real_y=None) -> "StabilityModel":
    model = load()
    if model is None:
        logger.info("No stability model found — training now...")
        model = train(real_X, real_y)
    return model


class StabilityModel:

    def __init__(self, pipeline, cv_auc: float = 0.0):
        self.pipeline = pipeline
        self.cv_auc = cv_auc

    def predict(self, feature_vector: np.ndarray) -> Dict:
        """
        Assess valuation stability for a single stock.

        Returns dict with:
          is_stable:   bool
          confidence:  float 0–1
          warnings:    list of human-readable instability reasons
        """
        X = feature_vector.reshape(1, -1)
        proba_stable = self.pipeline.predict_proba(X)[0][1]
        is_stable = proba_stable >= 0.5

        # Generate human-readable warnings regardless of model output
        # These give the user actionable context even if the model says "stable"
        feat_dict = dict(zip(FEATURE_NAMES, feature_vector))
        warnings = self._generate_warnings(feat_dict, proba_stable)

        return {
            "is_stable": is_stable,
            "confidence": round(proba_stable, 3),
            "stability_score": round(proba_stable * 100, 1),
            "warnings": warnings,
        }

    def _generate_warnings(self, feats: Dict[str, float], proba: float) -> list:
        warnings = []

        if feats.get("terminal_value_pct", 0) > 0.75:
            warnings.append(
                f"Terminal value is {feats['terminal_value_pct']:.0%} of EV — "
                "most of the valuation depends on post-year-5 assumptions."
            )
        if feats.get("fcf_stability", 1) < 0.40:
            warnings.append(
                "FCF history is inconsistent — base FCF may not be representative."
            )
        if feats.get("bear_base_spread", 0) > 0.80:
            warnings.append(
                "Wide bear-bull spread — fair value is highly assumption-dependent."
            )
        if feats.get("leverage_ratio", 0) > 6.0:
            warnings.append(
                f"Leverage is {feats['leverage_ratio']:.1f}x FCF — "
                "equity value is sensitive to debt refinancing risk."
            )
        if proba < 0.35:
            warnings.append(
                "Model confidence in stability is low — treat this valuation as a rough estimate only."
            )

        return warnings

    def feature_importances(self) -> dict:
        importances = self.pipeline.named_steps["clf"].feature_importances_
        return dict(sorted(
            zip(FEATURE_NAMES, importances),
            key=lambda x: x[1], reverse=True
        ))

    def save(self):
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Stability model saved to {MODEL_PATH}")
