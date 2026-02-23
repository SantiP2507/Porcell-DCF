"""
ml/prioritizer.py — Research Prioritization Model.

Replaces the fixed scoring weights in the rule-based screener with a
trained Random Forest classifier that learns which combinations of features
predict "worth researching."

Model: Random Forest Classifier
  - Interpretable: feature importances tell you exactly what drives the score
  - Robust to outliers (trees, not linear)
  - Works well with small datasets (good for early real-data phase)

Output: probability score 0.0–1.0 that a stock is worth researching.
This replaces the hand-weighted composite score in screener.py when the
model has been trained.

Training data:
  Phase 1 (now):   Synthetic bootstrap data
  Phase 2 (later): Real Supabase data blended with synthetic
  Phase 3 (6mo+):  Real data only, retrained weekly
"""

import os
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from ml.bootstrap import generate_prioritization_data, FEATURE_NAMES
from config import BASE_DIR as ROOT_DIR

logger = logging.getLogger(__name__)

MODEL_PATH = ROOT_DIR / "ml" / "models" / "prioritizer.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# Minimum real samples before we start blending real + synthetic data
MIN_REAL_SAMPLES_FOR_BLEND = 50
# Minimum real samples before we drop synthetic entirely
MIN_REAL_SAMPLES_ONLY = 500


def train(real_X: Optional[np.ndarray] = None,
          real_y: Optional[np.ndarray] = None) -> "PrioritizationModel":
    """
    Train the prioritization model.

    Args:
        real_X: Real feature matrix from Supabase (None = bootstrap only)
        real_y: Real labels from Supabase (None = bootstrap only)

    Returns:
        Trained PrioritizationModel, also saved to disk.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score

    # ── Build training dataset ────────────────────────────────────────────
    syn_X, syn_y = generate_prioritization_data(n=2000)

    if real_X is not None and len(real_X) >= MIN_REAL_SAMPLES_FOR_BLEND:
        if len(real_X) >= MIN_REAL_SAMPLES_ONLY:
            # Enough real data — use only real
            X, y = real_X, real_y
            logger.info(f"Prioritizer: training on {len(X)} real samples only.")
        else:
            # Blend: weight real data 3x synthetic
            real_weight = 3
            X = np.vstack([syn_X, np.tile(real_X, (real_weight, 1))])
            y = np.concatenate([syn_y, np.tile(real_y, real_weight)])
            logger.info(
                f"Prioritizer: blending {len(syn_X)} synthetic + "
                f"{len(real_X)} real ({real_weight}x weighted)."
            )
    else:
        X, y = syn_X, syn_y
        logger.info(f"Prioritizer: bootstrapping on {len(X)} synthetic samples.")

    # ── Train ─────────────────────────────────────────────────────────────
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",  # handles imbalanced labels
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(X, y)

    # ── Cross-validation score ────────────────────────────────────────────
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")
    logger.info(
        f"Prioritizer CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
    )

    # ── Feature importances ───────────────────────────────────────────────
    importances = pipeline.named_steps["clf"].feature_importances_
    feat_importance = sorted(
        zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True
    )
    logger.info("Prioritizer feature importances:")
    for name, imp in feat_importance:
        logger.info(f"  {name:<30} {imp:.3f}")

    model = PrioritizationModel(pipeline=pipeline, cv_auc=cv_scores.mean())
    model.save()
    return model


def load() -> Optional["PrioritizationModel"]:
    """Load trained model from disk. Returns None if not trained yet."""
    if not MODEL_PATH.exists():
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"Could not load prioritizer model: {e}")
        return None


def load_or_train(real_X=None, real_y=None) -> "PrioritizationModel":
    """Load existing model or train a new one."""
    model = load()
    if model is None:
        logger.info("No prioritizer model found — training now...")
        model = train(real_X, real_y)
    return model


class PrioritizationModel:
    """Wrapper around the trained sklearn pipeline."""

    def __init__(self, pipeline, cv_auc: float = 0.0):
        self.pipeline = pipeline
        self.cv_auc = cv_auc

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of being worth researching (class 1)."""
        return self.pipeline.predict_proba(X)[:, 1]

    def score(self, feature_vector: np.ndarray) -> float:
        """
        Return a 0–100 priority score for a single stock.
        Higher = more worth researching.
        """
        proba = self.predict_proba(feature_vector.reshape(1, -1))[0]
        return round(proba * 100, 1)

    def feature_importances(self) -> dict:
        importances = self.pipeline.named_steps["clf"].feature_importances_
        return dict(sorted(
            zip(FEATURE_NAMES, importances),
            key=lambda x: x[1], reverse=True
        ))

    def save(self):
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Prioritizer model saved to {MODEL_PATH}")
