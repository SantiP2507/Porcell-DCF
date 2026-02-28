"""
ml/trainer.py — Training pipeline orchestrator.

Handles:
  1. Loading real training data from Supabase (when available)
  2. Training all three models (prioritizer, stability, clustering)
  3. Retraining schedule (weekly retrain when enough real data exists)
  4. Model versioning / replacement

Run manually:
  python -c "from ml.trainer import train_all; train_all()"

Or it's called automatically by main.py on the first run (no models exist)
and weekly thereafter.
"""

import logging
import json
from datetime import date, timedelta
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def train_all(force: bool = False) -> dict:
    """
    Train all three ML models.

    Args:
        force: If True, retrain even if models already exist.

    Returns:
        Dict with training results for each model.
    """
    from ml.prioritizer import train as train_prioritizer, load as load_prioritizer
    from ml.stability import train as train_stability, load as load_stability
    from ml.clustering import train as train_clustering, load as load_clustering

    results = {}

    # Load real data from Supabase for blending
    real_data = _load_real_training_data()

    # ── Prioritization Model ───────────────────────────────────────────────
    if force or load_prioritizer() is None:
        logger.info("Training prioritization model...")
        real_X = real_data.get("prioritizer_X")
        real_y = real_data.get("prioritizer_y")
        model = train_prioritizer(real_X, real_y)
        results["prioritizer"] = {
            "status": "trained",
            "cv_auc": model.cv_auc,
            "top_features": list(model.feature_importances().keys())[:3],
        }
    else:
        logger.info("Prioritizer model already exists — skipping (use force=True to retrain).")
        results["prioritizer"] = {"status": "skipped"}

    # ── Stability Model ────────────────────────────────────────────────────
    if force or load_stability() is None:
        logger.info("Training stability model...")
        real_X = real_data.get("stability_X")
        real_y = real_data.get("stability_y")
        model = train_stability(real_X, real_y)
        results["stability"] = {
            "status": "trained",
            "cv_auc": model.cv_auc,
            "top_features": list(model.feature_importances().keys())[:3],
        }
    else:
        logger.info("Stability model already exists — skipping.")
        results["stability"] = {"status": "skipped"}

    # ── Clustering Model ───────────────────────────────────────────────────
    if force or load_clustering() is None:
        logger.info("Training clustering model...")
        real_X = real_data.get("clustering_X")
        model = train_clustering(real_X)
        results["clustering"] = {
            "status": "trained",
            "cluster_labels": model.cluster_labels,
        }
    else:
        logger.info("Clustering model already exists — skipping.")
        results["clustering"] = {"status": "skipped"}

    logger.info(f"Training complete: {results}")
    return results


def should_retrain() -> bool:
    """
    Returns True if it's time to retrain models.
    Retrains weekly once enough real data has accumulated.
    """
    from ml.prioritizer import MODEL_PATH
    from config import BASE_DIR as ROOT_DIR

    retrain_flag = ROOT_DIR / "ml" / "models" / "last_retrain.txt"

    if not MODEL_PATH.exists():
        return True  # never trained

    if not retrain_flag.exists():
        return True  # no record of last retrain

    try:
        last = date.fromisoformat(retrain_flag.read_text().strip())
        return (date.today() - last).days >= 7
    except Exception:
        return True


def mark_retrained():
    """Record today as the last retrain date."""
    from config import BASE_DIR as ROOT_DIR
    flag = ROOT_DIR / "ml" / "models" / "last_retrain.txt"
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text(date.today().isoformat())


def _load_real_training_data() -> dict:
    """
    Load real historical data from Supabase for ML training.

    Returns dict with arrays for each model, or empty dict if no data.
    Requires at least MIN_REAL_SAMPLES rows to be useful.
    """
    try:
        from db.supabase_client import _get_client
        from ml.bootstrap import FEATURE_NAMES

        client = _get_client()
        if client is None:
            logger.info("No Supabase client — using synthetic data only.")
            return {}

        # ── Load valuation history ─────────────────────────────────────────
        response = (
            client.table("valuations")
            .select("*")
            .eq("scenario", "base")
            .order("date", desc=False)
            .limit(2000)
            .execute()
        )
        rows = response.data if hasattr(response, "data") else []

        if not rows or len(rows) < 10:
            logger.info(f"Only {len(rows)} real valuation rows — using synthetic bootstrap.")
            return {}

        logger.info(f"Loaded {len(rows)} real valuation rows from Supabase.")

        # ── Extract features from stored assumptions JSON ──────────────────
        feature_rows = []
        for row in rows:
            try:
                assumptions = json.loads(row.get("assumptions") or "{}")
                feat = _row_to_features(row, assumptions)
                if feat is not None:
                    feature_rows.append(feat)
            except Exception as e:
                logger.debug(f"Skipping malformed row: {e}")

        if len(feature_rows) < 10:
            return {}

        X = np.array(feature_rows)

        # ── Prioritization labels ──────────────────────────────────────────
        # Proxy: was the base upside > 25% AND bear upside > -15%?
        # (In the future: replace with actual outcome — did price converge to fair value?)
        pri_y = np.array([
            1 if (r.get("base_value", 0) / max(r.get("market_price", 1), 1) - 1 > 0.25)
            else 0
            for r in rows[:len(feature_rows)]
        ])

        # ── Stability labels ───────────────────────────────────────────────
        # Proxy: TV% < 70% AND bear/bull spread not too wide
        stab_y = np.array([
            1 if (r.get("terminal_value_pct", 1) < 0.70)
            else 0
            for r in rows[:len(feature_rows)]
        ])

        return {
            "prioritizer_X": X,
            "prioritizer_y": pri_y,
            "stability_X": X,
            "stability_y": stab_y,
            "clustering_X": X,
        }

    except Exception as e:
        logger.warning(f"Failed to load real training data: {e}")
        return {}


def _row_to_features(row: dict, assumptions: dict) -> Optional[list]:
    """Convert a Supabase valuation row into an ML feature vector."""
    try:
        from ml.bootstrap import generate_stock_features, features_to_array

        market_price = float(row.get("market_price") or 0)
        base_value   = float(row.get("base_value") or 0)
        bear_value   = float(row.get("bear_value") or 0)
        bull_value   = float(row.get("bull_value") or 0)
        tv_pct       = float(row.get("terminal_value_pct") or 0.65)

        if market_price <= 0 or base_value <= 0:
            return None

        valuation_gap = (base_value - market_price) / market_price
        bear_upside   = (bear_value - market_price) / market_price if bear_value else valuation_gap - 0.3
        bull_upside   = (bull_value - market_price) / market_price if bull_value else valuation_gap + 0.4

        # Reasonable defaults for fields not stored in valuations table
        fcf_stability = 0.6
        leverage_ratio = 2.0
        fcf_yield = 0.04
        fcf_growth_3y = float(assumptions.get("fcf_growth_rate", 0.07))

        feats = generate_stock_features(
            valuation_gap=valuation_gap,
            bear_upside=bear_upside,
            bull_upside=bull_upside,
            fcf_stability=fcf_stability,
            leverage_ratio=leverage_ratio,
            fcf_yield=fcf_yield,
            terminal_value_pct=tv_pct,
            fcf_growth_3y=fcf_growth_3y,
            market_price=market_price,
            bear_fv=bear_value,
            bull_fv=bull_value,
        )
        return list(features_to_array(feats))

    except Exception:
        return None
