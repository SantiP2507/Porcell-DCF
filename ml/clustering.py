"""
ml/clustering.py — Stock Clustering Model.

Groups stocks into financial archetypes based on their valuation profile.
This lets you ask "find me more stocks like AAPL" or "which cluster does
this stock fall into?"

Model: KMeans clustering (unsupervised)
  - No labels needed — discovers natural groupings in the data
  - Fast inference (important for real-time screener use)
  - Number of clusters (k=6) chosen to match real-world archetypes:
      0: Deep Value        — large gap, moderate stability, high yield
      1: Quality Compounder— moderate gap, high stability, low leverage
      2: Growth/GARP       — moderate gap, high growth, lower yield
      3: Overvalued Quality— negative gap, high stability (wait for pullback)
      4: Value Trap        — large gap, low stability, high leverage
      5: Distressed        — large gap, very low stability, extreme leverage

Cluster numbers/names may shift between retrains — always read the
centroid characteristics, not the number.

Usage:
  - After screening, label each candidate with its cluster
  - Use clusters to diversify: don't fill a portfolio with cluster 0 only
  - Find similar stocks: query Supabase for other tickers in the same cluster
"""

import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from ml.bootstrap import generate_clustering_data, FEATURE_NAMES
from config import BASE_DIR as ROOT_DIR

logger = logging.getLogger(__name__)

MODEL_PATH = ROOT_DIR / "ml" / "models" / "clustering.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

N_CLUSTERS = 6

# Human-readable archetype names assigned after training by inspecting centroids.
# These are heuristic labels — re-inspect after each retrain.
ARCHETYPE_NAMES = [
    "Deep Value",
    "Quality Compounder",
    "Growth / GARP",
    "Overvalued Quality",
    "Value Trap",
    "Distressed / Speculative",
]


def train(real_X: Optional[np.ndarray] = None) -> "ClusteringModel":
    """
    Train the clustering model.

    Args:
        real_X: Real feature matrix (optional). If provided and large enough,
                blended with synthetic data.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    syn_X = generate_clustering_data(n=1000)

    if real_X is not None and len(real_X) >= 30:
        X = np.vstack([syn_X, real_X])
        logger.info(f"Clustering: training on {len(syn_X)} synthetic + {len(real_X)} real samples.")
    else:
        X = syn_X
        logger.info(f"Clustering: bootstrapping on {len(X)} synthetic samples.")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(
            n_clusters=N_CLUSTERS,
            n_init=20,          # run 20 times, pick best inertia
            max_iter=500,
            random_state=42,
        )),
    ])
    pipeline.fit(X)

    inertia = pipeline.named_steps["kmeans"].inertia_
    logger.info(f"Clustering inertia: {inertia:.1f} (lower = tighter clusters)")

    model = ClusteringModel(pipeline=pipeline)
    model._label_clusters()
    model.save()
    return model


def load() -> Optional["ClusteringModel"]:
    if not MODEL_PATH.exists():
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"Could not load clustering model: {e}")
        return None


def load_or_train(real_X=None) -> "ClusteringModel":
    model = load()
    if model is None:
        logger.info("No clustering model found — training now...")
        model = train(real_X)
    return model


class ClusteringModel:

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.cluster_labels: Dict[int, str] = {}   # cluster_id → archetype name
        self.cluster_centroids: Dict[int, Dict] = {}  # cluster_id → feature means

    def predict(self, feature_vector: np.ndarray) -> Dict:
        """
        Assign a stock to a cluster.

        Returns:
            cluster_id:   int (0 to N_CLUSTERS-1)
            archetype:    str (human-readable name)
            description:  str (what this cluster looks like)
            similar_to:   str (what kind of stocks are in this cluster)
        """
        cluster_id = int(self.pipeline.predict(feature_vector.reshape(1, -1))[0])
        archetype = self.cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
        centroid = self.cluster_centroids.get(cluster_id, {})
        description = self._describe_cluster(cluster_id, centroid)

        return {
            "cluster_id": cluster_id,
            "archetype": archetype,
            "description": description,
        }

    def predict_batch(self, X: np.ndarray) -> List[Dict]:
        return [self.predict(x) for x in X]

    def _label_clusters(self):
        """
        After training, inspect centroids and assign archetype names.
        Uses heuristic rules on the centroid feature values.
        """
        kmeans = self.pipeline.named_steps["kmeans"]
        scaler = self.pipeline.named_steps["scaler"]

        # Inverse-transform centroids back to original feature space
        centroids_scaled = kmeans.cluster_centers_
        centroids = scaler.inverse_transform(centroids_scaled)

        for i, centroid in enumerate(centroids):
            feat = dict(zip(FEATURE_NAMES, centroid))
            self.cluster_centroids[i] = feat
            self.cluster_labels[i] = self._classify_centroid(feat)

        logger.info("Cluster archetypes assigned:")
        for cid, name in sorted(self.cluster_labels.items()):
            c = self.cluster_centroids[cid]
            logger.info(
                f"  Cluster {cid}: {name:<28} "
                f"gap={c.get('valuation_gap', 0):.0%} "
                f"stab={c.get('fcf_stability', 0):.2f} "
                f"lev={c.get('leverage_ratio', 0):.1f}x"
            )

    def _classify_centroid(self, feat: Dict[str, float]) -> str:
        """Assign an archetype name based on centroid feature values."""
        gap   = feat.get("valuation_gap", 0)
        stab  = feat.get("fcf_stability", 0.5)
        lev   = feat.get("leverage_ratio", 0)
        yield_ = feat.get("fcf_yield", 0)
        bear  = feat.get("bear_upside", 0)

        if gap < 0:
            return "Overvalued Quality" if stab > 0.6 else "Overvalued / Avoid"
        if lev > 8:
            return "Distressed / Speculative"
        if lev > 5 and stab < 0.4:
            return "Value Trap"
        if gap > 0.50 and yield_ > 0.06:
            return "Deep Value"
        if stab > 0.70 and lev < 3 and gap > 0.15:
            return "Quality Compounder"
        if feat.get("fcf_growth_3y", 0) > 0.12 and gap > 0.10:
            return "Growth / GARP"
        return "Moderate Value"

    def _describe_cluster(self, cluster_id: int, centroid: Dict) -> str:
        """Generate a human-readable description of a cluster."""
        if not centroid:
            return "Unknown cluster."
        gap  = centroid.get("valuation_gap", 0)
        stab = centroid.get("fcf_stability", 0)
        lev  = centroid.get("leverage_ratio", 0)
        tv   = centroid.get("terminal_value_pct", 0)
        descriptions = {
            "Deep Value":              f"Cheap stocks with {gap:.0%} avg upside, high FCF yield — look for catalyst.",
            "Quality Compounder":      f"High-quality businesses at {gap:.0%} discount, consistent FCF, low debt.",
            "Growth / GARP":           f"Growth companies at reasonable prices — verify growth assumptions carefully.",
            "Overvalued Quality":      f"Good businesses, but priced above DCF fair value. Wait for pullback.",
            "Value Trap":              f"Appears cheap but FCF is inconsistent and leverage is high ({lev:.1f}x). Be careful.",
            "Distressed / Speculative":f"High upside IF the business survives — very high risk, not for conservative portfolios.",
            "Overvalued / Avoid":      f"Priced above intrinsic value with no compelling quality offset.",
            "Moderate Value":          f"Modest {gap:.0%} upside with average stability. No strong signal either way.",
        }
        name = self.cluster_labels.get(cluster_id, "")
        return descriptions.get(name, f"Cluster with {gap:.0%} avg valuation gap and {stab:.2f} stability score.")

    def feature_importances(self) -> Dict[str, float]:
        """
        Approximate feature importance for clustering via centroid variance.
        Features with high variance across centroids are most discriminating.
        """
        centroids = np.array([
            [v for _, v in sorted(c.items())]
            for c in self.cluster_centroids.values()
        ])
        variances = np.var(centroids, axis=0)
        names = sorted(self.cluster_centroids[0].keys())
        return dict(sorted(zip(names, variances), key=lambda x: x[1], reverse=True))

    def save(self):
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Clustering model saved to {MODEL_PATH}")
