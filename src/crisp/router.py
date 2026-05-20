from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np


class ClusterRouter:
    def __init__(
        self,
        method: str = "minibatch_kmeans",
        n_clusters: int = 3,
        random_state: int = 42,
        router_kwargs: Optional[Dict] = None,
    ) -> None:
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")
        self.method = method
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.router_kwargs = router_kwargs or {}
        self.model = None
        self.is_fitted = False

    def fit(self, embeddings: np.ndarray) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if len(embeddings) < self.n_clusters:
            self.n_clusters = max(1, len(embeddings))

        method = self.method.lower()

        if method == "kmeans":
            from sklearn.cluster import KMeans
            self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init="auto", **self.router_kwargs)
        elif method == "minibatch_kmeans":
            from sklearn.cluster import MiniBatchKMeans
            self.model = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init="auto", **self.router_kwargs)
        else:
            raise ValueError("Unsupported cluster method. Use 'kmeans' or 'minibatch_kmeans'.")

        self.model.fit(embeddings)
        self.is_fitted = True

    def predict(self, embedding: np.ndarray) -> int:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("ClusterRouter must be fitted before prediction.")
        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        return int(self.model.predict(embedding)[0])

    def predict_top_m(self, embedding: np.ndarray, top_m: int = 1) -> List[Dict]:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("ClusterRouter must be fitted before prediction.")

        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        centers = np.asarray(self.model.cluster_centers_, dtype=np.float32)
        distances = np.linalg.norm(centers - embedding, axis=1)

        top_m = min(max(1, top_m), len(centers))
        indices = np.argsort(distances)[:top_m]

        results = []
        for idx in indices:
            distance = float(distances[int(idx)])
            router_score = 1.0 / (1.0 + distance)
            results.append({
                "cluster_id": int(idx),
                "distance": distance,
                "router_score": float(router_score),
            })
        return results
