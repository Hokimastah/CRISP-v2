from __future__ import annotations

import pickle
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from .encoders import build_encoder
from .memory import MemoryBank
from .memory_policies import HerdingMemoryPolicy
from .retrievers import build_retriever
from .router import ClusterRouter
from .utils import infer_label_from_parent, list_images
from .voting import majority_vote, weighted_vote


class ClusteredCRISPClassifier:
    def __init__(
        self,
        router_encoder: str = "resnet18",
        cluster_method: str = "minibatch_kmeans",
        n_clusters: int = 3,
        cluster_top_m: int = 1,
        specialist_backbones: Optional[Dict[int, str]] = None,
        specialist_encoder_kwargs: Optional[Dict[int, Dict[str, Any]]] = None,
        retriever: str = "numpy",
        retriever_kwargs: Optional[Dict[str, Any]] = None,
        voting: str = "weighted",
        top_k: int = 5,
        max_exemplars_per_class: Optional[int] = 50,
        memory_policy: str = "herding",
        device: Optional[str] = None,
        pretrained: bool = True,
        router_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.router_encoder_name = router_encoder
        self.cluster_method = cluster_method
        self.n_clusters = n_clusters
        self.cluster_top_m = cluster_top_m
        self.specialist_backbones = specialist_backbones or {i: "resnet50" for i in range(n_clusters)}
        self.specialist_encoder_kwargs = specialist_encoder_kwargs or {}
        self.retriever_name = retriever
        self.retriever_kwargs = retriever_kwargs or {}
        self.voting = voting
        self.top_k = top_k
        self.max_exemplars_per_class = max_exemplars_per_class
        self.memory_policy = memory_policy
        self.device = device
        self.pretrained = pretrained
        self.router_kwargs = router_kwargs or {}

        self.router_encoder = build_encoder(
            encoder=router_encoder,
            device=device,
            pretrained=pretrained,
            encoder_kwargs=self.router_kwargs.get("encoder_kwargs", {}),
        )

        self.router = ClusterRouter(
            method=cluster_method,
            n_clusters=n_clusters,
            random_state=self.router_kwargs.get("random_state", 42),
            router_kwargs=self.router_kwargs.get("cluster_kwargs", {}),
        )

        self.specialists: Dict[int, Any] = {}
        self.memory_banks: Dict[int, MemoryBank] = {i: MemoryBank() for i in range(n_clusters)}
        self.retrievers: Dict[int, Any] = {}

        self._build_specialists_and_retrievers()

    def _build_specialists_and_retrievers(self) -> None:
        for cluster_id in range(self.n_clusters):
            backbone = self.specialist_backbones.get(cluster_id, "resnet50")
            encoder_kwargs = self.specialist_encoder_kwargs.get(cluster_id, {})
            self.specialists[cluster_id] = build_encoder(
                encoder=backbone,
                device=self.device,
                pretrained=self.pretrained,
                encoder_kwargs=encoder_kwargs,
            )
            self.retrievers[cluster_id] = build_retriever(
                retriever=self.retriever_name,
                retriever_kwargs=self.retriever_kwargs,
            )
            self.memory_banks.setdefault(cluster_id, MemoryBank())

    def _apply_memory_policy(self, cluster_id: int) -> None:
        if self.memory_policy == "none" or self.max_exemplars_per_class is None:
            return
        if self.memory_policy != "herding":
            raise ValueError("Unsupported memory_policy. Use 'herding' or 'none'.")
        policy = HerdingMemoryPolicy(max_exemplars_per_class=self.max_exemplars_per_class)
        policy.apply(self.memory_banks[cluster_id])

    def _rebuild_retriever(self, cluster_id: int) -> None:
        self.retrievers[cluster_id].build(self.memory_banks[cluster_id])

    def _fit_router_from_paths(self, image_paths: List[str]) -> None:
        router_embeddings = []
        for image_path in tqdm(image_paths, desc="Extracting router embeddings"):
            emb = self.router_encoder.encode_path(str(image_path))
            router_embeddings.append(emb)
        router_embeddings = np.stack(router_embeddings).astype(np.float32)
        self.router.fit(router_embeddings)

    def add_folder(self, folder: str, fit_router: bool = True) -> None:
        image_paths = list_images(folder)
        if not image_paths:
            raise ValueError(f"No images found in folder: {folder}")

        if fit_router or not self.router.is_fitted:
            self._fit_router_from_paths([str(p) for p in image_paths])

        for image_path in tqdm(image_paths, desc="Indexing images into clustered memory"):
            label = infer_label_from_parent(image_path)
            self.add_image(str(image_path), label=label, fit_router=False, rebuild_index=False)

        for cluster_id in range(self.n_clusters):
            self._apply_memory_policy(cluster_id)
            self._rebuild_retriever(cluster_id)

    def add_image(
        self,
        image_path: str,
        label: str,
        metadata: Optional[Dict[str, Any]] = None,
        fit_router: bool = False,
        rebuild_index: bool = True,
    ) -> int:
        router_embedding = self.router_encoder.encode_path(image_path)

        if fit_router and not self.router.is_fitted:
            self.router.fit(np.asarray([router_embedding], dtype=np.float32))

        cluster_id = self.router.predict(router_embedding)

        specialist = self.specialists[cluster_id]
        embedding = specialist.encode_path(image_path)

        meta = metadata or {}
        meta.setdefault("path", image_path)
        meta["cluster_id"] = cluster_id
        meta["router_encoder"] = self.router_encoder_name
        meta["specialist_backbone"] = self.specialist_backbones.get(cluster_id, "resnet50")

        self.memory_banks[cluster_id].add(
            embedding=embedding,
            label=label,
            metadata=meta,
        )

        if rebuild_index:
            self._apply_memory_policy(cluster_id)
            self._rebuild_retriever(cluster_id)

        return cluster_id

    def predict(
        self,
        image_path: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        cluster_top_m: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self.router.is_fitted:
            raise RuntimeError("Cluster router is not fitted. Call add_folder() or load() first.")

        router_embedding = self.router_encoder.encode_path(image_path)
        selected_clusters = self.router.predict_top_m(
            router_embedding,
            top_m=cluster_top_m or self.cluster_top_m,
        )

        all_neighbors = []

        for route in selected_clusters:
            cluster_id = int(route["cluster_id"])

            if self.memory_banks[cluster_id].is_empty():
                continue

            specialist = self.specialists[cluster_id]
            query_embedding = specialist.encode_path(image_path)

            neighbors = self.retrievers[cluster_id].search(
                query_embedding=query_embedding,
                memory=self.memory_banks[cluster_id],
                top_k=top_k or self.top_k,
            )

            for neighbor in neighbors:
                neighbor["cluster_id"] = cluster_id
                neighbor["router_score"] = float(route["router_score"])
                neighbor["routed_distance"] = float(route["distance"])
                neighbor["raw_similarity"] = float(neighbor["similarity"])
                neighbor["similarity"] = float(neighbor["similarity"]) * float(route["router_score"])

            all_neighbors.extend(neighbors)

        all_neighbors = sorted(all_neighbors, key=lambda x: x["similarity"], reverse=True)

        if not all_neighbors:
            return {
                "status": "empty_memory",
                "predicted_label": None,
                "scores": {},
                "best_similarity": None,
                "selected_clusters": selected_clusters,
                "neighbors": [],
            }

        vote_result = weighted_vote(all_neighbors) if self.voting == "weighted" else majority_vote(all_neighbors)

        best_similarity = float(all_neighbors[0]["similarity"])
        status = "known"
        predicted_label = vote_result["predicted_label"]

        if threshold is not None and best_similarity < threshold:
            status = "unknown"
            predicted_label = None

        return {
            "status": status,
            "predicted_label": predicted_label,
            "scores": vote_result["scores"],
            "best_similarity": best_similarity,
            "selected_clusters": selected_clusters,
            "neighbors": all_neighbors[: top_k or self.top_k],
            "router_encoder": self.router_encoder_name,
            "retriever": self.retriever_name,
        }

    def save(self, path: str) -> None:
        state = {
            "config": {
                "router_encoder": self.router_encoder_name,
                "cluster_method": self.cluster_method,
                "n_clusters": self.n_clusters,
                "cluster_top_m": self.cluster_top_m,
                "specialist_backbones": self.specialist_backbones,
                "specialist_encoder_kwargs": self.specialist_encoder_kwargs,
                "retriever": self.retriever_name,
                "retriever_kwargs": self.retriever_kwargs,
                "voting": self.voting,
                "top_k": self.top_k,
                "max_exemplars_per_class": self.max_exemplars_per_class,
                "memory_policy": self.memory_policy,
                "device": self.device,
                "pretrained": self.pretrained,
                "router_kwargs": self.router_kwargs,
            },
            "router": self.router,
            "memory_banks": {
                cluster_id: memory.to_serializable()
                for cluster_id, memory in self.memory_banks.items()
            },
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.router = state["router"]

        loaded_memory = {}
        for cluster_id, data in state["memory_banks"].items():
            loaded_memory[int(cluster_id)] = MemoryBank.from_serializable(data)

        self.memory_banks = {i: loaded_memory.get(i, MemoryBank()) for i in range(self.n_clusters)}

        self._build_specialists_and_retrievers()

        for cluster_id in range(self.n_clusters):
            self._rebuild_retriever(cluster_id)

    def summary(self) -> Dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "cluster_top_m": self.cluster_top_m,
            "memory_sizes": {cluster_id: len(memory) for cluster_id, memory in self.memory_banks.items()},
            "specialist_backbones": self.specialist_backbones,
            "retriever": self.retriever_name,
            "memory_policy": self.memory_policy,
            "max_exemplars_per_class": self.max_exemplars_per_class,
        }
