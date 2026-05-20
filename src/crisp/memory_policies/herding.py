from __future__ import annotations

from collections import defaultdict
from typing import List
import numpy as np
from crisp.memory import MemoryBank, MemoryItem


class HerdingMemoryPolicy:
    def __init__(self, max_exemplars_per_class: int = 50) -> None:
        if max_exemplars_per_class <= 0:
            raise ValueError("max_exemplars_per_class must be positive.")
        self.max_exemplars_per_class = max_exemplars_per_class

    def apply(self, memory: MemoryBank) -> MemoryBank:
        if len(memory) == 0:
            return memory

        grouped = defaultdict(list)
        for item in memory.items:
            grouped[item.label].append(item)

        kept_items: List[MemoryItem] = []

        for label, items in grouped.items():
            if len(items) <= self.max_exemplars_per_class:
                kept_items.extend(items)
                continue

            embeddings = np.stack([item.embedding for item in items]).astype(np.float32)
            prototype = embeddings.mean(axis=0)
            prototype = prototype / max(np.linalg.norm(prototype), 1e-12)
            similarities = embeddings @ prototype
            selected_indices = np.argsort(similarities)[::-1][: self.max_exemplars_per_class]

            for idx in selected_indices:
                kept_items.append(items[int(idx)])

        memory.replace_items(kept_items)
        return memory
