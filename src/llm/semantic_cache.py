"""Semantic Plan Cache for Cloud LLM Response Reuse (Optimization 4)."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CacheEntry:
    timestamp: float
    step: int
    state_vector: np.ndarray
    state_hash: str
    result: Any
    tokens: int
    latency: float


class SemanticPlanCache:
    """Semantic plan cache that reuses previous Cloud planning decisions

    when state similarity exceeds similarity_threshold (Optimization 4).
    """

    def __init__(
        self,
        enabled: bool = True,
        similarity_threshold: float = 0.90,
        max_cache_age: int = 15,
    ):
        self.enabled = enabled
        self.similarity_threshold = similarity_threshold
        self.max_cache_age = max_cache_age

        self.entries: list[CacheEntry] = []
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.saved_cloud_calls: int = 0
        self.saved_tokens: int = 0
        self.saved_latency: float = 0.0

    def reset_metrics(self) -> None:
        self.cache_hits = 0
        self.cache_misses = 0
        self.saved_cloud_calls = 0
        self.saved_tokens = 0
        self.saved_latency = 0.0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return round(self.cache_hits / total, 4)

    def _state_to_hash_and_vector(self, state_dict: dict[str, Any]) -> tuple[str, np.ndarray]:
        """Convert state summary dict into an exact hash and a normalized feature vector."""
        serialized = json.dumps(state_dict, sort_keys=True, default=str)
        exact_hash = hashlib.sha256(serialized.encode()).hexdigest()

        # Build feature vector: agent count, subtask count, cqi, completed ratio, skill counts
        agents = state_dict.get("active_agents", state_dict.get("agents", []))
        subtasks = state_dict.get("subtasks", [])
        sys_cqi = float(state_dict.get("cqi", 1.0))
        n_agents = float(len(agents))
        n_subtasks = float(len(subtasks))

        vec = [n_agents, n_subtasks, sys_cqi]
        # Position averages if available
        targets = [s.get("target") for s in subtasks if isinstance(s.get("target"), (list, tuple))]
        if targets:
            avg_x = float(np.mean([t[0] for t in targets if len(t) > 0]))
            avg_y = float(np.mean([t[1] for t in targets if len(t) > 1]))
        else:
            avg_x, avg_y = 0.0, 0.0
        vec.extend([avg_x, avg_y])

        arr = np.array(vec, dtype=float)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return exact_hash, arr

    def lookup(
        self,
        state_dict: dict[str, Any],
        current_step: int = 0,
    ) -> Any | None:
        if not self.enabled or not self.entries:
            self.cache_misses += 1
            return None

        exact_hash, target_vec = self._state_to_hash_and_vector(state_dict)

        best_sim = -1.0
        best_entry: CacheEntry | None = None

        cur_step = 0 if current_step is None else current_step
        for entry in self.entries:
            ent_step = 0 if entry.step is None else entry.step
            if cur_step - ent_step > self.max_cache_age:
                continue

            if entry.state_hash == exact_hash:
                best_sim = 1.0
                best_entry = entry
                break

            # Cosine similarity
            sim = float(np.dot(target_vec, entry.state_vector))
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry is not None and best_sim >= self.similarity_threshold:
            self.cache_hits += 1
            self.saved_cloud_calls += 1
            self.saved_tokens += best_entry.tokens
            self.saved_latency += best_entry.latency
            print(f"[SEMANTIC_CACHE] HIT sim={best_sim:.3f} >= {self.similarity_threshold} (saved {best_entry.tokens} tokens)")
            return best_entry.result

        self.cache_misses += 1
        return None

    def put(
        self,
        state_dict: dict[str, Any],
        result: Any,
        tokens: int = 200,
        latency: float = 0.1,
        current_step: int = 0,
    ) -> None:
        if not self.enabled:
            return
        exact_hash, vec = self._state_to_hash_and_vector(state_dict)
        ent_step = 0 if current_step is None else current_step
        entry = CacheEntry(
            timestamp=time.perf_counter(),
            step=ent_step,
            state_vector=vec,
            state_hash=exact_hash,
            result=result,
            tokens=tokens,
            latency=latency,
        )
        self.entries.append(entry)
        if len(self.entries) > 50:
            self.entries.pop(0)
