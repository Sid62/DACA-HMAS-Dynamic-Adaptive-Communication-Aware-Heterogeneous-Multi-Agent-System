"""Semantic Plan Cache for Cloud LLM Response Reuse (Optimization 4)."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

# Canonical list of known domain skills
ALL_KNOWN_SKILLS = [
    "comm_relay",
    "inspect",
    "lift",
    "navigate",
    "patrol",
    "recon",
    "repair",
    "rescue",
    "sense",
    "transport",
]


@dataclass
class CacheEntry:
    timestamp: float
    step: int
    state_vector: np.ndarray
    state_hash: str
    result: Any
    tokens: int
    latency: float
    operation: str = "decompose"
    constraints: dict[str, Any] = field(default_factory=dict)


class SemanticPlanCache:
    """Constraint-aware semantic plan cache that separates operations

    (decomposition vs. coalition) and validates feasibility before reuse.
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
        self.accepted_reuse: int = 0
        self.rejected_reuse: int = 0
        self.saved_cloud_calls: int = 0
        self.saved_tokens: int = 0
        self.saved_latency: float = 0.0
        self.miss_reason_log: list[dict] = []

    @property
    def invalid_reuse(self) -> int:
        return self.rejected_reuse

    def reset_metrics(self) -> None:
        self.cache_hits = 0
        self.cache_misses = 0
        self.accepted_reuse = 0
        self.rejected_reuse = 0
        self.saved_cloud_calls = 0
        self.saved_tokens = 0
        self.saved_latency = 0.0
        self.miss_reason_log = []

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return round(self.cache_hits / total, 4)

    def _extract_constraints(
        self, state_dict: dict[str, Any], operation: str = "decompose"
    ) -> dict[str, Any]:
        """Extract constraint signature including agent IDs, skills, subtask IDs,

        required skills, targets, agent positions, dist_mat, cqi_matrix, thresholds, task identity, and CQI."""
        agents = state_dict.get("active_agents", state_dict.get("agents", []))
        agent_ids = set()
        agent_skills: dict[str, tuple[str, ...]] = {}
        agent_positions: dict[str, tuple[float, ...]] = {}
        for a in agents:
            aid = str(a.get("id", a.get("agent_id", "")))
            if aid:
                agent_ids.add(aid)
                skills = a.get("skills", [])
                agent_skills[aid] = tuple(sorted(skills))
                pos = a.get("pos", a.get("position"))
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    agent_positions[aid] = tuple(round(float(c), 1) for c in pos[:2])

        subtasks = state_dict.get("subtasks", [])
        subtask_ids = set()
        required_skills: dict[str, tuple[str, ...]] = {}
        targets: dict[str, tuple[float, ...]] = {}
        for s in subtasks:
            sid = str(s.get("id", s.get("subtask_id", "")))
            if sid:
                subtask_ids.add(sid)
                s_skills = s.get("skills", s.get("required_skills", []))
                required_skills[sid] = tuple(sorted(s_skills))
                tgt = s.get("target")
                if isinstance(tgt, (list, tuple)) and len(tgt) >= 2:
                    targets[sid] = tuple(round(float(c), 1) for c in tgt[:2])

        if not targets and "targets" in state_dict and isinstance(state_dict["targets"], dict):
            targets = {
                str(k): tuple(round(float(c), 1) for c in v[:2])
                for k, v in state_dict["targets"].items()
                if isinstance(v, (list, tuple)) and len(v) >= 2
            }
        elif not targets and "target" in state_dict and isinstance(state_dict["target"], (list, tuple)) and len(state_dict["target"]) >= 2:
            targets["__global__"] = tuple(round(float(c), 1) for c in state_dict["target"][:2])

        raw_dist = state_dict.get("dist_mat", state_dict.get("distance_matrix"))
        if raw_dist is not None:
            if hasattr(raw_dist, "tolist"):
                raw_dist = raw_dist.tolist()
            if isinstance(raw_dist, (list, tuple)):
                dist_mat = tuple(
                    tuple(round(float(d), 1) for d in row)
                    for row in raw_dist
                    if isinstance(row, (list, tuple))
                )
            else:
                dist_mat = None
        else:
            dist_mat = None

        raw_cqi = state_dict.get("cqi_matrix", state_dict.get("cqi_mat"))
        if raw_cqi is not None:
            if hasattr(raw_cqi, "tolist"):
                raw_cqi = raw_cqi.tolist()
            if isinstance(raw_cqi, (list, tuple)):
                cqi_mat = tuple(
                    tuple(round(float(q), 2) for q in row)
                    for row in raw_cqi
                    if isinstance(row, (list, tuple))
                )
            else:
                cqi_mat = None
        else:
            cqi_mat = None

        c_task = None
        for k in ("c_task", "C_task"):
            if k in state_dict and state_dict[k] is not None:
                c_task = round(float(state_dict[k]), 2)
                break

        r_reach = None
        for k in ("r_reach", "R_reach"):
            if k in state_dict and state_dict[k] is not None:
                r_reach = round(float(state_dict[k]), 2)
                break

        gamma_min = None
        for k in ("gamma_min", "gamma", "Gamma_min"):
            if k in state_dict and state_dict[k] is not None:
                gamma_min = round(float(state_dict[k]), 3)
                break

        task_id = str(state_dict.get("task", state_dict.get("instruction", ""))).strip().lower()
        cqi = round(float(state_dict.get("cqi", 1.0)), 3)
        cqi_min = round(float(state_dict.get("cqi_min", cqi)), 3)

        return {
            "operation": operation,
            "task_identity": task_id,
            "agent_ids": agent_ids,
            "agent_skills": agent_skills,
            "agent_positions": agent_positions,
            "subtask_ids": subtask_ids,
            "required_skills": required_skills,
            "targets": targets,
            "dist_mat": dist_mat,
            "cqi_matrix": cqi_mat,
            "c_task": c_task,
            "r_reach": r_reach,
            "gamma_min": gamma_min,
            "cqi": cqi,
            "cqi_min": cqi_min,
        }

    def _constraints_compatible(
        self, entry_c: dict[str, Any], query_c: dict[str, Any]
    ) -> bool:
        """Check if cached entry constraints are compatible with query constraints."""
        # 1. Operation isolation
        if entry_c.get("operation") != query_c.get("operation"):
            return False

        # 2. Agent IDs must match
        if entry_c.get("agent_ids") != query_c.get("agent_ids"):
            return False

        # 3. Agent skills must match
        if entry_c.get("agent_skills") != query_c.get("agent_skills"):
            return False

        # 4. Subtask IDs must match
        if entry_c.get("subtask_ids") != query_c.get("subtask_ids"):
            return False

        # 5. Required skills must match
        if entry_c.get("required_skills") != query_c.get("required_skills"):
            return False

        # 6. Task identity must match (if provided)
        t_entry = entry_c.get("task_identity", "")
        t_query = query_c.get("task_identity", "")
        if t_entry and t_query and t_entry != t_query:
            return False

        # 7. Target positions compatibility (Gap 1)
        targets_entry = entry_c.get("targets", {})
        targets_query = query_c.get("targets", {})
        if targets_entry or targets_query:
            if targets_entry != targets_query:
                return False

        # 8. Distance matrix compatibility (Gap 2)
        dist_entry = entry_c.get("dist_mat")
        dist_query = query_c.get("dist_mat")
        if (dist_entry is not None) and (dist_query is not None):
            if dist_entry != dist_query:
                return False

        # 9. CQI / link matrix compatibility (Gap 2)
        cqi_mat_entry = entry_c.get("cqi_matrix")
        cqi_mat_query = query_c.get("cqi_matrix")
        if (cqi_mat_entry is not None) and (cqi_mat_query is not None):
            if cqi_mat_entry != cqi_mat_query:
                return False

        # 10. Thresholds compatibility (C_task, R_reach, gamma_min) (Gap 2)
        for key in ("c_task", "r_reach", "gamma_min"):
            val_entry = entry_c.get(key)
            val_query = query_c.get(key)
            if (val_entry is not None) and (val_query is not None):
                if val_entry != val_query:
                    return False

        # 11. Agent positions compatibility
        pos_entry = entry_c.get("agent_positions", {})
        pos_query = query_c.get("agent_positions", {})
        if pos_entry or pos_query:
            if pos_entry != pos_query:
                return False

        # 12. Network CQI degradation check
        if abs(entry_c.get("cqi", 1.0) - query_c.get("cqi", 1.0)) > 0.2:
            return False

        return True

    def _state_to_hash_and_vector(
        self, state_dict: dict[str, Any], operation: str = "decompose"
    ) -> tuple[str, np.ndarray, dict[str, Any]]:
        """Convert state summary dict into an exact hash, a normalized feature vector,

        and constraint signature."""
        constraints = self._extract_constraints(state_dict, operation)

        # Exact canonical hash
        canonical = {
            "op": operation,
            "task": constraints["task_identity"],
            "agent_ids": sorted(list(constraints["agent_ids"])),
            "agent_skills": {k: list(v) for k, v in sorted(constraints["agent_skills"].items())},
            "agent_positions": {k: list(v) for k, v in sorted(constraints["agent_positions"].items())},
            "subtask_ids": sorted(list(constraints["subtask_ids"])),
            "required_skills": {k: list(v) for k, v in sorted(constraints["required_skills"].items())},
            "targets": {k: list(v) for k, v in sorted(constraints["targets"].items())},
            "dist_mat": [list(r) for r in constraints["dist_mat"]] if constraints.get("dist_mat") else None,
            "cqi_matrix": [list(r) for r in constraints["cqi_matrix"]] if constraints.get("cqi_matrix") else None,
            "c_task": constraints.get("c_task"),
            "r_reach": constraints.get("r_reach"),
            "gamma_min": constraints.get("gamma_min"),
            "cqi": constraints["cqi"],
        }
        exact_hash = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()

        # Build rich constraint-aware feature vector:
        vec: list[float] = []

        # 1. Scaled counts
        n_agents = float(len(constraints["agent_ids"]))
        n_subtasks = float(len(constraints["subtask_ids"]))
        vec.extend([n_agents, n_subtasks, constraints["cqi"], constraints["cqi_min"]])

        # 2. Agent skill presence
        skill_to_agent_count = {sk: 0.0 for sk in ALL_KNOWN_SKILLS}
        for s_tuple in constraints["agent_skills"].values():
            for sk in s_tuple:
                if sk in skill_to_agent_count:
                    skill_to_agent_count[sk] += 1.0
                else:
                    h = int(hashlib.md5(sk.encode()).hexdigest(), 16) % len(ALL_KNOWN_SKILLS)
                    skill_to_agent_count[ALL_KNOWN_SKILLS[h]] += 1.0
        vec.extend([skill_to_agent_count[sk] for sk in ALL_KNOWN_SKILLS])

        # 3. Subtask required skill presence
        skill_to_req_count = {sk: 0.0 for sk in ALL_KNOWN_SKILLS}
        for s_tuple in constraints["required_skills"].values():
            for sk in s_tuple:
                if sk in skill_to_req_count:
                    skill_to_req_count[sk] += 1.0
                else:
                    h = int(hashlib.md5(sk.encode()).hexdigest(), 16) % len(ALL_KNOWN_SKILLS)
                    skill_to_req_count[ALL_KNOWN_SKILLS[h]] += 1.0
        vec.extend([skill_to_req_count[sk] for sk in ALL_KNOWN_SKILLS])

        # 4. Agent ID hash projection (8 buckets)
        aid_buckets = [0.0] * 8
        for aid in constraints["agent_ids"]:
            h = int(hashlib.md5(aid.encode()).hexdigest(), 16) % 8
            aid_buckets[h] += 1.0
        vec.extend(aid_buckets)

        # 5. Subtask ID hash projection (8 buckets)
        sid_buckets = [0.0] * 8
        for sid in constraints["subtask_ids"]:
            h = int(hashlib.md5(sid.encode()).hexdigest(), 16) % 8
            sid_buckets[h] += 1.0
        vec.extend(sid_buckets)

        # 6. Task identity hash projection (4 buckets)
        task_buckets = [0.0] * 4
        if constraints["task_identity"]:
            h = int(hashlib.md5(constraints["task_identity"].encode()).hexdigest(), 16) % 4
            task_buckets[h] += 1.0
        vec.extend(task_buckets)

        # 7. Coordinates summary
        if constraints["targets"]:
            xs = [t[0] for t in constraints["targets"].values() if len(t) > 0]
            ys = [t[1] for t in constraints["targets"].values() if len(t) > 1]
            avg_x = float(np.mean(xs)) if xs else 0.0
            avg_y = float(np.mean(ys)) if ys else 0.0
        else:
            avg_x, avg_y = 0.0, 0.0
        vec.extend([avg_x, avg_y])

        # 8. Operation indicator
        vec.extend([1.0 if operation == "decompose" else 0.0, 1.0 if operation == "coalition" else 0.0])

        arr = np.array(vec, dtype=float)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return exact_hash, arr, constraints

    def _default_validate(self, result: Any, constraints: dict[str, Any]) -> bool:
        """Validate cached result against current constraints."""
        op = constraints.get("operation", "decompose")
        if op == "decompose":
            if not isinstance(result, dict) or not result:
                return False
            assignments_map = (
                result.get("assignments")
                if isinstance(result.get("assignments"), dict)
                else result
            )
            agent_ids = constraints.get("agent_ids", set())
            agent_skills = constraints.get("agent_skills", {})
            agent_positions = constraints.get("agent_positions", {})
            subtask_ids = constraints.get("subtask_ids", set())
            required_skills = constraints.get("required_skills", {})
            targets = constraints.get("targets", {})
            r_reach = constraints.get("r_reach") or 100.0
            c_task = constraints.get("c_task") or 30.0

            for sid in subtask_ids:
                assigned = assignments_map.get(sid, [])
                if not assigned or not isinstance(assigned, list):
                    return False
                team_skills = set()
                for aid in assigned:
                    if aid not in agent_ids:
                        return False
                    team_skills.update(agent_skills.get(aid, ()))
                req = set(required_skills.get(sid, ()))
                if not req.issubset(team_skills):
                    return False

                # Feasibility check on reachability & joint distance if positions and targets known
                target = targets.get(sid)
                if target and agent_positions:
                    for aid in assigned:
                        pos = agent_positions.get(aid)
                        if pos:
                            d = math.hypot(pos[0] - target[0], pos[1] - target[1])
                            if d > r_reach:
                                return False
                    if len(assigned) > 1:
                        for i in range(len(assigned)):
                            for j in range(i + 1, len(assigned)):
                                p1 = agent_positions.get(assigned[i])
                                p2 = agent_positions.get(assigned[j])
                                if p1 and p2:
                                    if math.hypot(p1[0] - p2[0], p1[1] - p2[1]) > c_task:
                                        return False
            return True
        elif op == "coalition":
            if not isinstance(result, list) or not result:
                return False
            agent_ids = constraints.get("agent_ids", set())
            seen_agents = set()
            for c in result:
                if not isinstance(c, dict):
                    return False
                members = c.get("members", [])
                if not members or not isinstance(members, list):
                    return False
                for mid in members:
                    if mid not in agent_ids:
                        return False
                    if mid in seen_agents:
                        return False
                    seen_agents.add(mid)

            # Link feasibility if cqi_matrix and gamma_min are known
            cqi_mat = constraints.get("cqi_matrix")
            gamma_min = constraints.get("gamma_min") or 0.3
            if cqi_mat and agent_ids:
                sorted_aids = sorted(list(agent_ids))
                id_to_idx = {aid: idx for idx, aid in enumerate(sorted_aids)}
                for c in result:
                    members = c.get("members", [])
                    for i in range(len(members)):
                        for j in range(i + 1, len(members)):
                            idx_i = id_to_idx.get(members[i])
                            idx_j = id_to_idx.get(members[j])
                            if (
                                idx_i is not None
                                and idx_j is not None
                                and idx_i < len(cqi_mat)
                                and idx_j < len(cqi_mat[idx_i])
                            ):
                                if cqi_mat[idx_i][idx_j] < gamma_min:
                                    return False
            return True
        return True

    def lookup(
        self,
        state_dict: dict[str, Any],
        current_step: int = 0,
        operation: str = "decompose",
        validator: Callable[[Any], bool] | None = None,
        original_tokens: int | None = None,
    ) -> Any | None:
        if not self.enabled or not self.entries:
            self.cache_misses += 1
            reason = "cache_empty" if not self.entries else "disabled"
            self.miss_reason_log.append({"step": current_step, "reason": reason})
            return None

        exact_hash, target_vec, query_constraints = self._state_to_hash_and_vector(
            state_dict, operation=operation
        )

        cur_step = 0 if current_step is None else current_step
        best_sim = -1.0
        best_entry: CacheEntry | None = None
        compatible_found = False

        for entry in self.entries:
            # Operation check
            if getattr(entry, "operation", "decompose") != operation:
                continue

            ent_step = 0 if entry.step is None else entry.step
            if cur_step - ent_step > self.max_cache_age:
                continue

            # Constraint check
            entry_constraints = getattr(entry, "constraints", {})
            if not self._constraints_compatible(entry_constraints, query_constraints):
                continue

            compatible_found = True

            if entry.state_hash == exact_hash:
                best_sim = 1.0
                best_entry = entry
                break

            # Cosine similarity
            sim = float(np.dot(target_vec, entry.state_vector))
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if not compatible_found:
            self.cache_misses += 1
            self.miss_reason_log.append({
                "step": current_step,
                "reason": "no_compatible_candidate",
                "operation": operation,
            })
            return None

        if best_entry is not None and best_sim >= self.similarity_threshold:
            # Cache hit detected! Now validate result before accepting reuse.
            self.cache_hits += 1

            is_valid = True
            if validator is not None:
                try:
                    is_valid = bool(validator(best_entry.result))
                except Exception:
                    is_valid = False
            else:
                is_valid = self._default_validate(best_entry.result, query_constraints)

            if is_valid:
                self.accepted_reuse += 1
                self.saved_cloud_calls += 1
                saved_tok = original_tokens if original_tokens is not None else best_entry.tokens
                self.saved_tokens += saved_tok
                self.saved_latency += best_entry.latency
                print(
                    f"[SEMANTIC_CACHE] HIT ACCEPTED op={operation} sim={best_sim:.3f} >= {self.similarity_threshold} "
                    f"(saved {saved_tok} tokens, call={self.saved_cloud_calls})"
                )
                return best_entry.result
            else:
                self.rejected_reuse += 1
                print(
                    f"[SEMANTIC_CACHE] HIT REJECTED op={operation} sim={best_sim:.3f} "
                    f"(feasibility validation failed)"
                )
                self.miss_reason_log.append({
                    "step": current_step,
                    "reason": "validation_failed",
                    "operation": operation,
                    "similarity": best_sim,
                })
                return None

        self.cache_misses += 1
        self.miss_reason_log.append({
            "step": current_step,
            "reason": "similarity_below_threshold",
            "operation": operation,
            "best_similarity": round(best_sim, 6),
            "threshold": self.similarity_threshold,
        })
        return None

    def put(
        self,
        state_dict: dict[str, Any],
        result: Any,
        tokens: int = 200,
        latency: float = 0.1,
        current_step: int = 0,
        operation: str = "decompose",
    ) -> None:
        if not self.enabled:
            return
        exact_hash, vec, constraints = self._state_to_hash_and_vector(
            state_dict, operation=operation
        )
        ent_step = 0 if current_step is None else current_step
        entry = CacheEntry(
            timestamp=time.perf_counter(),
            step=ent_step,
            state_vector=vec,
            state_hash=exact_hash,
            result=result,
            tokens=tokens,
            latency=latency,
            operation=operation,
            constraints=constraints,
        )
        self.entries.append(entry)
        if len(self.entries) > 50:
            self.entries.pop(0)
