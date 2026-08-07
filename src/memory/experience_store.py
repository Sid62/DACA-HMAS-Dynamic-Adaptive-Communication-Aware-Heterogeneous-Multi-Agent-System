"""Subtask Experience Store — persistent plan-reuse memory layer.

This module provides a retrieval/reuse layer that sits in front of the
existing LLM planning calls.  It stores successful subtask plans keyed on a
*canonicalized signature* (invariant to exact coordinates and agent IDs) so
that conceptually similar subtasks in future missions can reuse previously
validated plans instead of invoking the LLM from scratch.

This is NOT:
- An LLM prompt cache (that already exists in ``.llm_cache/``).
- LLM fine-tuning (no model weights are touched).
- A replacement for feasibility validation (reused candidates are always
  re-validated through the same checks a fresh LLM plan goes through).
"""

from __future__ import annotations

import json
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import project_root


# ---------------------------------------------------------------------------
# Canonical signature generation
# ---------------------------------------------------------------------------

def _distance_bucket(distance: float, max_distance: float = 200.0, n_buckets: int = 10) -> int:
    """Coarse-bucket a distance into one of *n_buckets* bands.

    Follows the same style as ``QLearningCA._discretize_state`` in
    ``src/control/q_learning.py`` (``int(min(nearest, 20.0) / 2.0)``),
    scaled to the full coordinate range.
    """
    step = max_distance / n_buckets  # 20.0 by default
    return int(min(distance, max_distance) / step)


def compute_signature(
    scenario: str,
    required_skills: list[str],
    agent_types: list[str],
    lead_agent_distance: float,
) -> str:
    """Produce a canonical, hashable signature for a subtask + assignment context.

    The signature is invariant to:
    - Exact absolute positions (only a coarse distance bucket is used).
    - Exact agent/subtask IDs (only agent *types* are counted).

    It is sensitive to what determines whether a plan is a good fit:
    - The scenario name.
    - The sorted set of required skills.
    - The agent-type composition of the assigned coalition.
    - A coarse-bucketed distance from the lead agent to the target.
    """
    key_parts = {
        "scenario": scenario,
        "skills": tuple(sorted(required_skills)),
        "agent_composition": dict(sorted(Counter(agent_types).items())),
        "distance_bucket": _distance_bucket(lead_agent_distance),
    }
    raw = json.dumps(key_parts, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Store entry
# ---------------------------------------------------------------------------

@dataclass
class ExperienceEntry:
    """One recorded experience: signature + plan + outcome."""
    signature: str
    plan: dict[str, list[str]]  # subtask_id -> [agent_ids]
    success: bool
    scenario: str
    skills: list[str]
    agent_types: list[str]
    distance_bucket: int
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "plan": self.plan,
            "success": self.success,
            "scenario": self.scenario,
            "skills": self.skills,
            "agent_types": self.agent_types,
            "distance_bucket": self.distance_bucket,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperienceEntry:
        return cls(
            signature=d["signature"],
            plan=d["plan"],
            success=d["success"],
            scenario=d.get("scenario", ""),
            skills=d.get("skills", []),
            agent_types=d.get("agent_types", []),
            distance_bucket=d.get("distance_bucket", 0),
            timestamp=d.get("timestamp", ""),
        )


# ---------------------------------------------------------------------------
# Main store class
# ---------------------------------------------------------------------------

@dataclass
class SubtaskExperienceStore:
    """Persistent store of subtask plans keyed by canonical signature.

    Parameters
    ----------
    store_path : str
        Path to the JSON file (relative to project root or absolute).
    enabled : bool
        Master switch — when ``False``, all methods are no-ops.
    """

    store_path: str = "experience_store.json"
    enabled: bool = False

    # Internal state
    _entries: dict[str, list[ExperienceEntry]] = field(
        default_factory=dict, init=False, repr=False
    )

    # Metrics counters (reset per-run)
    reuse_attempts: int = field(default=0, init=False)
    reuse_hits: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.enabled:
            self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _resolved_path(self) -> Path:
        p = Path(self.store_path)
        if p.is_absolute():
            return p
        return project_root() / p

    def load(self) -> None:
        """Load the store from disk.  Missing file is a clean start."""
        path = self._resolved_path()
        if not path.exists():
            self._entries = {}
            return
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            self._entries = {}
            for sig, entries in raw.items():
                self._entries[sig] = [ExperienceEntry.from_dict(e) for e in entries]
            print(f"[EXPERIENCE] Loaded {sum(len(v) for v in self._entries.values())} entries from {path}")
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"[EXPERIENCE] Failed to load store ({exc}), starting fresh")
            self._entries = {}

    def save(self) -> None:
        """Persist the store to disk."""
        if not self.enabled:
            return
        path = self._resolved_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            sig: [e.to_dict() for e in entries]
            for sig, entries in self._entries.items()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def lookup(self, signature: str) -> dict[str, list[str]] | None:
        """Return the most recent *successful* plan for *signature*, or None."""
        if not self.enabled:
            return None
        entries = self._entries.get(signature, [])
        # Walk backward to find the most recent success
        for entry in reversed(entries):
            if entry.success:
                return dict(entry.plan)  # return a copy
        return None

    def record(
        self,
        signature: str,
        plan: dict[str, list[str]],
        success: bool,
        scenario: str = "",
        skills: list[str] | None = None,
        agent_types: list[str] | None = None,
        distance_bucket: int = 0,
    ) -> None:
        """Record a subtask outcome.  Saves incrementally."""
        if not self.enabled:
            return
        import datetime

        entry = ExperienceEntry(
            signature=signature,
            plan=plan,
            success=success,
            scenario=scenario,
            skills=skills or [],
            agent_types=agent_types or [],
            distance_bucket=distance_bucket,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        self._entries.setdefault(signature, []).append(entry)
        self.save()  # incremental persistence

    def reset_run_metrics(self) -> None:
        """Reset per-run counters at the start of each mission."""
        self.reuse_attempts = 0
        self.reuse_hits = 0
