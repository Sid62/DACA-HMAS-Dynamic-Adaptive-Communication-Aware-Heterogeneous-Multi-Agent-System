"""Research-mode failure types and structured failure reporting.

Used exclusively to preserve experimental integrity: when the Cloud LLM
cannot produce a genuine response after all configured retries, the
experiment must stop and be flagged INVALID rather than silently
continuing under a different (unlabeled) planning algorithm.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.config import project_root


@dataclass
class FailureReport:
    experiment_status: str = "FAILED"
    failure_reason: str = ""
    provider: str = ""
    model: str = ""
    scenario: str | None = None
    architecture: str | None = None
    network_profile: str | None = None
    seed: int | None = None
    simulation_step: int | None = None
    retry_count: int = 0
    exception_type: str = ""
    timestamp: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def log(self) -> None:
        """Concise research-log block — deliberately not a stack trace."""
        print(
            "[Experiment]\n"
            f"Status: {self.experiment_status}\n"
            f"Reason: {self.failure_reason}\n"
            f"Provider: {self.provider}\n"
            f"Step: {self.simulation_step}\n"
            f"Seed: {self.seed}\n"
            f"Architecture: {self.architecture}\n"
            f"Scenario: {self.scenario}\n"
            "This run is SCIENTIFICALLY INVALID and must be excluded from analysis."
        )

    def persist(self) -> Path:
        """Write to results/failed_runs/ so invalid runs are discoverable
        and excludable at analysis time without re-running experiments."""
        out_dir = project_root() / "results" / "failed_runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = (
            f"FAILED_{self.scenario}_{self.architecture}_{self.network_profile}"
            f"_seed{self.seed}_step{self.simulation_step}_{int(time.time())}.json"
        )
        path = out_dir / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


class ExperimentFailed(Exception):
    """Raised when the Cloud LLM cannot produce a genuine response after all
    configured retries. This terminates only the current experiment run —
    callers (CLI entry points, sweep scripts) catch this and move on to the
    next seed/config rather than the process crashing or silently continuing
    under a substitute planner.
    """

    def __init__(self, report: FailureReport):
        self.report = report
        super().__init__(
            f"Experiment terminated: {report.failure_reason} "
            f"(provider={report.provider}, retries={report.retry_count}, "
            f"step={report.simulation_step}, seed={report.seed})"
        )