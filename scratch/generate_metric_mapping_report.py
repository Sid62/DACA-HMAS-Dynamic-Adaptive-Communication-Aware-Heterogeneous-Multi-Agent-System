#!/usr/bin/env python3
"""Generate metric mapping analysis report artifact comparing DACA-HMAS metrics to AutoHMA-LLM paper."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ARTIFACT_DIR = Path(r"C:\Users\siddh\.gemini\antigravity-ide\brain\c23d3900-40ff-4e7d-9d6e-757044178297")


def format_metric_mapping_report() -> str:
    md = []
    md.append("# Code Inspection & Metric Mapping Analysis: DACA-HMAS vs. AutoHMA-LLM Paper\n")
    md.append("**Role**: Senior AI Researcher, IEEE Transactions Reviewer, Multi-Agent Systems Architect")
    md.append("**Objective**: Deep code inspection tracing every metric from calculation to JSON export, comparing definitions with AutoHMA-LLM base paper.\n")
    md.append("---\n")

    md.append("## 1. Mapping Summary Table\n\n")
    md.append("| Base Paper Metric | AutoHMA-LLM Paper Definition | DACA-HMAS Metric Name | Same? | What DACA-HMAS Actually Measures | Recommended Paper Metric |")
    md.append("| :--- | :--- | :--- | :---: | :--- | :--- |")
    md.append("| **Success** | Task Completion Accuracy (% of subtasks completed within max steps) | `success_rate` | **YES** | `len(completed_subtasks) / total_subtasks * 100` | `success_rate` |")
    md.append("| **Communication Steps** | Number of coordination / message rounds required for consensus | `steps` | **NO** | Discrete simulation timesteps (agent movement ticks) | `consensus_rounds` (+ `cloud_planning_calls`) |")
    md.append("| **Tokens / Data Size** | LLM prompt + completion tokens exchanged | `tokens` (`total_tokens`) | **PARTIALLY** | Combined Cloud LLM prompt/completion tokens + Device LLM prompt/completion tokens | `total_tokens` (or report Cloud vs. Device tokens separately) |")
    md.append("| **API Calls** | Invocations of LLM planners for algorithm execution | `api_calls` (`total_api_calls`) | **YES** | Sum of Cloud LLM decomposition calls + Device LLM local planning calls | `total_api_calls` (or `cloud_planning_calls` + `device_planning_calls`) |")
    md.append("| **Memory** | Memory usage per Device LLM (MB) | `memory_mb` | **NO** | Hardcoded config parameter (`8192.0` MB / 8 GB threshold) | Omit static `memory_mb` or measure dynamic `psutil` RAM |")
    md.append("| **Computation** | System-wide Computational Overhead | `computation_s` | **YES** | Total wall-clock runtime (seconds) from `run()` start to finish | `computation_s` (plus `avg_planning_latency`) |")

    md.append("\n---\n")
    md.append("## 2. Detailed Line-by-Line Code Tracing & Analysis\n\n")

    md.append("### 1. Task Completion Accuracy (Success)\n")
    md.append("- **A. Computed Location**: `src/env/daca_env.py` (lines 118–122) & `src/coordination/orchestrator.py` (line 436).\n")
    md.append("- **B. Code**: \n")
    md.append("  ```python\n")
    md.append("  def success_rate(self) -> float:\n")
    md.append("      total = len(self._subtasks)\n")
    md.append("      if total == 0: return 0.0\n")
    md.append("      return len(self.state.completed_subtasks) / total\n")
    md.append("  ```\n")
    md.append("- **C. Representation**: Exact ratio of subtasks reached by assigned agents (`dist < 8.0`) before hitting `max_steps`.\n")
    md.append("- **D. Equivalent to Paper?**: **YES**. Matches 1:1.\n\n")

    md.append("### 2. Communication Steps (Steps)\n")
    md.append("- **A. Computed Location**: `src/env/daca_env.py` (lines 104–106) & `src/coordination/orchestrator.py` (line 437).\n")
    md.append("- **B. Code**: \n")
    md.append("  ```python\n")
    md.append("  def advance(self) -> dict[str, Any]:\n")
    md.append("      self.state.timestep += 1\n")
    md.append("      if self.check_mission_complete() or self.state.timestep >= self.max_steps:\n")
    md.append("          self.state.mission_complete = True\n")
    md.append("      return self.get_observation()\n")
    md.append("  ```\n")
    md.append("- **C. Representation**: **Simulation Timesteps** (movement physics ticks in Gym environment). It does NOT measure communication rounds.\n")
    md.append("- **D. Equivalent to Paper?**: **NO**. The AutoHMA-LLM paper uses `Steps` to denote **Communication Steps** (coordination rounds), whereas DACA-HMAS `steps` represents physical simulation duration.\n")
    md.append("- **E. Recommended Replacement**: Use `consensus_rounds` (from `PeerCommunicationManager`) plus `cloud_planning_calls` as the true Communication Steps metric!\n\n")

    md.append("### 3. Tokens / Data Size\n")
    md.append("- **A. Computed Location**: `src/llm/cloud_llm_client.py` (lines 158, 280), `src/llm/device_llm_client.py` (lines 127, 166), & `src/metrics/evaluation.py` (lines 117–119).\n")
    md.append("- **B. Code**: \n")
    md.append("  ```python\n")
    md.append("  total_tokens = cloud_tokens + device_tokens\n")
    md.append("  ```\n")
    md.append("- **C. Representation**: Sum of prompt tokens and completion tokens across Cloud LLM calls and local Device LLM calls. (In mock mode, estimated via word count `len(prompt.split()) + len(response.split())`).\n")
    md.append("- **D. Equivalent to Paper?**: **PARTIALLY**. AutoHMA-LLM measures centralized planner exchange tokens. DACA-HMAS includes both Central Cloud LLM tokens AND distributed Edge/Device LLM tokens.\n")
    md.append("- **E. Recommendation**: Report both `cloud_tokens` and `total_tokens` to show how DACA-HMAS reduces central cloud token traffic!\n\n")

    md.append("### 4. API Calls\n")
    md.append("- **A. Computed Location**: `src/llm/cloud_llm_client.py` (line 147), `src/llm/device_llm_client.py` (line 128), & `src/metrics/evaluation.py` (lines 120–122).\n")
    md.append("- **B. Code**: \n")
    md.append("  ```python\n")
    md.append("  total_api_calls = cloud_api_calls + device_api_calls\n")
    md.append("  ```\n")
    md.append("- **C. Representation**: Total count of LLM reasoning invocations (Cloud LLM task decomposition + Device LLM local plan generation).\n")
    md.append("- **D. Equivalent to Paper?**: **YES**. Counts total API calls needed to execute the coordination algorithm.\n\n")

    md.append("### 5. Memory\n")
    md.append("- **A. Computed Location**: `src/llm/device_llm_client.py` (line 124) & `src/metrics/evaluation.py` (line 123).\n")
    md.append("- **B. Code**: \n")
    md.append("  ```python\n")
    md.append("  self.usage.memory_mb = self.config.get(\"device\", {}).get(\"memory_mb\", 8192.0)\n")
    md.append("  ```\n")
    md.append("- **C. Representation**: **Static Configuration Threshold** (`8192.0` MB = 8 GB), NOT dynamic runtime RAM/GPU memory usage measured during execution.\n")
    md.append("- **D. Equivalent to Paper?**: **NO**. The paper refers to dynamic device memory footprint.\n")
    md.append("- **E. Recommendation**: Either clarify in the paper that memory represents configured device memory allocation (8 GB), or omit `memory_mb` from the comparative table unless profiled via `psutil`.\n\n")

    md.append("### 6. Computation\n")
    md.append("- **A. Computed Location**: `src/coordination/orchestrator.py` (line 431) & `src/metrics/evaluation.py` (line 124).\n")
    md.append("- **B. Code**: \n")
    md.append("  ```python\n")
    md.append("  elapsed = time.perf_counter() - start\n")
    md.append("  ```\n")
    md.append("- **C. Representation**: Total wall-clock execution time (in seconds) of the entire simulation run.\n")
    md.append("- **D. Equivalent to Paper?**: **YES**. Accurately measures system-wide computational overhead.\n\n")

    md.append("---\n")
    md.append("## 3. Recommended Paper Reporting Strategy\n\n")

    md.append("When comparing DACA-HMAS against AutoHMA-LLM in your paper:\n\n")
    md.append("1. **Success**: Use `success_rate` (Direct 1:1 match).\n")
    md.append("2. **Communication Steps**: Clarify that DACA-HMAS `steps` represents physical simulation timesteps, while `consensus_rounds` (+ `cloud_planning_calls`) represents **Communication Steps**.\n")
    md.append("3. **Tokens**: Report `total_tokens` alongside `cloud_tokens` to highlight DACA-HMAS's reduction in central network load.\n")
    md.append("4. **API Calls**: Report `total_api_calls` (Cloud + Device calls).\n")
    md.append("5. **Computation**: Report `computation_s` and `avg_planning_latency`.\n")
    md.append("6. **Memory**: Specify that `memory_mb` is configured 8 GB device allocation.\n")

    return "\n".join(md)


def main():
    report_md = format_metric_mapping_report()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report_file = ARTIFACT_DIR / "metric_mapping_analysis.md"
    report_file.write_text(report_md, encoding="utf-8")
    print(f"Metric Mapping Analysis Report written to: {report_file}")


if __name__ == "__main__":
    main()
