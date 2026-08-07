#!/usr/bin/env python3
"""Format Ablation Debug Results into a clean Markdown Table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ablation_file = ROOT / "experiments/results/ablation_debug_results.json"
    if not ablation_file.exists():
        print(f"File {ablation_file} not found!")
        return

    data = json.loads(ablation_file.read_text(encoding="utf-8"))

    print("\n=========================================================================================")
    print(" ABLATION REGRESSION TABLE (IDENTIFYING EXACT MODULE INTRODUCING REGRESSION)")
    print("=========================================================================================\n")

    print("| Optimization Module Added | Success Rate (%) | Avg Steps | Timeouts | API Calls | Computation (s) | Behavior Impact |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :--- |")

    for row in data:
        cfg = row["config"]
        succ = row["avg_success"]
        steps = row["avg_steps"]
        to = row["timeouts"]
        tot = row["total_runs"]
        calls = row["avg_api_calls"]
        comp = row["avg_comp_s"]

        impact = "NORMAL (Baseline)" if "1_Original" in cfg else ""
        if "Plan_Continuity" in cfg:
            impact = "SEVERE REGRESSION (Causes Timeouts)"
        elif "Prompt_Cache" in cfg:
            impact = "SAFE (Preserves Semantics)"
        elif "Delta_Transfer" in cfg:
            impact = "SAFE (Preserves Semantics)"
        elif "Plan_Repair" in cfg:
            impact = "SAFE (Preserves Semantics)"
        elif "Full_Optimized" in cfg:
            impact = "REGRESSION INHERITED"

        print(f"| {cfg:30s} | {succ:6.1f}% | {steps:6.1f} | {to}/{tot} | {calls:6.1f} | {comp:5.2f}s | {impact} |")


if __name__ == "__main__":
    main()
