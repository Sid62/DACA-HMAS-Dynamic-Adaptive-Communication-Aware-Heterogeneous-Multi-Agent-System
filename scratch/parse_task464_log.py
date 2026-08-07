#!/usr/bin/env python3
"""Parse Task-464 Log file to reconstruct all 180 run metrics and save JSON artifact."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ARTIFACT_DIR = Path(r"C:\Users\siddh\.gemini\antigravity-ide\brain\c23d3900-40ff-4e7d-9d6e-757044178297")


def parse_log():
    log_file = Path(r"C:\Users\siddh\.gemini\antigravity-ide\brain\c23d3900-40ff-4e7d-9d6e-757044178297\.system_generated\tasks\task-464.log")
    if not log_file.exists():
        print("Log file not found!")
        return []

    pattern = re.compile(
        r"\[\s*\d+/180\]\s+(?P<ver>\S+)\s+\|\s+(?P<sc>\S+)\s+\|\s+(?P<prof>\S+)\s+\|\s+s(?P<seed>\d+)\s+\|\s+Success:\s+(?P<succ>[\d\.]+)%\s+\|\s+Steps:\s+(?P<steps>\d+)\s+\|\s+Violations:\s+(?P<viol>\d+)\s+\|\s+Tokens:\s+(?P<tok>\d+)\s+\|\s+Calls:\s+(?P<calls>\d+)\s+\|\s+Comp:\s+(?P<comp>[\d\.]+)s"
    )

    records = []
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                records.append({
                    "version": m.group("ver"),
                    "scenario": m.group("sc"),
                    "profile": m.group("prof"),
                    "seed": int(m.group("seed")),
                    "success_rate": float(m.group("succ")) / 100.0,
                    "steps": int(m.group("steps")),
                    "invariant_violations": int(m.group("viol")),
                    "tokens": int(m.group("tok")),
                    "api_calls": int(m.group("calls")),
                    "computation_s": float(m.group("comp")),
                })

    print(f"Parsed {len(records)} records from task-464 log file!")

    out_dir = ROOT / "experiments" / "results" / "full_independent_verification"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "all_3way_verification_runs.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Saved reconstructed records to: {out_file}")
    return records



if __name__ == "__main__":
    parse_log()
