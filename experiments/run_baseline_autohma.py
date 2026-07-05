#!/usr/bin/env python3
"""Run AutoHMA-LLM baseline experiments (B1/B2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import CONFIGS, DACAOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoHMA-LLM baseline runner")
    parser.add_argument("--scenario", default="logistics",
                        choices=["logistics", "inspection", "search_rescue"])
    parser.add_argument("--architecture", default="centralized",
                        choices=["centralized", "decentralized"])
    parser.add_argument("--profile", default="stable",
                        choices=["stable", "gradual", "sudden", "oscillatory"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config_key = "B1" if args.architecture == "centralized" else "B2"
    config = CONFIGS[config_key]

    orch = DACAOrchestrator(
        scenario=args.scenario,
        network_profile=args.profile,
        seed=args.seed,
        config=config,
        max_steps=args.max_steps,
    )
    metrics = orch.run()
    result = metrics.to_dict()
    print(json.dumps(result, indent=2))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
