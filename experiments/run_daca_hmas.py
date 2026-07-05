#!/usr/bin/env python3
"""Run DACA-HMAS ablation and full-system experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import CONFIGS, DACAOrchestrator
from src.metrics.evaluation import MetricsCollector


def main() -> None:
    parser = argparse.ArgumentParser(description="DACA-HMAS experiment runner")
    parser.add_argument("--config", default="A5",
                        choices=list(CONFIGS.keys()))
    parser.add_argument("--scenario", default="inspection",
                        choices=["logistics", "inspection", "search_rescue"])
    parser.add_argument("--profile", default="gradual",
                        choices=["stable", "gradual", "sudden", "oscillatory"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of seeds to run (starting from --seed)")
    parser.add_argument("--output-dir", default="experiments/results")
    args = parser.parse_args()

    config = CONFIGS[args.config]
    collector = MetricsCollector()
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.seeds):
        seed = args.seed + i
        orch = DACAOrchestrator(
            scenario=args.scenario,
            network_profile=args.profile,
            seed=seed,
            config=config,
            max_steps=args.max_steps,
        )
        metrics = orch.run()
        collector.records.append(metrics)
        result_path = out_dir / f"{args.config}_{args.scenario}_{args.profile}_s{seed}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        print(f"Seed {seed}: success={metrics.success_rate:.2%}, SC={metrics.switch_count}")

    summary_path = out_dir / f"summary_{args.config}_{args.scenario}_{args.profile}.json"
    agg = MetricsCollector.aggregate_by_config(collector.records)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
