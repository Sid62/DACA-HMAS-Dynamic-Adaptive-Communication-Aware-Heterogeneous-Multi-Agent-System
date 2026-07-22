import sys, json
from pathlib import Path

ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT))

from experiments.run_daca_hmas import run_experiment

ARTIFACT_DIR = Path(r"C:\Users\siddh\.gemini\antigravity-ide\brain\b527b2da-3323-4b8f-afae-838fe0629f21")

results_all = {}

scenarios = ['search_rescue', 'inspection', 'logistics']

print("=== RUNNING SEEDS 0-4 FOR SEARCH_RESCUE, INSPECTION, LOGISTICS (OSCILLATORY) ===")
for scenario in scenarios:
    for seed in range(5):
        key = f"{scenario}_oscillatory_seed_{seed}"
        print(f"Running {key}...")
        data = run_experiment(
            config_name='A5',
            scenario_name=scenario,
            profile='oscillatory',
            seed=seed,
            max_steps=200,
        )
        results_all[key] = data

print("\n=== RUNNING STABLE, GRADUAL, SUDDEN PROFILES FOR SEARCH_RESCUE (SEED 0) ===")
for profile in ['stable', 'gradual', 'sudden']:
    key = f"search_rescue_{profile}_seed_0"
    print(f"Running {key}...")
    data = run_experiment(
        config_name='A5',
        scenario_name='search_rescue',
        profile=profile,
        seed=0,
        max_steps=200,
    )
    results_all[key] = data

json_out = ARTIFACT_DIR / "all_execution_results.json"
with open(json_out, 'w') as f:
    json.dump(results_all, f, indent=2)

print(f"\nALL EXPERIMENTS COMPLETED! Summary saved to {json_out}")
