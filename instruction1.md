# DACA-HMAS — Detailed Run Guide (Domain-Level Device LLM)

Step-by-step instructions for installing, configuring, and running DACA-HMAS with the **domain-level multi-Device-LLM** architecture. For GPU hardware requirements and vLLM deep-dive, see also [INSTRUCTION.md](INSTRUCTION.md).

---

## 1. What Changed (Domain-Level Architecture)

In decentralized mode (B2, A5), the framework now runs **one Device LLM per agent type**, not one per robot:

| Mode | Device LLM layout | Peer network |
|------|-------------------|--------------|
| **Centralized (B1)** | Single `dispatch` client — unchanged | No peer comm |
| **Decentralized (B2/A5)** | One client per domain (`uav`, `vehicle`, `robot`) | Domain ↔ domain messaging |

Agent types are **auto-discovered** from the fleet (`AgentState.agent_type`). No hardcoded UAV/Vehicle/Robot lists in coordination code.

```
Fleet:  uav_0, uav_1, vehicle_0, robot_0, robot_1
           │         │              │
           ▼         ▼              ▼
Domains: [  uav  ] [ vehicle ] [ robot ]  ← 3 Device LLM instances
           └──────── peer_manager ────────┘
```

**Coalition coordination:** the domain with the most coalition members becomes the leader (`dominant_domain_for_coalition`). Peer review and plan merge happen between domains.

---

## 2. Prerequisites

| Item | Notes |
|------|-------|
| Python | 3.11+ |
| OS | Windows 10/11 or Linux |
| GPU (optional) | Required only when `use_mock: false` for Device LLM |
| Cloud API key | OpenAI or Anthropic — required when `use_mock: false` |
| Local LLM server | Ollama or vLLM — one instance serves all domain clients |

---

## 3. Installation (Step by Step)

### Step 1 — Open project directory

**Windows (PowerShell):**
```powershell
cd c:\Users\siddh\Downloads\MTP\daca-hmas
```

**Linux:**
```bash
cd ~/MTP/daca-hmas
```

### Step 2 — Create and activate virtual environment

**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install package and dev tools

```powershell
pip install -e ".[dev]"
```

### Step 4 — Configure API keys (for real Cloud LLM)

```powershell
copy .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=sk-your-key-here
```

For Anthropic, set `cloud.provider: anthropic` and `api_key_env: ANTHROPIC_API_KEY` in `configs/llm.yaml`.

### Step 5 — Verify install

```powershell
pytest tests/ -v
```

All tests in `tests/` should pass. This uses mock LLMs by default — no GPU or API needed.

---

## 4. LLM Modes: Mock vs Real

Controlled by `use_mock` in `configs/llm.yaml`:

| Setting | Cloud LLM | Device LLM | Cost | Use case |
|---------|-----------|------------|------|----------|
| `use_mock: true` (default) | Deterministic JSON stubs | Deterministic JSON stubs | Free | Dev, CI, pytest |
| `use_mock: false` | Real API (GPT-4o / Claude) | Local GPU via Ollama/vLLM | API + GPU time | Paper experiments |

### Mock mode (recommended first run)

Leave `configs/llm.yaml` as shipped (`use_mock: true`). Run:

```powershell
python experiments/run_daca_hmas.py --config B2 --scenario inspection --profile gradual --seed 0 --max-steps 50
```

Expected console output (approximate):
```
Seed 0: success=XX.XX%, SC=0
Summary written to experiments/results/summary_B2_inspection_gradual.json
```

Check `experiments/results/B2_inspection_gradual_s0.json` for full metrics including distributed fields.

### Real LLM mode

1. Start Ollama or vLLM (Section 5).
2. Set `use_mock: false` in `configs/llm.yaml`.
3. Ensure `.env` has a valid API key.
4. Run the same command as above.

---

## 5. GPU / Local Device LLM Setup

One Ollama or vLLM server is shared by all domain Device LLM clients. You do **not** need separate GPU processes per agent type.

### Option A — Ollama (simplest)

```powershell
ollama pull llama3.1:8b
curl http://localhost:11434/api/tags
```

`configs/llm.yaml`:
```yaml
device:
  provider: ollama
  model: llama3.1:8b
  base_url: http://localhost:11434
  num_gpu: -1
use_mock: false
```

### Option B — vLLM (higher throughput for sweeps)

```powershell
pip install vllm
python -m vllm.entrypoints.openai.api_server `
  --model meta-llama/Llama-3.1-8B-Instruct `
  --gpu-memory-utilization 0.90 `
  --port 8000
```

`configs/llm.yaml`:
```yaml
device:
  provider: vllm
  model: meta-llama/Llama-3.1-8B-Instruct
  base_url: http://localhost:8000/v1
use_mock: false
```

Monitor GPU during a decentralized run:
```powershell
nvidia-smi -l 2
```

---

## 6. Running Experiments

All commands assume venv is active and cwd is `daca-hmas/`.

### 6.1 Baselines — B1 / B2

B1 uses **centralized** dispatch (single Device LLM). B2 uses **decentralized** domain-level planning.

```powershell
# B1: Static Centralized Hybrid
python experiments/run_baseline_autohma.py --scenario logistics --architecture centralized --profile stable --seed 0

# B2: Static Decentralized Hybrid (domain-level Device LLMs)
python experiments/run_baseline_autohma.py --scenario inspection --architecture decentralized --profile gradual --seed 0
```

Save output to file:
```powershell
python experiments/run_baseline_autohma.py --scenario logistics --architecture centralized --seed 0 --output experiments/results/B1_logistics_s0.json
```

### 6.2 Ablations — A1 through A5

| Config | Flags enabled | Architecture |
|--------|---------------|--------------|
| A1 | Distance decomposition | Static centralized |
| A2 | + Coalition feasibility | Static centralized |
| A3 | CQM + ACDS switching | Dynamic |
| A4 | Switching, no hysteresis | Dynamic |
| A5 | Full system (handoff + reallocation) | Dynamic |

```powershell
# Single run
python experiments/run_daca_hmas.py --config A5 --scenario search_rescue --profile oscillatory --seed 0

# Multiple seeds (writes per-seed JSON + summary)
python experiments/run_daca_hmas.py --config A5 --scenario inspection --profile gradual --seed 0 --seeds 10

# Shorter dev run
python experiments/run_daca_hmas.py --config A3 --scenario logistics --profile stable --seed 0 --max-steps 100
```

Results directory: `experiments/results/`

### 6.3 Full sweep

```powershell
# Quick validation (~minutes, mock or real)
python experiments/run_full_sweep.py --quick --seeds 3 --max-steps 150

# Full paper matrix (~hours, real LLMs recommended with cache)
python experiments/run_full_sweep.py --seeds 10 --max-steps 200
```

Outputs under `experiments/results/full_sweep/`:
- `all_results.json`
- `aggregate.json`
- `significance.json`

---

## 7. Scenarios and Network Profiles

| `--scenario` | Typical best static arch | Notes |
|--------------|--------------------------|-------|
| `logistics` | B1 (centralized) | Stable comm, coordination-heavy |
| `inspection` | B2 (decentralized) | Delay/loss tolerant |
| `search_rescue` | A5 (dynamic) | Mixed degradation |

| `--profile` | Behavior |
|-------------|----------|
| `stable` | No degradation |
| `gradual` | Loss increases over mission |
| `sudden` | Step drop at ~40% mission time |
| `oscillatory` | Sinusoidal CQI — tests ACDS hysteresis |

---

## 8. Interpreting Output Metrics

Each run produces JSON like:

```json
{
  "config": "B2",
  "scenario": "inspection",
  "profile": "gradual",
  "seed": 0,
  "success_rate": 85.0,
  "steps": 120,
  "tokens": 450,
  "api_calls": 12,
  "memory_mb": 12288.0,
  "computation_s": 3.421,
  "tfr": 0.9500,
  "cfr": 0.8800,
  "switch_count": 0,
  "peer_messages": 24,
  "broadcast_count": 6,
  "consensus_rounds": 6,
  "consensus_latency": 0.0842,
  "plan_merge_count": 6,
  "distributed_replanning_count": 6
}
```

### Core metrics

| Field | Meaning |
|-------|---------|
| `success_rate` | % subtasks completed (0–100 in JSON) |
| `steps` | Simulation timesteps |
| `tokens` / `api_calls` | Cloud + all Device LLM domains combined |
| `memory_mb` | Sum of reported Device LLM memory (3 domains × ~4096 mock) |
| `tfr` | Task Feasibility Rate (distance decomposition) |
| `cfr` | Coalition Feasibility Rate |
| `switch_count` | Architecture switches (ACDS); 0 for B1/B2 |

### Distributed / domain-level metrics (B2, A5 decentralized phases)

| Field | Meaning | Expected when |
|-------|---------|---------------|
| `peer_messages` | Unicast messages between domain peers | B2/A5 with coalitions |
| `broadcast_count` | Leader domain plan proposals | Multiple coalitions replanned |
| `consensus_rounds` | Plan review + merge cycles | Distributed planning active |
| `consensus_latency` | Cumulative simulated comm delay (seconds) | Non-zero with peer_manager |
| `plan_merge_count` | Successful leader merge operations | Peer domains reviewed plans |
| `distributed_replanning_count` | Distributed planning or realloc triggers | Decentralized mode steps |

**Sanity checks for domain architecture:**

1. **B1** — `peer_messages` should be **0** (no peer network in centralized mode).
2. **B2** — `peer_messages` > 0, `consensus_rounds` > 0 when coalitions form.
3. **A5** with switching — `switch_count` may be > 0; after handoff, snapshot restores domain `node_states` and `pending_messages`.
4. **Device LLM count** — mock memory ~4096 MB per domain; 3 domains ≈ 12288 MB in metrics (plus dispatch client for B1).

---

## 9. pytest Reference

```powershell
# All tests
pytest tests/ -v

# Specific modules
pytest tests/test_cqm.py -v
pytest tests/test_acds.py -v
pytest tests/test_handoff.py -v
pytest tests/test_integration.py -v

# Integration smoke (B1, B2, A5 × 2 scenarios)
pytest tests/test_integration.py -v
```

`test_integration.py` validates orchestrator end-to-end including domain-level decentralized path for B2/A5.

---

## 10. Configuration Files

| File | Purpose |
|------|---------|
| `configs/llm.yaml` | Cloud + Device LLM providers, `use_mock`, cache |
| `configs/thresholds.yaml` | C1, C_task, gamma_min, ACDS thresholds, scenarios |
| `.env` | API keys (not committed) |

Key thresholds (`configs/thresholds.yaml`):

| Key | Default | Role |
|-----|---------|------|
| `C1` | 50.0 | Communication range (m) |
| `C_task` | 30.0 | Subtask collaboration range |
| `gamma_min` | 0.3 | Minimum coalition feasibility |
| `acds.cqi_crossover` | 0.65 | Switch threshold |
| `acds.delta` | 0.08 | Hysteresis band |

---

## 11. Recommended First-Time Run Order

```
1. pip install -e ".[dev]"
2. pytest tests/ -v
3. run_daca_hmas.py --config B2 ...     (mock — verify domain metrics > 0)
4. run_baseline_autohma.py B1 + B2      (mock — Phase 0 gate)
5. Start Ollama/vLLM + set use_mock: false
6. run_daca_hmas.py --config A5 ...     (one real-LLM smoke test)
7. run_daca_hmas.py --seeds 10          (ablation seeds)
8. run_full_sweep.py --quick            (pipeline validation)
```

---

## 12. Troubleshooting

### `peer_messages` is 0 in B2

- Confirm `--architecture decentralized` or `--config B2` / `A5`.
- Check coalitions are forming (`cfr` in output).
- Ensure `device_llms` has multiple domain keys (orchestrator logs no error).

### Import / module errors

Run from project root; `pyproject.toml` sets `pythonpath = ["."]` for pytest. For manual runs:

```powershell
cd c:\Users\siddh\Downloads\MTP\daca-hmas
python experiments/run_daca_hmas.py --config A5 --scenario logistics --seed 0
```

### Ollama connection refused

```powershell
ollama serve
curl http://localhost:11434/api/tags
```

### High API cost

- Keep `cache_responses: true` (default).
- Use `--max-steps 50` for smoke tests.
- Run `run_full_sweep.py --quick` before full matrix.

### Handoff / switch issues (A5)

After ACDS switch, orchestrator captures domain `node_states` and `pending_messages`, restores via `restore_distributed_state()`. If missions stall after switch, run with `--max-steps 100` and inspect `switch_count` vs `success_rate`.

---

## 13. Architecture Code Map (Domain Path)

| Component | File | Role |
|-----------|------|------|
| Domain discovery | `src/communication/models.py` | `discover_agent_type_domains()`, `dominant_domain_for_coalition()` |
| Peer network | `src/communication/peer_manager.py` | Domain registration, domain×domain CQI |
| Device LLM | `src/llm/device_llm_client.py` | One client per agent type, `managed_agent_ids` |
| Decentralized planning | `src/coordination/decentralized_hybrid.py` | Domain leader → peer review → merge |
| Orchestrator | `src/coordination/orchestrator.py` | `_build_device_llms_by_type()`, `dispatch_llm` for B1 |
| Handoff | `src/handoff/snapshot.py` | Domain `node_states` in G(t*) |
| Reallocation | `src/reallocation/post_switch.py` | Domain leader-peer realloc consensus |

**Unchanged by design:** `src/coordination/centralized_hybrid.py`, `src/llm/cloud_llm_client.py`, experiment CLI scripts.

---

## 14. Related Docs

| Document | Content |
|----------|---------|
| [README.md](README.md) | Quick start |
| [INSTRUCTION.md](INSTRUCTION.md) | GPU hardware, vLLM details, troubleshooting |
| [README1.md](README1.md) | Full reverse-engineered system documentation |
