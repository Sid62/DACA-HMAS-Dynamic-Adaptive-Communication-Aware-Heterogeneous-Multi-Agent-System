# DACA-HMAS — GPU Setup and Run Instructions

Complete guide to running the DACA-HMAS framework on a machine with GPU, using a Cloud API for global planning and a local Device LLM (Ollama or vLLM) on GPU.

---

## 1. Hardware and Software Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 16 GB VRAM (e.g. RTX 4080, A10) | 24 GB VRAM (RTX 4090, A100 40GB) |
| RAM | 16 GB | 32 GB |
| Python | 3.11+ | 3.11 or 3.12 |
| CUDA | 11.8+ or 12.x | Match your GPU driver |
| OS | Windows 10/11, Linux | Ubuntu 22.04+ |

**Cloud LLM:** OpenAI (GPT-4o) or Anthropic (Claude) API key with billing enabled.

**Device LLM:** Llama-3.1-8B-Instruct (or Phi-3.5-mini) served locally via **Ollama** or **vLLM**.

---

## 2. Project Setup

### 2.1 Clone / navigate to project

```powershell
cd c:\Users\siddh\Downloads\MTP\daca-hmas
```

### 2.2 Create virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2.3 Install dependencies

```powershell
pip install -e ".[dev]"
```

### 2.4 Configure API keys

Copy the example env file and add your Cloud LLM key:

```powershell
copy .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=sk-your-actual-key
```

Or for Anthropic, edit `configs/llm.yaml` and set `cloud.provider: anthropic` and `api_key_env: ANTHROPIC_API_KEY`.

---

## 3. GPU Setup — Option A: Ollama (Recommended for simplicity)

Ollama automatically uses your NVIDIA GPU when CUDA is available.

### 3.1 Install Ollama

Download from [https://ollama.com](https://ollama.com) and install.

Verify GPU is detected:
```powershell
ollama ps
nvidia-smi
```

### 3.2 Pull the Device LLM model

```powershell
ollama pull llama3.1:8b
```

Alternative smaller model (fits 8 GB VRAM):
```powershell
ollama pull phi3.5:3.8b
```

### 3.3 Start Ollama server

Ollama runs as a background service after install. Verify:
```powershell
curl http://localhost:11434/api/tags
```

### 3.4 Configure DACA-HMAS for Ollama

Edit `configs/llm.yaml`:

```yaml
device:
  provider: ollama
  model: llama3.1:8b          # or phi3.5:3.8b
  base_url: http://localhost:11434
  max_tokens: 512
  temperature: 0.1
  num_gpu: -1                 # use all GPU layers
  memory_mb: 8192

use_mock: false               # IMPORTANT: enable real LLMs
```

---

## 4. GPU Setup — Option B: vLLM (Higher throughput)

vLLM is preferred for large experiment sweeps (450+ runs).

### 4.1 Install vLLM (requires CUDA)

```powershell
pip install vllm
```

### 4.2 Start vLLM OpenAI-compatible server

```powershell
python -m vllm.entrypoints.openai.api_server `
  --model meta-llama/Llama-3.1-8B-Instruct `
  --dtype auto `
  --gpu-memory-utilization 0.90 `
  --port 8000
```

**Linux equivalent:**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --port 8000
```

First run downloads the model (~16 GB). Confirm server is up:
```powershell
curl http://localhost:8000/v1/models
```

### 4.3 Configure DACA-HMAS for vLLM

Edit `configs/llm.yaml`:

```yaml
device:
  provider: vllm
  model: meta-llama/Llama-3.1-8B-Instruct
  base_url: http://localhost:8000/v1
  max_tokens: 512
  temperature: 0.1
  memory_mb: 16384

use_mock: false
```

---

## 5. Verify Installation (No Full Experiment Yet)

### 5.1 Run unit tests (CPU only, no GPU needed)

```powershell
pytest tests/ -v
```

Expected: all tests in `test_cqm`, `test_acds`, `test_feasibility`, `test_handoff`, `test_decomposition` pass.

### 5.2 Smoke test with mock LLMs (no API/GPU cost)

Leave `use_mock: true` in `configs/llm.yaml`, then:

```powershell
python experiments/run_baseline_autohma.py --scenario logistics --architecture centralized --seed 0
```

You should see JSON output with `success_rate`, `steps`, `tokens`, etc.

### 5.3 Smoke test with real LLMs (API + GPU)

Set `use_mock: false`, ensure Ollama/vLLM is running, then:

```powershell
python experiments/run_daca_hmas.py --config A5 --scenario inspection --profile gradual --seed 0 --max-steps 50
```

Monitor GPU usage in another terminal:
```powershell
nvidia-smi -l 2
```

---

## 6. Running Experiments

All commands assume you are in `daca-hmas/` with venv activated and `use_mock: false`.

### 6.1 Phase 0 — Baseline AutoHMA-LLM (B1 / B2)

Reproduce Table I–III architecture ordering:

```powershell
# B1: Static Centralized Hybrid — expect best on logistics + stable comm
python experiments/run_baseline_autohma.py --scenario logistics --architecture centralized --profile stable --seed 0
python experiments/run_baseline_autohma.py --scenario logistics --architecture centralized --profile stable --seed 1
python experiments/run_baseline_autohma.py --scenario logistics --architecture centralized --profile stable --seed 2

# B2: Static Decentralized Hybrid — expect best on inspection + degraded comm
python experiments/run_baseline_autohma.py --scenario inspection --architecture decentralized --profile gradual --seed 0
python experiments/run_baseline_autohma.py --scenario inspection --architecture decentralized --profile gradual --seed 1
```

**Gate check:** Centralized success on logistics ≥ Decentralized success on inspection (qualitative ordering).

### 6.2 Ablation runs (A1–A5)

| Config | Command flag | What it tests |
|--------|-------------|---------------|
| A1 | `--config A1` | Distance-feasible decomposition only |
| A2 | `--config A2` | Distance + CQI coalition formation only |
| A3 | `--config A3` | CQM + architecture switching only |
| A4 | `--config A4` | Switching without hysteresis (high SC expected) |
| A5 | `--config A5` | Full DACA-HMAS |

Example:
```powershell
python experiments/run_daca_hmas.py --config A5 --scenario inspection --profile oscillatory --seed 0 --seeds 10
```

Results saved to `experiments/results/`.

### 6.3 Full experimental sweep (Contribution 6)

Reduced quick sweep (dev validation):
```powershell
python experiments/run_full_sweep.py --quick --seeds 3 --max-steps 150
```

Full paper sweep (450+ runs, budget ~$150–400 API + GPU time):
```powershell
python experiments/run_full_sweep.py --seeds 10 --max-steps 200
```

Outputs:
- `experiments/results/full_sweep/all_results.json` — per-run metrics
- `experiments/results/full_sweep/aggregate.json` — mean/std per config
- `experiments/results/full_sweep/significance.json` — t-test results

---

## 7. Configuration Reference

### 7.1 Thresholds (`configs/thresholds.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `C1` | 50.0 | Communication range (m) — Eq 6 |
| `C_task` | 30.0 | Subtask collaboration range (m) |
| `gamma_min` | 0.3 | Minimum coalition feasibility — Eq 24 |
| `acds.cqi_crossover` | 0.65 | CQI threshold calibration point |
| `acds.delta` | 0.08 | Hysteresis band half-width |
| `acds.persistence_window` | 5 | N consecutive intervals before switch |

### 7.2 Scenarios and network profiles

| Scenario | Communication | Best static architecture |
|----------|--------------|--------------------------|
| `logistics` | Stable (0% loss) | Centralized (B1) |
| `inspection` | 10% delay, 1% loss | Decentralized (B2) |
| `search_rescue` | Moderate degradation | DACA-HMAS (A5) |

| Profile | Behavior |
|---------|----------|
| `stable` | No degradation |
| `gradual` | Linearly increasing loss over mission |
| `sudden` | Step drop at 40% mission time |
| `oscillatory` | Sinusoidal CQI fluctuation (tests hysteresis) |

---

## 8. Metrics Reported

| Metric | Module | Description |
|--------|--------|-------------|
| Success (%) | `metrics/evaluation.py` | Fraction of subtasks completed |
| Steps | orchestrator | Simulation timesteps |
| Tokens | LLM clients | Cloud + Device token usage |
| API Calls | LLM clients | Total LLM invocations |
| Memory (MB) | Device LLM | GPU memory for Device LLM |
| Computation (s) | orchestrator | Wall-clock runtime |
| **TFR** | decomposition | Task Feasibility Rate (C1) |
| **CFR** | coalition | Coalition Feasibility Rate (C3) |
| **SC** | acds | Switch Count (C4) |

---

## 9. Troubleshooting

### Ollama not using GPU
```powershell
ollama run llama3.1:8b "hello"
nvidia-smi   # should show ollama process using VRAM
```
If CPU-only: reinstall Ollama with CUDA support or update NVIDIA drivers.

### vLLM CUDA out of memory
Reduce GPU utilization:
```powershell
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --gpu-memory-utilization 0.75 --port 8000
```
Or use quantized model:
```powershell
python -m vllm.entrypoints.openai.api_server --model TheBloke/Llama-3.1-8B-Instruct-GPTQ --quantization gptq --port 8000
```

### OpenAI API errors
- Verify `OPENAI_API_KEY` in `.env` or environment
- Check billing/quota at platform.openai.com
- Temporarily set `use_mock: true` to isolate simulation bugs from API issues

### High API cost during sweep
- Enable `cache_responses: true` in `configs/llm.yaml` (default on)
- Run Device LLM locally (never via API)
- Use `--quick` flag first to validate pipeline before full 450-run sweep

### Import errors
Always run from project root with venv active:
```powershell
cd c:\Users\siddh\Downloads\MTP\daca-hmas
$env:PYTHONPATH = "."
python experiments/run_daca_hmas.py --config A5 --scenario logistics --seed 0
```

---

## 10. Recommended Run Order

```
1. pip install -e ".[dev]"
2. pytest tests/ -v                          # validate math modules
3. Start Ollama or vLLM on GPU
4. Set use_mock: false in configs/llm.yaml
5. Set OPENAI_API_KEY in .env
6. run_baseline_autohma.py (B1 logistics, B2 inspection) — Phase 0 gate
7. run_daca_hmas.py --config A1..A5          # ablations
8. run_full_sweep.py --seeds 10                # Contribution 6
```

---

## 11. Project Structure Quick Reference

```
daca-hmas/
├── configs/
│   ├── thresholds.yaml     # C1, gamma_min, ACDS thresholds
│   └── llm.yaml            # Cloud + Device LLM settings (set use_mock: false)
├── src/
│   ├── env/                # Simulation: agents, scenarios, network
│   ├── cqm/                # Communication Quality Monitor (Eqs 17-19)
│   ├── acds/               # Architecture switching (Eqs 20-22)
│   ├── decomposition/      # Distance-feasible decomposition (Gap 1)
│   ├── coalition/          # Feasibility + formation (Eqs 23-27)
│   ├── handoff/            # State snapshot + CA transfer (Eqs 28-30)
│   ├── reallocation/       # Post-switch reallocation (Eq 31)
│   ├── llm/                # Cloud + Device LLM clients + prompts/
│   ├── control/            # PID, NMPC, Q-learning
│   ├── coordination/       # Orchestrator + hybrid coordinators
│   └── metrics/            # Evaluation metrics
├── experiments/
│   ├── run_baseline_autohma.py
│   ├── run_daca_hmas.py
│   └── run_full_sweep.py
└── tests/                  # Unit tests for core math modules
```

---

## 12. Citation

Extends AutoHMA-LLM (Yang et al., IEEE TCCN 2025). See `DACA-HMAS_Mathematical_Formulation.md` and `DACA-HMAS_Implementation_Plan.md` in the parent Downloads folder for full methodology.
