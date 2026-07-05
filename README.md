# DACA-HMAS

Dynamic Architecture and Coalition Adaptation for Heterogeneous Multi-Agent Systems.

Extends AutoHMA-LLM with runtime architecture switching, distance-aware coalition formation, state snapshot handoff, and post-switch reallocation.

## Setup

See **[INSTRUCTION.md](INSTRUCTION.md)** for complete GPU setup and run guide.

Quick start:

```bash
cd daca-hmas
pip install -e ".[dev]"
```

Set `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`) for Cloud LLM. Start Ollama with `llama3.1:8b` for Device LLM, or set `use_mock: true` in `configs/llm.yaml`.

## Run

```bash
# Baseline AutoHMA-LLM (B1/B2)
python experiments/run_baseline_autohma.py --scenario logistics --architecture centralized --seed 0

# Full DACA-HMAS ablations
python experiments/run_daca_hmas.py --config A5 --scenario inspection --profile gradual --seed 0

# Unit tests
pytest tests/ -v
```

## Project Structure

See `DACA-HMAS_Implementation_Plan.md` in parent Downloads folder for full methodology.
