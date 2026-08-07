# Remaining_Implementation_Guide.md

**Audit basis:** `DACA-HMAS-...-main` @ `a1fd1ff` + uncommitted working-tree changes
**Method:** every claim below verified by reading the shipped source and/or executing it. Reports were not trusted.
**Scope:** only issues that survived direct verification appear here.

---

## Verification Summary

### ✓ VERIFIED — correctly implemented, do not touch

| Item | Evidence |
|---|---|
| **R1 deterministic seeding** | `src/env/scenarios.py:54,100` use `zlib.crc32(...)`. Zero residual `hash(name)` seeding. Executed: `ENV_HASH = b161291a3717f67c` identical at `PYTHONHASHSEED` 0/1/2/3. End-to-end mission metrics identical across 4 hash seeds. |
| **R8 honest API accounting** | `cloud_llm_client.py:369` increments `cloud_network_calls` **per attempt** inside `_call_with_retries`; `:302` for the mock path; `cloud_failed_attempts` per failed attempt; `:278` records `cloud_disk_cache_hits` **without** incrementing `cloud_api_calls`. All three serialized at `evaluation.py:157-159`. |
| **Optimization B (coalition retry)** | `coalition/formation.py` — `break  # Optimization B` present; the cloud re-query loop no longer iterates. |
| **Scoped tier-walk fix** | `distance_feasible_decomp.py` is **byte-identical** to the shipped spec (`diff` returns empty). `legacy_idx` gate at `:284`, `use_eta=(idx > legacy_idx)` at `:293`. |
| **Zero-perturbation property** | Executed: `CHANGED-EXISTING = 0` for logistics (30 subtasks, 7 rescued), inspection (40, 6), search_rescue (50, 14). |
| **Test suite** | 50/50 pass (with `experience_store.json` removed first). |

### ✗ NOT IMPLEMENTED — previously requested, verified absent

| Issue | Location | Proof |
|---|---|---|
| **I-1** `is_valid` ignores configured threshold | `plan_continuity.py:30-31` | `return self.total_validity_score >= 0.75` hardcoded, while `validity_threshold` is accepted at `:54,59` and passed from `orchestrator.py:141`. |
| **I-2** `cache_hit_rate` cross-contamination | `evaluation.py:129-130` | `cache_hits` fed from disk caches (`orchestrator.py:584`), `cache_misses` from the semantic cache (`orchestrator.py:624`). |
| **I-3** Latency-inflated completion radius | `orchestrator.py:474` | `effective_radius = 8.0 + max(0.0, float(avg_latency)) * v_agent * 2.0` unchanged. |
| **I-4** Attribution counters never reported | `cloud_llm_client.py:40-52` vs `evaluation.py` | 13 counters incremented via `record_call_category`; **0 occurrences** in `evaluation.py` or `orchestrator.py`. |
| **I-5** `PlanRepairer` dead | `orchestrator.py:85,145` | Imported and instantiated; **zero** `plan_repairer.` call sites anywhere in `src/`. |
| **I-6** Dead config keys | `configs/thresholds.yaml` | `confidence_gate`, `early_exit`: 1 occurrence in config, **0 consumers** in `src/`. |
| **I-7** Silent-default config keys | `configs/thresholds.yaml` | `plan_validity_threshold` and `min_dwell_steps` are **absent** from the file but consumed in code (1 and 3 sites) — silently defaulting. |
| **I-8** Three divergent completion conditions | `orchestrator.py:474`, `centralized_hybrid.py:196`, `decentralized_hybrid.py:606` | Latency-inflated radius vs two hardcoded `< 5.0` checks. |

### ⚠ NEW — found in this audit

| Issue | Proof |
|---|---|
| **I-9** `results_after.json` is not reproducible from the shipped tree | Repo file reports `search_rescue_s5 = 80.00`. Executed on the working tree: **90.00**, stable across `PYTHONHASHSEED` 0/1/2/3 and both `cache_responses` settings. All other 14 cells match. |
| **I-10** Shipped config is mock mode | `configs/llm.yaml:47` → `use_mock: true`. The repo as-shipped cannot produce real-LLM results. |
| **I-11** `logistics_s4` regresses to 33.33 under the scoped fix | Executed: 33.33 (scoped) vs 66.67 (pre-fix). My previous per-seed table reported 50.00 for this cell — that figure came from the pre-correction `idx > 0` build and was **stale**. Correcting the record. |
| **I-12** Repo hygiene | 68 files tracked under `scratch/`, `logs/`, `.llm_cache/`; untracked `files (5)/`, `files (5).zip` in the tree. |

---

# ISSUE I-1 — `is_valid` ignores the configured validity threshold

**Priority: HIGH**

**Problem.** `PlanValidityScore.is_valid` compares against a hardcoded `0.75`. `PlanContinuityEngine.__init__` accepts `validity_threshold` and `orchestrator.py:141` feeds it from `thresholds.get("plan_validity_threshold", 0.75)`. The configured value has no effect.

**Scientific impact.** Plan continuity is the single largest determinant of cloud planning calls — measured earlier, a seed with a 20.1% continuity pass rate incurred 72 cloud calls versus 2 for a 97.1% seed. The threshold governing that gate cannot currently be varied, so no sensitivity analysis is possible.

**IEEE reviewer impact.** A reviewer will ask how sensitive the results are to `0.75`, especially given that `V_plan` on affected seeds clusters within ±0.05 of it. "We cannot vary it" is not an answer. This is a Required-Revision-class question, not a nitpick.

**File:** `src/coordination/plan_continuity.py`
**Class:** `PlanValidityScore` (dataclass) and `PlanContinuityEngine.evaluate_plan_validity`
**Lines:** ~19-31 (dataclass), and the three `PlanValidityScore(...)` construction sites in `evaluate_plan_validity`

**Current logic.**
```python
@dataclass
class PlanValidityScore:
    """Quantitative evaluation breakdown for active global plan continuity."""
    task_completion_score: float = 1.0
    distance_feasibility_score: float = 1.0
    communication_quality_score: float = 1.0
    coalition_feasibility_score: float = 1.0
    resource_network_score: float = 1.0
    total_validity_score: float = 1.0

    @property
    def is_valid(self) -> bool:
        return self.total_validity_score >= 0.75
```

**Why it is incorrect.** The threshold is a property of the *engine configuration*, not of the score object, but the score object is what performs the comparison. The two were never connected.

**Required change.** Carry the threshold on the score and compare against it. Default stays `0.75` so behaviour at the default config is bit-for-bit unchanged.

**Implementation steps.**
1. Add a `validity_threshold: float = 0.75` field to `PlanValidityScore`, **after** `total_validity_score` so positional construction is unaffected.
2. Change `is_valid` to compare against `self.validity_threshold`.
3. Pass `validity_threshold=self.validity_threshold` at **all three** `PlanValidityScore(...)` construction sites inside `evaluate_plan_validity` (the two early-return guards and the final full construction). Missing one leaves a silent `0.75` path.
4. Add `plan_validity_threshold: 0.75` to `configs/thresholds.yaml` under the same block that holds `R_reach` (see I-7).

**Exact code.**
```python
    total_validity_score: float = 1.0
    # CONFIG FIX: the threshold was hardcoded below while
    # PlanContinuityEngine.__init__ accepted a `validity_threshold` argument fed
    # from thresholds.yaml:plan_validity_threshold. The configured value was
    # therefore dead. The threshold now travels with the score.
    validity_threshold: float = 0.75

    @property
    def is_valid(self) -> bool:
        return self.total_validity_score >= self.validity_threshold
```
At each construction site add the keyword argument, e.g.:
```python
            return PlanValidityScore(
                total_validity_score=0.0, validity_threshold=self.validity_threshold
            )
```
```python
            return PlanValidityScore(
                total_validity_score=1.0, validity_threshold=self.validity_threshold
            )
```
```python
            resource_network_score=s_res,
            total_validity_score=v_plan,
            validity_threshold=self.validity_threshold,
        )
```

**Modification order.** `plan_continuity.py` first, then `configs/thresholds.yaml`. No other file.
**Dependent files.** None. `orchestrator.py:141` already passes the value.

**Validation.**
1. `python3 -m pytest tests/ -q` → 50 passed (delete `experience_store.json` first).
2. `grep -c "validity_threshold=self.validity_threshold" src/coordination/plan_continuity.py` → **3**.
3. Behaviour-identity check at default: run `validate.py` before and after; every cell must be identical, because `0.75 == 0.75`.
4. Sensitivity check: set `plan_validity_threshold: 0.70` in config, re-run, confirm `cloud_planning_calls` changes. If it does not, the wiring is still broken.

**Expected metric impact.** At the default: **zero change to every metric** — this is the acceptance criterion. Only after deliberately changing the config value should anything move (lower threshold → fewer replans → fewer cloud calls/tokens; higher → the reverse).

**Side effects.** None at default. Positional construction of `PlanValidityScore` elsewhere would break if the field were inserted in the middle — it is appended, so it is safe.
**Regression risk: Very low.** **Confidence: High.**

---

# ISSUE I-2 — `cache_hit_rate` divides two unrelated subsystems

**Priority: HIGH**

**Problem.** `evaluation.py:129-130` computes `cache_hits / (cache_hits + cache_misses)` where `cache_hits` is fed from the **on-disk response caches** (`orchestrator.py:584`: `cloud_llm.usage.cache_hits + device_usage.cache_hits`) and `cache_misses` from the **semantic plan cache** (`orchestrator.py:624`). Numerator and denominator come from different caches.

**Scientific impact.** Every published `cache_hit_rate` value is not a rate of anything. If the manuscript cites semantic-cache effectiveness, that claim is unsupported.

**IEEE reviewer impact.** This is the kind of finding that costs credibility disproportionately: it is trivially checkable, and once found, a reviewer will re-examine every other metric.

**File:** `src/metrics/evaluation.py` then `src/coordination/orchestrator.py`
**Lines:** `evaluation.py:129-130`, dataclass field block, `finalize()` signature and body; `orchestrator.py` ~624

**Current logic.**
```python
        tot_cache = self.cache_hits + self.cache_misses
        hit_rate = round(self.cache_hits / tot_cache, 4) if tot_cache > 0 else self.cache_hit_rate
```

**Required change.** Use the semantic cache's own hit counter as the numerator, and expose the disk-cache hits under a separate, honestly-named key.

**Implementation steps.**
1. In `evaluation.py`, add `semantic_cache_hits: int = 0` next to `cache_misses` in the dataclass.
2. Replace the two lines above.
3. Add `"semantic_cache_hits"` and `"disk_cache_hits"` to the `to_dict()` payload.
4. Add `semantic_cache_hits: int = 0` to the `finalize()` signature and forward it in the constructor call.
5. In `orchestrator.py`, alongside the existing `cache_misses=` line, add the hits source.

**Exact code — `evaluation.py`:**
```python
        # METRIC-INTEGRITY FIX: `cache_hits` counts ON-DISK response-cache hits
        # recorded in CloudLLMClient.complete()/DeviceLLMClient, while
        # `cache_misses` counts SEMANTIC plan-cache misses recorded in
        # SemanticPlanCache.lookup(). Dividing one by the sum of both combined a
        # numerator and denominator from two unrelated subsystems, so the
        # published hit rates were not rates of anything.
        sem_hits = self.semantic_cache_hits
        tot_cache = sem_hits + self.cache_misses
        hit_rate = round(sem_hits / tot_cache, 4) if tot_cache > 0 else self.cache_hit_rate
```
In `to_dict()`, immediately after the existing `"cache_misses"` entry:
```python
            "semantic_cache_hits": self.semantic_cache_hits,
            "disk_cache_hits": self.cache_hits,
```

**Exact code — `orchestrator.py`**, immediately after the existing `cache_misses=` line:
```python
            semantic_cache_hits=getattr(
                getattr(self.cloud_llm, "semantic_cache", None), "cache_hits", 0
            ),
```

**Modification order.** `evaluation.py` first (it defines the parameter), then `orchestrator.py` (it supplies it). Reversing this raises `TypeError`.
**Dependent files.** Any downstream consumer of `cache_hit_rate` in `baseline_results_compare/` must be regenerated.

**Validation.**
1. `python3 -m pytest tests/ -q` → 50 passed.
2. Run `validate.py`; inspect one output JSON — `semantic_cache_hits`, `disk_cache_hits`, `cache_misses` all present.
3. Sanity: `cache_hit_rate == semantic_cache_hits / (semantic_cache_hits + cache_misses)`.

**Expected metric impact.** Reporting only. Success, cloud calls, tokens, communication, latency, memory: **all unchanged**. `cache_hit_rate` will change value — that is the fix, not a regression. Previously published hit rates must be retracted or recomputed.
**Regression risk: Very low.** **Confidence: High.**

---

# ISSUE I-3 — Completion radius inflates with network latency

**Priority: CRITICAL**

**Problem.** `orchestrator.py:474`:
```python
effective_radius = 8.0 + max(0.0, float(avg_latency)) * v_agent * 2.0
if dist(agent.position, subtask.target) < effective_radius:
```
With `latency.tau_max: 2.0` and a UAV at `max_speed: 15.0`, the completion radius grows from 8 m to **68 m** — an 8.5× larger completion zone under degraded communication.

**Scientific impact.** The benchmark becomes *easier* exactly as the network degrades. DACA-HMAS's central claim is robustness under degraded communication, so this inflates the headline metric precisely in the regime the claim concerns. Every reported success number is affected by an unquantified amount.

**IEEE reviewer impact.** **This is the single largest publication blocker in the codebase.** A reviewer who reads this line will discount every success comparison in the paper, including the AutoHMA-LLM comparison. It cannot be left unaddressed and unexplained.

**File:** `src/coordination/orchestrator.py`
**Function:** the per-step subtask completion check
**Lines:** ~472-477

**Current logic.** Radius = 8 m base, plus `avg_latency × agent_speed × 2`. Two other completion sites use a flat `< 5.0` (`centralized_hybrid.py:196`, `decentralized_hybrid.py:606`), so completion semantics also differ by execution path (see I-8).

**Why it is incorrect.** Latency is a *communication* property; completion is a *physical proximity* event. Coupling them means a worse network makes the physical task easier, which is not defensible under any reading of the scenario semantics.

**Required change.** Choose one of the three options below and document it in the manuscript. **Option A is recommended.**

### Option A — fixed radius (recommended)
```python
                    # BENCHMARK VALIDITY FIX: the completion radius previously
                    # grew with avg_latency (8.0 + latency*v_agent*2.0), which at
                    # tau_max=2.0 and uav speed 15.0 inflated it from 8 m to 68 m.
                    # That made missions systematically easier under exactly the
                    # degraded-communication conditions the system claims to
                    # handle well. Completion is a physical proximity event and
                    # must not depend on link quality.
                    effective_radius = COMPLETION_RADIUS_M
                    if dist(agent.position, subtask.target) < effective_radius:
```
with, at module scope near the other constants:
```python
# Physical completion radius (metres). Shared by all completion checks so that
# completion semantics do not vary by execution path. See also
# centralized_hybrid.py and decentralized_hybrid.py.
COMPLETION_RADIUS_M = 8.0
```

**Why 8.0 and not 5.0:** 8.0 is the existing base of the orchestrator's expression and is the value in force whenever latency is ~0, so it preserves the low-latency behaviour that most runs already exhibit. Using 5.0 would additionally change every stable-network result.

### Option B — cap the inflation
```python
                    effective_radius = min(
                        8.0 + max(0.0, float(avg_latency)) * v_agent * 2.0,
                        COMPLETION_RADIUS_CAP_M,   # e.g. 15.0
                    )
```
Acceptable **only** if the cap is derived from a stated physical argument (sensing uncertainty under delayed feedback), not chosen to preserve a number.

### Option C — leave as-is
**Not recommended.** If chosen, the manuscript must disclose the dependence explicitly and report success at both the inflated and fixed radius.

**Implementation steps (Option A).**
1. Add `COMPLETION_RADIUS_M = 8.0` at module scope in `orchestrator.py`.
2. Replace the `effective_radius` expression.
3. Import and use the same constant at `centralized_hybrid.py:196` and `decentralized_hybrid.py:606`, replacing `< 5.0` (this is I-8; do them together — splitting them leaves the inconsistency).
4. **Re-baseline every result.** All prior success numbers are void under the new semantics.

**Modification order.** `orchestrator.py` (defines the constant) → `centralized_hybrid.py` → `decentralized_hybrid.py`.
**Dependent files.** All of `experiments/results/`, `baseline_results_compare/*.csv`, and every success figure in the manuscript.

**Validation.**
1. `grep -rn "8.0 + max(0.0, float(avg_latency))" src/` → **no matches**.
2. `grep -rn "< 5.0" src/coordination/` → no completion-check matches remain.
3. `grep -rn "COMPLETION_RADIUS_M" src/` → 3 use sites + 1 definition.
4. `python3 -m pytest tests/ -q` → 50 passed. If a test asserts on the old radius, fix the test to the new constant and say so.
5. Run `validate.py` and compare against the stable-network profile: low-latency cells should move little; oscillatory cells will drop.

**Expected metric impact.** **Success: will decrease**, most in degraded profiles — this is the correct direction and must be reported as such, not treated as a regression. Cloud API calls / tokens: slight increase, since subtasks stay incomplete longer and trigger more replans. Communication steps: slight increase. Computation, memory, switching: unchanged. Wall clock: slight increase.

**Side effects.** Prior numbers become non-comparable. That is unavoidable and is the point.
**Regression risk: High on the reported numbers, zero on correctness.** **Confidence: High.**

---

# ISSUE I-4 — Call-attribution counters are computed but never reported

**Priority: MEDIUM**

**Problem.** `cloud_llm_client.py:40-52` defines 13 counters (`initial_planning_calls`, `completion_replan_calls`, `cqi_replan_calls`, `packet_loss_replan_calls`, `latency_replan_calls`, `switch_replan_calls`, `coalition_retry_calls`, `hallucination_retry_calls`, `experience_store_hits`, `semantic_cache_hits`, `plan_continuity_reuse`, `device_local_reallocation`, `cache_misses`). `record_call_category` is invoked at `:305` and `:319`. **Zero** of them appear in `evaluation.py` or `orchestrator.py`, so none reach the output JSON.

**Scientific impact.** The data needed to answer "*why* were there N cloud calls?" is being computed each run and discarded. That is exactly the breakdown a reviewer asks for when an efficiency claim is made.

**IEEE reviewer impact.** Not a blocker, but a strong asset currently going to waste. The work is ~90% done.

**File:** `src/metrics/evaluation.py` then `src/coordination/orchestrator.py`
**Lines:** `evaluation.py` dataclass block ~58-60 (next to the R8 counters), `to_dict()` ~157-159, `finalize()` signature ~287-289 and body ~397-399; `orchestrator.py` ~584-590.

**Required change.** Serialize the counters as a single nested dict rather than 13 flat fields — fewer signature changes and it keeps the JSON readable.

**Implementation steps.**
1. In `cloud_llm_client.py`, add a method to the usage dataclass:
```python
    def call_attribution(self) -> dict[str, int]:
        """Cloud call counts broken down by triggering cause (reporting only)."""
        return {
            "initial_planning_calls": self.initial_planning_calls,
            "completion_replan_calls": self.completion_replan_calls,
            "cqi_replan_calls": self.cqi_replan_calls,
            "packet_loss_replan_calls": self.packet_loss_replan_calls,
            "latency_replan_calls": self.latency_replan_calls,
            "switch_replan_calls": self.switch_replan_calls,
            "coalition_retry_calls": self.coalition_retry_calls,
            "hallucination_retry_calls": self.hallucination_retry_calls,
        }
```
2. In `evaluation.py`, add the field next to the R8 counters:
```python
    cloud_call_attribution: dict = field(default_factory=dict)
```
(ensure `from dataclasses import field` is imported), add to `to_dict()`:
```python
            "cloud_call_attribution": self.cloud_call_attribution,
```
add `cloud_call_attribution: dict | None = None,` to `finalize()` and forward `cloud_call_attribution=cloud_call_attribution or {},`.
3. In `orchestrator.py`, next to the existing `cloud_network_calls=` line:
```python
            cloud_call_attribution=self.cloud_llm.usage.call_attribution(),
```
4. **Also fix the attribution gap:** `record_call_category` is called in the disk-cache-hit branch (`:279`) but the categories are only meaningful for calls that reached the provider. Either exclude it there or add a `disk_cache_served` category. Document whichever you choose.

**Modification order.** `cloud_llm_client.py` → `evaluation.py` → `orchestrator.py`.
**Validation.** Run `validate.py`; confirm `cloud_call_attribution` in the JSON and that its values sum to `cloud_planning_calls` (or, if they do not, that the difference is exactly `cloud_disk_cache_hits` — either is fine, but state which invariant holds).
**Expected metric impact.** **Reporting only — no metric changes.**
**Regression risk: Very low.** **Confidence: High.**

---

# ISSUE I-5 — `PlanRepairer` is instantiated but never called

**Priority: LOW**

**Problem.** `orchestrator.py:85` imports `PlanRepairer`; `:145` constructs `self.plan_repairer = PlanRepairer(r_reach=...)`. There are **zero** `plan_repairer.` call sites in `src/`. The 119-LOC module never executes.

**Scientific impact.** None on results. It is a maintenance and honesty issue: a reader of the architecture description will assume plan repair runs.

**IEEE reviewer impact.** Minor, but if the manuscript lists "coalition/plan repair" as a component, the claim is unsupported.

**Required change — pick one:**
- **(a) Remove it** (recommended if not needed): delete the import at `:85` and the construction at `:145`; delete `src/coordination/plan_repair.py`.
- **(b) Wire it in**: call it from the repair path in `centralized_hybrid.py`. Do **not** do this before submission — it changes runtime behaviour and would require full re-validation for no measured benefit.

**Validation (a).** `grep -rn "PlanRepairer\|plan_repair" src/` → no matches; `python3 -m pytest tests/ -q` → 50 passed.
**Expected metric impact.** **None** (dead code).
**Regression risk: Very low.** **Confidence: High.**

---

# ISSUE I-6 / I-7 — Dead and silently-defaulting config keys

**Priority: LOW (I-6) / MEDIUM (I-7)**

**Problem.**
- **I-6:** `confidence_gate` and `early_exit` appear in `configs/thresholds.yaml` with **0 consumers** in `src/`. Setting them does nothing.
- **I-7:** `plan_validity_threshold` and `min_dwell_steps` are **consumed** in code (1 and 3 sites) but **absent** from `configs/thresholds.yaml`, so both silently use hardcoded defaults (`0.75` and `5`).

**Scientific impact.** I-7 is the real one: two parameters that materially affect switching and replanning are not visible in the configuration a reader would inspect for reproducibility.

**IEEE reviewer impact.** A reviewer reproducing the work from the config file will not know these parameters exist.

**Required change.**
1. Delete the `confidence_gate` and `early_exit` blocks from `configs/thresholds.yaml`, **or** add a comment `# NOT IMPLEMENTED - reserved` above each. Deletion is cleaner.
2. Add the two missing keys with their current effective defaults, alongside the block that already contains `R_reach`:
```yaml
  # Plan continuity: minimum V_plan score for the active plan to be reused.
  plan_validity_threshold: 0.75

acds:
  # Minimum steps an architecture must be held before another switch is allowed.
  min_dwell_steps: 5
```
Place `min_dwell_steps` in whichever block `switch_engine.py` reads (`acds`), matching the existing `.get("min_dwell_steps", 5)` lookup path.

**Validation.** Run `validate.py` before and after; **every cell must be identical**, since the added values equal the current defaults. If anything changes, a key was placed in the wrong block.
**Expected metric impact.** **None.**
**Regression risk: Very low.** **Confidence: High.**

---

# ISSUE I-8 — Three divergent subtask-completion conditions

**Priority: HIGH** (fix together with I-3)

**Problem.** Completion is decided in three places with three different rules: `orchestrator.py:474` (latency-inflated), `centralized_hybrid.py:196` (`< 5.0`), `decentralized_hybrid.py:606` (`< 5.0`). Which rule fires depends on execution path, so **completion semantics differ between centralized and decentralized modes.**

**Scientific impact.** Success is not measured consistently across the architectures being compared. Since adaptive switching between those architectures is the paper's contribution, the comparison is confounded at the measurement level.

**IEEE reviewer impact.** Serious. "Your success metric is defined differently in the two modes you are comparing" is a direct threat to the central claim.

**Required change.** Route all three through one shared constant/function (see I-3, Option A, step 3).

**Exact code.** In `centralized_hybrid.py` and `decentralized_hybrid.py`, replace `< 5.0` with the imported constant:
```python
from src.coordination.constants import COMPLETION_RADIUS_M   # or from orchestrator
...
                if dist(agent.position, subtask.target) < COMPLETION_RADIUS_M:
```
Prefer a small `src/coordination/constants.py` to avoid a circular import between `orchestrator` and the two planners.

**Validation.** `grep -rn "mark_subtask_complete" src/` → every call site is guarded by the same constant. `python3 -m pytest tests/ -q` → 50 passed.
**Expected metric impact.** Success will shift where the two rules previously disagreed (5.0 vs 8.0 base). Re-baseline with I-3.
**Regression risk: Medium on numbers, low on correctness.** **Confidence: High.**

---

# ISSUE I-9 — `results_after.json` does not reproduce from the shipped tree

**Priority: CRITICAL**

**Problem.** The repo's `results_after.json` reports `search_rescue_s5 = 80.00`. Executing the shipped `validate.py` against the shipped working tree gives **90.00**, stable across `PYTHONHASHSEED` 0/1/2/3 and both `cache_responses: true|false`. The other 14 of 15 cells match exactly. Running the **committed** (pre-scoped) decomposer instead gives `search_rescue = 94.00` with s5 = 100.00 — so the file matches neither revision.

**Scientific impact.** A results file that cannot be regenerated from the code in the same commit is the definition of a reproducibility failure, and it is one cell away from being fine.

**IEEE reviewer impact.** **Blocker.** Artefact evaluation regenerates the results file. A mismatch here, in a paper whose contribution is partly reproducibility, is fatal.

**Required change.** Regenerate `results_after.json` from the current tree and commit it together with the code that produced it.

**Implementation steps.**
1. Commit the working-tree decomposer change (`git add src/decomposition/distance_feasible_decomp.py`).
2. `rm -f experience_store.json`
3. `PYTHONHASHSEED=0 python3 validate.py results_after.json`
4. Commit `results_after.json` in the **same commit** as the code.
5. Record the exact invocation (including `PYTHONHASHSEED=0` and `use_mock`) in `README.md`.

**Validation.** Re-run step 3 into a temp file and `diff` against the committed one — must be byte-identical.
**Expected metric impact.** Reporting only; the numbers become correct rather than stale.
**Regression risk: None.** **Confidence: High.**

---

# ISSUE I-10 — Shipped configuration is mock mode

**Priority: CRITICAL**

**Problem.** `configs/llm.yaml:47` → `use_mock: true`, `:49` → `cache_responses: false`. The repository as shipped runs against a canned mock, not the NVIDIA cloud planner.

**Scientific impact.** Anyone cloning the repo and running it reproduces **mock** numbers, which are not the numbers in the manuscript. Mock success (e.g. inspection 82.50) and real success (85.00–97.50) differ materially.

**IEEE reviewer impact.** **Blocker.** Combined with I-9 this reads as "the shipped artefact does not reproduce the paper."

**Required change.**
1. Set `use_mock: false` and `cache_responses: true` as the committed default, matching how the manuscript results were produced.
2. Add a clearly-labelled mock block in the same file, commented out, for offline testing.
3. Document in `README.md`: which setting produces manuscript results, and that `PYTHONHASHSEED=0` plus `rm -f experience_store.json` are required for reproducibility.
4. **Additionally set `temperature: 0`** (`configs/llm.yaml:7`, currently `0.2`) for the reproducibility run. With `0.2` and no API seed, two runs of identical code produce different plans — this is why no previous A/B on the real stack could attribute a change to the code. This is the highest-value single line in the whole guide.

**Validation.** Fresh clone → `grep use_mock configs/llm.yaml` → `false`; `grep temperature configs/llm.yaml` → `0` for the active cloud block.
**Expected metric impact.** None from the config change itself; it makes the real-stack numbers reproducible, which is the point.
**Regression risk: None.** **Confidence: High.**

---

# ISSUE I-11 — `logistics_s4` regresses to 33.33 under the scoped fix

**Priority: MEDIUM (disclose; do not "fix")**

**Problem.** Executed on the shipped tree: `logistics_s4 = 33.33` versus `66.67` before any decomposer fix. This is despite the zero-perturbation property holding (0 existing assignments moved). Correcting my own prior record: I previously reported `50.00` for this cell, which came from a superseded `idx > 0` build.

**Verified mechanism.** Two compounding effects, both previously traced:
1. `robot_5` (speed 3.0) is the **only** in-reach agent for three rescued targets and cannot service them within 200 steps.
2. Before the fix those three subtasks were assigned to nobody and completed **incidentally** — completion is proximity-triggered, so an unassigned subtask finishes whenever any agent wanders within the radius. The 66.67% baseline therefore included accidental completions.

**Scientific impact.** This is evidence for the I-3/I-8 benchmark-validity problem, not an argument against the fix. A change that assigns *more* subtasks correctly lowers measured success because the baseline was inflated by proximity accidents.

**IEEE reviewer impact.** Disclose it. Presenting the logistics mean without this cell would be selective reporting.

**Required change.** **None to the solver.** Fix I-3/I-8 first, then re-measure. If the incidental-completion channel is closed and s4 still regresses, revisit — but not before, because the current baseline is not trustworthy.

**Explicitly rejected "fixes":** raising `r_reach`; adding a speed floor to candidate selection; capping per-agent assignments; any scenario- or seed-specific special case. All are metric gaming.

**Confidence: High** (mechanism traced and executed).

---

# ISSUE I-12 — Repository hygiene

**Priority: LOW**

**Problem.** 68 files tracked under `scratch/`, `logs/`, `.llm_cache/`. Untracked `files (5)/`, `files (5).zip`, `experience_store.json` sit in the tree. `experience_store.json` in the repo root causes a spurious failure in `tests/test_determinism_audit.py` when stale.

**Required change.** Add to `.gitignore`:
```
scratch/
logs/
.llm_cache/
experience_store.json
files*/
files*.zip
```
then `git rm -r --cached scratch logs .llm_cache`.

**Validation.** `git status --porcelain` clean after a `validate.py` run; `python3 -m pytest tests/ -q` → 50 passed from a fresh clone without manual deletion.
**Expected metric impact.** None.
**Regression risk: None.** **Confidence: High.**

---

## Recommended Modification Order

1. **I-10** (config: `use_mock: false`, `temperature: 0`) — everything downstream depends on being able to measure.
2. **I-3 + I-8 together** (completion radius unified and decoupled from latency) — changes semantics, so it must precede any re-baselining.
3. **I-1** (validity threshold) and **I-6/I-7** (config keys) — behaviour-neutral at defaults; land before re-baselining so the config is final.
4. **Re-baseline everything**: `n ≥ 20` seeds × 4 profiles, `PYTHONHASHSEED=0`, `temperature: 0`.
5. **I-2** and **I-4** (reporting only) — can land any time before writing tables.
6. **I-9** (regenerate `results_after.json`) — must be last, after all behaviour changes.
7. **I-5**, **I-12** (cleanup) — any time.
8. **I-11** — re-measure only after step 4; take no action before then.

---

## Final Review

**1. Implemented correctly:** R1 deterministic seeding (verified executing across 4 hash seeds); R8 honest API accounting (per-attempt network counting, cache hits excluded from `cloud_api_calls`, all serialized); Optimization B; the scoped tier-walk fix (byte-identical to spec, zero-perturbation verified); 50/50 tests.

**2. Partially implemented:** call attribution (computed, not reported — I-4); `PlanRepairer` (constructed, never called — I-5); config surface (two keys dead, two missing — I-6/I-7).

**3. Still missing:** I-1, I-2, I-3, I-8.

**4. Incorrect:** `cache_hit_rate` (I-2) is arithmetically meaningless; the latency-coupled completion radius (I-3) inverts the benchmark's difficulty; `results_after.json` (I-9) does not correspond to any revision in the repo.

**5. Do NOT change:** the scoped decomposer — it is byte-identical to spec with the zero-perturbation property verified; R1 seeding; R8 counters; Optimization B; the six-category communication breakdown; `paper_communication_steps` **during this pass** (changing the comparison variable while also changing completion semantics would confound both — do it as a separate, labelled revision).

**6. Discard these earlier recommendations:** re-implementing coalition repair / constraint verification / incremental handoff / hierarchical planning (all verified present); the unscoped travel-time cost (superseded by the scoped variant); any further optimization targeting `paper_communication_steps` (it remains branch-asymmetric, so optimizing it optimizes an artefact).

**7. Now over-engineering:** batch planning; parallel coalition formation; adaptive thresholds; task-weighted CQI; connectivity-graph coalition criterion; attention/GNN/learned routing. None touch a publication blocker, and each would need its own re-validation.

**8. Publication blockers:** **I-3** (benchmark easier under degradation), **I-8** (success defined differently per architecture), **I-9** (results not reproducible from code), **I-10** (shipped config is mock + stochastic temperature). Everything else is a revision item.

**9. Submission judgement.**

- **IEEE Transactions — no, not yet.** I-3 alone would draw a Reject on experimental validity: a reviewer who reads `orchestrator.py:474` will discount every success number, and the comparison against AutoHMA-LLM rests on those numbers. I-8 compounds it by defining success differently in the two architectures the paper compares. I-9 and I-10 mean the artefact does not reproduce the manuscript. These are four days of work, not four months — but they are load-bearing.
- **Top-tier conference (NeurIPS/ICML/ICRA/IROS/AAAI) — no.** Same blockers, plus these venues weight ablations and baselines heavily, and no baseline (B1/B2) result records exist in the repo at all.
- **Mid-tier conference / workshop — plausible after I-3, I-8, I-9, I-10**, provided claims are restricted to *within-study* comparisons ("relative to our own ablated configurations") rather than cross-paper efficiency claims. The within-study framing is genuinely defensible today and needs no new science.

**The engineering here is better than the evaluation.** R1 and R8 are properly done, the instrumentation breadth is above average for this venue, and the scoped decomposer fix carries a mechanically-verified safety property that most submissions could not state. What is blocking publication is four measurement decisions, none of which require touching the contribution.
