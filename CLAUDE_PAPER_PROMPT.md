# Master Research Paper Generation Prompt for Claude

> **Instructions for Claude**: You are acting as a Senior AI Researcher and IEEE Transactions Author. Use the system architecture, empirical dataset, baseline comparison, and metric auditing rules below to write or generate the manuscript/results section for the **DACA-HMAS** (Dynamic Adaptive Communication-Aware Heterogeneous Multi-Agent System) paper.

---

## 4. Operational Baseline & Baseline Results

### 4.1 Baseline Results (AutoHMA-LLM Table III)

Use the following values exactly as reported in **Table III** of the AutoHMA-LLM paper.

| Scenario        | Success (%) | Steps | API Calls | Tokens | Memory (MB) | Computation (s) |
| --------------- | ----------: | ----: | --------: | -----: | ----------: | --------------: |
| Logistics       |       85.73 |  5.11 |      4.23 | 152.87 |          50 |             8.5 |
| Inspection      |       85.67 |  3.84 |      4.85 |  97.10 |          40 |             7.8 |
| Search & Rescue |       82.03 |  4.30 |      3.41 | 166.69 |          55 |             9.2 |

*These values represent the published baseline and must be used exclusively for comparative benchmarking.*

---

## 5. Important Metric Definition Audit

Before generating any comparison figures or comparative narrative, compare the metric definitions in the AutoHMA-LLM paper against the implementation in the DACA-HMAS codebase.

### Audit Summary Matrix

| Metric | AutoHMA-LLM Definition | DACA-HMAS Implementation | Directly Comparable? | Recommended Treatment |
| :--- | :--- | :--- | :---: | :--- |
| **Success** | Task completion rate (% subtasks finished within step limit) | `success_rate` (`completed / total * 100`) | **YES** | Compare directly in text and figures. |
| **Steps** | Communication / coordination steps for task decomposition | `steps` (physical simulation ticks / Gym movement steps) | **NO** | Do not force direct step-to-step equivalency. Use `consensus_rounds` or annotate definition difference. |
| **API Calls** | Total planner invocations across architecture | `api_calls` (`cloud_planning_calls + device_planning_calls`) | **NOT DIRECTLY** | State counting methodology differences explicitly (Central Cloud + Edge Device calls vs Centralized only). |
| **Tokens** | Prompt + completion tokens exchanged during planning | `tokens` (`cloud_tokens + device_tokens`) | **COMPARABLE** | Compare directly while highlighting token distribution (Cloud vs. Edge offloading). |
| **Memory** | Measured runtime memory consumption of classical control tier | `memory_mb` (Google Colab allocated runtime ceiling ~12 GB / static threshold) | **NOT COMPARABLE** | Omit from efficiency claims or footnote as environment allocation ceiling vs. runtime algorithm usage. |
| **Computation Time** | End-to-end wall-clock execution time (seconds) | `computation_s` (`perf_counter` elapsed time) | **YES** | Compare directly as total latency / throughput metric. |

*Do **not** assume the metrics are equivalent without auditing.*

---

### Detailed Metric Verification Guidelines

#### 5.1 Success Rate
- **Verification**: Ensure `success_rate` in DACA-HMAS reflects subtask completion ratio equivalent to the paper's task completion rate.
- **Treatment**: If equivalent, compare directly without modification.

#### 5.2 Steps
- **Definition Gap**: The AutoHMA-LLM paper's **Steps** metric represents:
  > *Communication / coordination steps required for task decomposition and execution.*
- **Code Audit**: DACA-HMAS logs several distinct step-related metrics:
  - `steps` (physical environment timesteps / movement ticks)
  - `distributed_replanning_count`
  - `replanning_count`
  - `coalition_change_count`
- **Treatment**: Determine which metric is the closest paper-equivalent (e.g., consensus / replanning rounds). If no exact equivalent exists, **do not force a comparison**. Instead, explicitly state that definitions differ and explain why (physical movement steps vs. coordination rounds).

#### 5.3 API Calls
- **Definition Gap**: The paper counts API calls differently from DACA-HMAS. DACA-HMAS records:
  - `cloud_planning_calls`
  - `device_planning_calls`
  - `api_calls`
- **Verification**: Check whether `api_calls` equals `cloud_planning_calls + device_planning_calls`.
- **Treatment**: If the counting methodology differs from the paper (due to domain-level edge LLM invocations), state this explicitly. Do not claim a direct apples-to-apples comparison.

#### 5.4 Memory
- **Special Treatment Required**: The AutoHMA-LLM paper reports **measured runtime memory usage** of their system (whose device layer uses classical control algorithms like PID/NMPC/Q-learning consuming ~40–55 MB).
- **Code & Environment Audit**: DACA-HMAS experiments were executed on **Google Colab** (or GPU runtime). The reported `memory_mb` value (~12,288 MB / 12 GB) reflects the **allocated runtime memory limit of the Colab execution environment**, rather than actual memory consumed by the algorithm during execution. Furthermore, DACA-HMAS executes Device LLMs on edge nodes rather than classical PID loops.
- **Treatment**: The reported memory values are **architecturally different** and **must not be interpreted as an efficiency comparison**. Verify from implementation whether `memory_mb` is:
  1. Measured runtime memory,
  2. Allocated runtime memory limit,
  3. Configured threshold, or
  4. GPU allocation ceiling.
- **Action**: 
  - **Option 1 (Recommended)**: Remove the Memory comparison from comparative figures/tables.
  - **Option 2**: Include it with a clear footnote explaining that it reflects the execution environment allocation ceiling rather than algorithmic memory consumption.
  - *Do **not** present it as evidence that DACA-HMAS consumes 12 GB of memory if the implementation does not actually measure dynamic runtime usage.*

#### 5.5 Computation Time
- **Verification**: Verify that `computation_s` represents wall-clock execution time from start to finish.
- **Treatment**: If verified, compare directly.

---

## 6. Results Presentation Protocol: Dual Tables

Because certain metrics have different underlying definitions between AutoHMA-LLM and DACA-HMAS, generate **two distinct result tables** in the manuscript:

### Table A: Raw DACA-HMAS Metrics
*Presents raw DACA-HMAS metrics exactly as produced by the experimental execution framework (`run_a5_eval.py`).*

| Scenario | Success (%) | Steps (Ticks) | Cloud API Calls | Device API Calls | Total API Calls | Cloud Tokens | Device Tokens | Total Tokens | Computation Time (s) | Memory (Colab Limit MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |

### Table B: Paper-Equivalent Benchmarking Comparison
*Presents a scientifically rigorous comparison against AutoHMA-LLM Table III values.*

For every metric in Table B, explicitly state whether it is:
- **Directly Comparable**
- **Transformed**
- **Estimated**
- **Not Comparable**

If a transformation is applied, explain the exact mathematical / logical transformation. Never fabricate values. If a paper-equivalent metric cannot be derived, state this explicitly.

| Metric | AutoHMA-LLM Baseline | DACA-HMAS Equivalent | Status | Transformation / Methodological Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Success Rate (%)** | Table III Value | DACA-HMAS `success_rate` | Directly Comparable | 1:1 match on task completion percentage. |
| **Steps / Rounds** | Table III Value | `consensus_rounds` / Annotated | Definition Differs | Baseline measures coordination rounds; DACA-HMAS reports simulation movement ticks. |
| **API Calls** | Table III Value | `cloud_planning_calls` + `device_planning_calls` | Methodology Differs | Baseline counts single central planner calls; DACA-HMAS includes hierarchical edge LLM calls. |
| **Token Usage** | Table III Value | `total_tokens` | Comparable | Evaluates total token overhead across centralized and edge tiers. |
| **Memory (MB)** | Table III Value | Environment Ceiling Footnoted | Not Comparable | Baseline: Classical control runtime RAM (~40-55 MB). DACA-HMAS: Colab allocated GPU/Host limit (~12 GB). |
| **Computation (s)** | Table III Value | `computation_s` | Directly Comparable | 1:1 match on wall-clock execution time. |

---

## 7. Figure Generation Rules

When producing comparative charts and visual figures:

1. **Success Rate**: Compare directly on bar / line charts.
2. **Token Usage**: Compare directly; optionally show stacked bars for Cloud vs. Device tokens.
3. **Computation Time**: Compare directly on execution latency charts.
4. **Steps**: Compare **only if** a valid paper-equivalent metric exists; otherwise add a visible chart annotation stating that step definitions differ (Simulation Ticks vs. Coordination Rounds).
5. **API Calls**: Compare **only if** counting methodology matches; otherwise annotate that DACA-HMAS includes domain-level Device LLM calls.
6. **Memory**: Either omit memory from comparative figures entirely OR include a prominent caption note that values reflect environment allocation ceilings rather than runtime execution footprints.

> **Caption Transparency Requirement**: Every figure caption must mention any metric-definition differences affecting interpretation so IEEE reviewers cannot misread the comparison. This makes the manuscript transparent, defensible, and scientifically rigorous.
