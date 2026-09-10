"""Cloud LLM client for global task decomposition and coalition formation."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import get_llm_config, project_root
from src.llm.exceptions import ExperimentFailed, FailureReport

_FAILURE_SENTINEL = "__CLOUD_LLM_FAILURE__"

def _log(msg: str) -> None:
    print(f"[LLM] {msg}")


@dataclass
class LLMUsage:
    tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    retry_tokens: int = 0
    api_calls: int = 0
    cloud_api_calls: int = 0
    logical_requests: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    retried_calls: int = 0
    cache_hits: int = 0
    cloud_bytes: int = 0
    llm_wait_s: float = 0.0
    cloud_inference_time_s: float = 0.0

    # Explicit R8 Counters
    cloud_network_calls: int = 0
    cloud_disk_cache_hits: int = 0
    cloud_failed_attempts: int = 0

    # Phase 2 Breakdown Counters
    initial_planning_calls: int = 0
    completion_replan_calls: int = 0
    cqi_replan_calls: int = 0
    packet_loss_replan_calls: int = 0
    latency_replan_calls: int = 0
    switch_replan_calls: int = 0
    coalition_retry_calls: int = 0
    hallucination_retry_calls: int = 0
    cache_misses: int = 0
    experience_store_hits: int = 0
    semantic_cache_hits: int = 0
    plan_continuity_reuse: int = 0
    device_local_reallocation: int = 0

    def reset(self) -> None:
        self.tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.retry_tokens = 0
        self.api_calls = 0
        self.cloud_api_calls = 0
        self.logical_requests = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.retried_calls = 0
        self.cache_hits = 0
        self.cloud_bytes = 0
        self.llm_wait_s = 0.0
        self.cloud_inference_time_s = 0.0
        self.cloud_network_calls = 0
        self.cloud_disk_cache_hits = 0
        self.cloud_failed_attempts = 0
        self.initial_planning_calls = 0
        self.completion_replan_calls = 0
        self.cqi_replan_calls = 0
        self.packet_loss_replan_calls = 0
        self.latency_replan_calls = 0
        self.switch_replan_calls = 0
        self.coalition_retry_calls = 0
        self.hallucination_retry_calls = 0
        self.cache_misses = 0
        self.experience_store_hits = 0
        self.semantic_cache_hits = 0
        self.plan_continuity_reuse = 0
        self.device_local_reallocation = 0

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

    def record_call_category(self, caller: str, reason: str | None = None) -> None:
        """Map caller/reason tag to exact mutually exclusive Phase 2 category counter."""
        effective_reason = reason or caller or "initial_planning"
        tag = str(effective_reason).lower()
        caller_tag = str(caller).lower()
        if "form_coalitions_retry" in caller_tag or "hallucination" in tag:
            self.hallucination_retry_calls += 1
        elif "coalition_retry" in tag:
            self.coalition_retry_calls += 1
        elif "completion" in tag or "task_completed" in tag:
            self.completion_replan_calls += 1
        elif "cqi" in tag:
            self.cqi_replan_calls += 1
        elif "packet_loss" in tag:
            self.packet_loss_replan_calls += 1
        elif "latency" in tag:
            self.latency_replan_calls += 1
        elif "switch" in tag or "architecture" in tag:
            self.switch_replan_calls += 1
        else:
            self.initial_planning_calls += 1


@dataclass
class CloudLLMClient:
    config: dict[str, Any] = field(default_factory=get_llm_config)
    usage: LLMUsage = field(default_factory=LLMUsage)
    max_retries: int = 3
    backoff_base: float = 1.0  # seconds; sequence becomes 1, 2, 4

    _last_assignments: dict[str, list[str]] = field(default_factory=dict, init=False, repr=False)
    _last_coalitions: list[dict] = field(default_factory=list, init=False, repr=False)
    device_fallback_decompose: Any = field(default=None, init=False, repr=False)
    device_fallback_coalitions: Any = field(default=None, init=False, repr=False)
    

    _client: Any = field(default=None, init=False, repr=False)
    _client_provider: str | None = field(default=None, init=False, repr=False)
    hallucination_stats: dict[str, Any] = field(
        default_factory=lambda: {
            "total_events": 0,
            "partial_strips": 0,
            "full_strips": 0,
            "retry_attempts": 0,
            "retry_successes": 0,
            "substitutions": 0,
            "singleton_fallbacks": 0,
            "events_log": [],
        },
        init=False,
        repr=False,
    )

    # Experiment context, set once by the orchestrator so a failure report
    # can be fully populated at the point of failure. Purely descriptive —
    # never used to change planning behavior.
    experiment_scenario: str | None = field(default=None, init=False, repr=False)
    experiment_architecture: str | None = field(default=None, init=False, repr=False)
    experiment_network_profile: str | None = field(default=None, init=False, repr=False)
    experiment_seed: int | None = field(default=None, init=False, repr=False)
    current_step: int | None = field(default=None, init=False, repr=False)
    
    summarizer: Any = field(default=None, init=False, repr=False)
    semantic_cache: Any = field(default=None, init=False, repr=False)
    prompt_reduction_percent: float = field(default=0.0, init=False, repr=False)
    opt_prompt_compression: bool = field(default=True, init=False, repr=False)
    opt_constrained_output: bool = field(default=True, init=False, repr=False)
    max_completion_tokens: int = field(default=256, init=False, repr=False)

    def __post_init__(self) -> None:
        from src.config import get_thresholds
        from src.llm.state_summarizer import CompactStateSummarizer
        from src.llm.semantic_cache import SemanticPlanCache

        th = get_thresholds()
        opts = th.get("optimizations", {})

        sum_cfg = opts.get("summarization", {})
        self.summarizer = CompactStateSummarizer(enabled=sum_cfg.get("enabled", True))

        cache_cfg = opts.get("semantic_cache", {})
        self.semantic_cache = SemanticPlanCache(
            enabled=cache_cfg.get("enabled", True),
            similarity_threshold=float(cache_cfg.get("similarity_threshold", 0.90)),
            max_cache_age=int(cache_cfg.get("max_cache_age", 15)),
        )

        p_cfg = opts.get("prompt_compression", {})
        self.opt_prompt_compression = p_cfg.get("enabled", True)

        c_cfg = opts.get("constrained_output", {})
        self.opt_constrained_output = c_cfg.get("enabled", True)
        self.max_completion_tokens = int(c_cfg.get("max_completion_tokens", 256))

    def configure_experiment_context(
        self,
        scenario: str,
        architecture: str,
        network_profile: str,
        seed: int,
    ) -> None:
        """Called once by the orchestrator/runner before an experiment starts,
        purely so a failure can be reported with full metadata."""
        self.experiment_scenario = scenario
        self.experiment_architecture = architecture
        self.experiment_network_profile = network_profile
        self.experiment_seed = seed

    def set_step(self, step: int) -> None:
        """Called each simulation step so a mid-run failure records where it happened."""
        self.current_step = step

    # ------------------------------------------------------------------
    # Cache helpers — memoization of PRIOR genuine LLM responses, not a
    # failure-recovery mechanism. See explanation section below.
    # ------------------------------------------------------------------
    def _cache_path(self, prompt: str) -> Path | None:
        if not self.config.get("cache_responses", True):
            return None
        cache_dir = project_root() / self.config.get("cache_dir", ".llm_cache")
        cache_dir.mkdir(exist_ok=True)
        key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        return cache_dir / f"cloud_{key}.json"

    def _read_cache(self, path: Path) -> dict | None:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return None

    def _write_cache(self, path: Path, data: dict) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    # ------------------------------------------------------------------
    # Client lifecycle (unchanged from production version)
    # ------------------------------------------------------------------
    def _timeout(self) -> float:
        return float(self.config.get("cloud", {}).get("timeout", 420))

    def _get_client(self):
        cloud = self.config["cloud"]
        provider = cloud.get("provider", "openai")
        if self._client is not None and self._client_provider == provider:
            return self._client

        timeout = self._timeout()
        if provider == "openai":
            from openai import OpenAI
            key = os.environ.get(cloud.get("api_key_env", "OPENAI_API_KEY"))
            self._client = OpenAI(api_key=key, timeout=timeout)
        elif provider == "groq":
            from openai import OpenAI
            key = os.environ.get(cloud.get("api_key_env", "GROQ_API_KEY"))
            self._client = OpenAI(
                api_key=key, base_url="https://api.groq.com/openai/v1", timeout=timeout
            )
        elif provider == "anthropic":
            import anthropic
            key = os.environ.get(cloud.get("api_key_env", "ANTHROPIC_API_KEY"))
            self._client = anthropic.Anthropic(api_key=key, timeout=timeout)
        elif provider == "nvidia":
            from openai import OpenAI
            key = os.environ.get(cloud.get("api_key_env", "NVIDIA_API_KEY"))
            self._client = OpenAI(
                api_key=key,
                base_url="https://integrate.api.nvidia.com/v1",
                timeout=timeout,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        self._client_provider = provider
        return self._client

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def complete(self, prompt: str, system: str = "", caller: str = "") -> str:
        self.usage.logical_requests += 1
        t_start = time.perf_counter()
        step = getattr(self, "current_step", 0)
        cache_path = self._cache_path(prompt)
        if cache_path:
            cached = self._read_cache(cache_path)
            if cached:
                before = self.usage.cloud_api_calls
                # Option A (Actual HTTP/API requests): Cache hit does NOT increment cloud_api_calls network request counter!
                self.usage.cloud_disk_cache_hits += 1
                self.usage.record_call_category(caller, getattr(self, "active_replan_reason", None))
                self.usage.cache_hits += 1
                self.usage.successful_calls += 1
                after = self.usage.cloud_api_calls
                elapsed = time.perf_counter() - t_start
                self.usage.llm_wait_s += elapsed
                self.usage.cloud_inference_time_s += elapsed
                p_bytes = len(prompt.encode("utf-8")) + len(cached["response"].encode("utf-8"))
                self.usage.cloud_bytes += p_bytes
                print(f"[COUNTER] metric=cloud_planning_calls step={step} before={before} after={after} reason={caller or 'cloud_complete'} caller=CloudLLMClient.complete()")
                print(f"[CLOUD_COMPLETE] timestamp={time.time():.4f} step={step} caller={caller or 'cloud_complete'} provider={self.config.get('cloud', {}).get('provider', 'groq')} mock={self.config.get('use_mock', True)} before={before} after={after} prompt_chars={len(prompt)} latency={elapsed:.4f}s")
                return cached["response"]

        if self.config.get("use_mock", True):
            before = self.usage.cloud_api_calls
            response = self._mock_response(prompt)
            p_tok = len(prompt.split())
            c_tok = len(response.split())
            t_tok = p_tok + c_tok
            self.usage.prompt_tokens += p_tok
            self.usage.completion_tokens += c_tok
            self.usage.total_tokens += t_tok
            self.usage.tokens = self.usage.total_tokens
            self.usage.cloud_network_calls += 1
            self.usage.cloud_api_calls += 1
            self.usage.api_calls = self.usage.cloud_api_calls
            self.usage.record_call_category(caller, getattr(self, "active_replan_reason", None))
            self.usage.successful_calls += 1
            after = self.usage.cloud_api_calls
            elapsed = time.perf_counter() - t_start
            self.usage.llm_wait_s += elapsed
            self.usage.cloud_inference_time_s += elapsed
            p_bytes = len(prompt.encode("utf-8")) + len(response.encode("utf-8"))
            self.usage.cloud_bytes += p_bytes
            print(f"[COUNTER] metric=cloud_planning_calls step={step} before={before} after={after} reason={caller or 'cloud_complete'} caller=CloudLLMClient.complete()")
            print(f"[CLOUD_COMPLETE] timestamp={time.time():.4f} step={step} caller={caller or 'cloud_complete'} provider={self.config.get('cloud', {}).get('provider', 'groq')} mock=True before={before} after={after} prompt_chars={len(prompt)} latency={elapsed:.4f}s")
            if cache_path:
                self._write_cache(cache_path, {"response": response, "tokens": t_tok, "prompt_tokens": p_tok, "completion_tokens": c_tok})
            return response

        before = self.usage.cloud_api_calls
        response, p_tok, c_tok, t_tok = self._call_with_retries(prompt, system)
        if response == _FAILURE_SENTINEL:
            self.usage.failed_calls += 1
            return response
        self.usage.prompt_tokens += p_tok
        self.usage.completion_tokens += c_tok
        self.usage.total_tokens += t_tok
        self.usage.tokens = self.usage.total_tokens
        self.usage.cloud_api_calls += 1
        self.usage.api_calls = self.usage.cloud_api_calls
        self.usage.record_call_category(caller, getattr(self, "active_replan_reason", None))
        self.usage.successful_calls += 1
        after = self.usage.cloud_api_calls
        elapsed = time.perf_counter() - t_start
        self.usage.llm_wait_s += elapsed
        self.usage.cloud_inference_time_s += elapsed
        p_bytes = len(prompt.encode("utf-8")) + len(response.encode("utf-8"))
        self.usage.cloud_bytes += p_bytes
        print(f"[COUNTER] metric=cloud_planning_calls step={step} before={before} after={after} reason={caller or 'cloud_complete'} caller=CloudLLMClient.complete()")
        print(f"[CLOUD_COMPLETE] timestamp={time.time():.4f} step={step} caller={caller or 'cloud_complete'} provider={self.config.get('cloud', {}).get('provider', 'groq')} mock=False before={before} after={after} prompt_chars={len(prompt)} latency={elapsed:.4f}s")
        if cache_path:
            self._write_cache(cache_path, {"response": response, "tokens": t_tok, "prompt_tokens": p_tok, "completion_tokens": c_tok})
        return response

    # ------------------------------------------------------------------
    # Retry logic — on exhaustion, terminate the experiment. No fallback.
    # ------------------------------------------------------------------
    def _classify_error(self, e: Exception) -> str:
        name = type(e).__name__
        low = f"{name} {e}".lower()
        if "timeout" in low:
            return "Timeout"
        if "connect" in low:
            return "ConnectionError"
        if "rate" in low and "limit" in low:
            return "RateLimit"
        if "status" in low or "http" in low:
            return "HTTPTransportError"
        if isinstance(e, json.JSONDecodeError):
            return "JSONParseError"
        return f"UnexpectedError({name})"

    def _call_with_retries(self, prompt: str, system: str) -> tuple[str, int, int, int]:
        provider = self.config.get("cloud", {}).get("provider", "unknown")
        model = self.config.get("cloud", {}).get("model", "unknown")
        last_err: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                self.usage.cloud_network_calls += 1
                res = self._api_call(prompt, system)
                if attempt > 1:
                    self.usage.retried_calls += 1
                return res
            except Exception as e:  # noqa: BLE001 — classified below, then re-raised as ExperimentFailed
                self.usage.cloud_failed_attempts += 1
                last_err = e
                reason = self._classify_error(e)
                if attempt < self.max_retries:
                    delay = self.backoff_base * (2 ** (attempt - 1))
                    _log(
                        f"Attempt {attempt}/{self.max_retries} "
                        f"Provider={provider} {reason} Retrying in {delay:g}s"
                    )
                    time.sleep(delay)
                else:
                    _log(f"Attempt {attempt}/{self.max_retries} Provider={provider} {reason} exhausted")

        # All retries exhausted — this experiment is scientifically invalid
        if last_err is not None:
            print(
                "[LLM DEBUG]\n"
                f"Exception Type: {type(last_err).__name__}\n"
                f"Exception Message: {last_err}"
            )
        report = FailureReport(
            experiment_status="DEGRADED",
            failure_reason=self._classify_error(last_err) if last_err else "Unknown",
            provider=provider,
            model=model,
            scenario=self.experiment_scenario,
            architecture=self.experiment_architecture,
            network_profile=self.experiment_network_profile,
            seed=self.experiment_seed,
            simulation_step=self.current_step,
            retry_count=self.max_retries,
            exception_type=type(last_err).__name__ if last_err else "Unknown",
        )
        report.log()
        report.persist()
        _log(
            "Retries exhausted -- degrading gracefully "
            "(cache -> previous plan -> device LLM). Simulation continues."
        )
        return _FAILURE_SENTINEL, 0, 0, 0

    # ------------------------------------------------------------------
    # Raw provider call — UNCHANGED signature/behavior
    # ------------------------------------------------------------------
    def _api_call(self, prompt: str, system: str) -> tuple[str, int, int, int]:
        cloud = self.config["cloud"]
        provider = cloud.get("provider", "openai")
        client = self._get_client()

        max_tok = (
            self.max_completion_tokens
            if getattr(self, "opt_constrained_output", True)
            else cloud.get("max_tokens", 1024)
        )
        if provider in ("openai", "groq", "nvidia"):
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            if provider == "groq":
                default_model = "llama-3.3-70b-versatile"
            elif provider == "nvidia":
                default_model = "meta/llama-3.1-8b-instruct"
            else:
                default_model = "gpt-4o"
            resp = client.chat.completions.create(
                model=cloud.get("model", default_model),
                messages=messages,
                max_tokens=max_tok,
                temperature=cloud.get("temperature", 0.2),
            )
            text = resp.choices[0].message.content or ""
            p_tok = resp.usage.prompt_tokens if resp.usage and hasattr(resp.usage, "prompt_tokens") else len(prompt.split())
            c_tok = resp.usage.completion_tokens if resp.usage and hasattr(resp.usage, "completion_tokens") else len(text.split())
            t_tok = resp.usage.total_tokens if resp.usage and hasattr(resp.usage, "total_tokens") else p_tok + c_tok
            return text, p_tok, c_tok, t_tok

        elif provider == "anthropic":
            resp = client.messages.create(
                model=cloud.get("model", "claude-sonnet-4-20250514"),
                max_tokens=max_tok,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text if resp.content else ""
            p_tok = resp.usage.input_tokens if resp.usage else len(prompt.split())
            c_tok = resp.usage.output_tokens if resp.usage else len(text.split())
            t_tok = p_tok + c_tok
            return text, p_tok, c_tok, t_tok

        raise ValueError(f"Unknown provider: {provider}")

    # ------------------------------------------------------------------
    # Mock helpers — used ONLY for explicit use_mock=true baseline runs
    # ------------------------------------------------------------------
    def _mock_response(self, prompt: str) -> str:
        pl = prompt.lower()
        if "coalition" in pl:
            return self._mock_coalition(prompt)
        if "decompose" in pl or "task decomposer" in pl or "assign" in pl or "subtask" in pl:
            return self._mock_decomposition(prompt)
        return json.dumps({"status": "ok"})

    def _agent_id(self, agent: dict) -> str:
        return str(agent.get("id", agent.get("agent_id", "")))

    def _mock_assignments_from_inputs(
        self, agents: list[dict], subtasks: list[dict]
    ) -> dict[str, list[str]]:
        assignments: dict[str, list[str]] = {}
        if not agents:
            return assignments
        for i, st in enumerate(subtasks):
            st_id = str(st.get("id", st.get("subtask_id", f"T_{i}")))
            assignments[st_id] = [self._agent_id(agents[i % len(agents)])]
        return assignments

    def _mock_coalitions_from_inputs(self, agents: list[dict]) -> list[dict]:
        coalitions: list[dict] = []
        for i in range(0, len(agents), 2):
            group = [self._agent_id(a) for a in agents[i : i + 2] if self._agent_id(a)]
            if group:
                coalitions.append({"coalition_id": len(coalitions), "members": group})
        return coalitions

    def _extract_labeled_json(self, prompt: str, label: str) -> Any | None:
        low_prompt = prompt.lower()
        low_label = label.lower() + ":"
        idx = low_prompt.find(low_label)
        if idx < 0:
            return None
        rest = prompt[idx + len(low_label) :].lstrip()
        if not rest:
            return None
        opener = rest[0]
        if opener not in "[{":
            return None
        closer = "]" if opener == "[" else "}"
        depth = 0
        in_string = False
        escape = False
        for pos, ch in enumerate(rest):
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(rest[: pos + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    def _mock_decomposition(self, prompt: str) -> str:
        agents = self._extract_labeled_json(prompt, "Agents (with positions and skills)")
        if agents is None:
            agents = self._extract_labeled_json(prompt, "Agents")
        subtasks = self._extract_labeled_json(prompt, "Subtasks (with targets and required skills)")
        if subtasks is None:
            subtasks = self._extract_labeled_json(prompt, "Subtasks")
        if agents is not None and subtasks is not None:
            return json.dumps({"assignments": self._mock_assignments_from_inputs(agents, subtasks)})
        try:
            start = prompt.index("{")
            ctx = json.loads(
                prompt[start:].split("\n")[0] if "\n" in prompt[start:] else prompt[start:]
            )
            subtasks = ctx.get("subtasks", [])
            agents = ctx.get("agents", [])
            return json.dumps({"assignments": self._mock_assignments_from_inputs(agents, subtasks)})
        except (ValueError, json.JSONDecodeError):
            return json.dumps({"assignments": {}})

    def _mock_coalition(self, prompt: str) -> str:
        agents = self._extract_labeled_json(prompt, "Agents (with positions and skills)")
        if agents is None:
            agents = self._extract_labeled_json(prompt, "Agents")
        if agents is not None:
            return json.dumps({"coalitions": self._mock_coalitions_from_inputs(agents)})
        try:
            start = prompt.index("{")
            end = prompt.rindex("}") + 1
            ctx = json.loads(prompt[start:end])
            agents = ctx.get("agents", [])
            return json.dumps({"coalitions": self._mock_coalitions_from_inputs(agents)})
        except (ValueError, json.JSONDecodeError):
            return json.dumps({"coalitions": []})

    # ------------------------------------------------------------------
    # Public planning API — UNCHANGED
    # ------------------------------------------------------------------
    def decompose(
        self,
        instruction: str,
        agents: list[dict],
        subtasks: list[dict],
        distance_matrix: list[list[float]] | None = None,
        execution_feedback_context: str | None = None,
    ) -> dict[str, list[str]]:
        from src.config import get_thresholds
        from src.llm.prompts import format_prompt

        th = get_thresholds()

        # Optimization 1 & 4: State Summarization & Semantic Cache Lookup
        summary_res = self.summarizer.summarize_decomposition_context(
            instruction, agents, subtasks, distance_matrix
        )
        self.prompt_reduction_percent = summary_res.prompt_reduction_percent

        cached_plan = self.semantic_cache.lookup(
            summary_res.summary_dict, getattr(self, "current_step", 0)
        )
        if cached_plan is not None:
            self._last_assignments = cached_plan
            return cached_plan

        # Optimization 2: Compressed Prompting
        if getattr(self, "opt_prompt_compression", True):
            prompt = (
                f"Task: {instruction}\n"
                f"Agents: {json.dumps(summary_res.summary_dict.get('active_agents', []))}\n"
                f"Subtasks: {json.dumps(summary_res.summary_dict.get('subtasks', []))}\n"
                'Assign agents to subtasks. Return JSON ONLY: {"assignments": {"T_0": ["uav_1"]}}'
            )
        else:
            try:
                prompt = format_prompt(
                    "decomposition",
                    instruction=instruction,
                    agents=json.dumps(agents),
                    subtasks=json.dumps(subtasks),
                    distance_matrix=json.dumps(distance_matrix),
                    c_task=str(th.get("C_task", 30.0)),
                    r_reach=str(th.get("R_reach", 100.0)),
                )
            except (FileNotFoundError, KeyError):
                prompt = (
                    "Decompose the mission into subtask assignments.\n"
                    f"Instruction: {instruction}\n"
                    f"Context: {json.dumps({'agents': agents, 'subtasks': subtasks, 'distance_matrix': distance_matrix})}\n"
                    'Return JSON: {"assignments": {"T_0": ["agent_ids"], ...}}'
                )

        # AutoHMA self-correction: inject prior execution feedback into existing prompt
        if execution_feedback_context:
            prompt = prompt + "\n" + execution_feedback_context

        raw = self.complete(prompt, system="You are a Cloud LLM task decomposer.", caller="decompose")
        if raw == _FAILURE_SENTINEL:
            _log("decompose() degraded: cloud LLM unavailable")
            if self._last_assignments:
                _log("Falling back to previous valid decomposition plan")
                return self._last_assignments
            if self.device_fallback_decompose is not None:
                try:
                    _log("Falling back to local Device LLM for decomposition")
                    fallback = self.device_fallback_decompose(instruction, agents, subtasks)
                    if fallback:
                        return fallback
                except Exception as e:  # noqa: BLE001
                    _log(f"Device LLM fallback failed: {e}")
            _log("No fallback available -- returning empty assignment, simulation continues")
            return {}
        result = self._parse_assignments_response(raw)
        if result:
            self._last_assignments = result
            self.semantic_cache.put(
                summary_res.summary_dict,
                result,
                tokens=200,
                latency=0.05,
                current_step=getattr(self, "current_step", 0),
            )
        return result

    def form_coalitions(
        self,
        subtasks: list[dict],
        agents: list[dict],
        distance_matrix: list[list[float]] | None = None,
        cqi_matrix: list[list[float]] | None = None,
        execution_feedback_context: str | None = None,
    ) -> list[dict]:
        from src.config import get_thresholds
        from src.llm.prompts import format_prompt

        th = get_thresholds()

        # Optimization 1 & 4: State Summarization & Semantic Cache Lookup
        summary_res = self.summarizer.summarize_coalition_context(
            subtasks, agents, distance_matrix or [], cqi_matrix or []
        )
        self.prompt_reduction_percent = summary_res.prompt_reduction_percent

        cached_coalitions = self.semantic_cache.lookup(
            summary_res.summary_dict, getattr(self, "current_step", 0)
        )
        if cached_coalitions is not None:
            self._last_coalitions = cached_coalitions
            return cached_coalitions

        # Optimization 2: Compressed Prompting
        if getattr(self, "opt_prompt_compression", True):
            prompt = (
                f"Subtasks: {json.dumps(summary_res.summary_dict.get('subtasks', []))}\n"
                f"Agents: {json.dumps(summary_res.summary_dict.get('agents', []))}\n"
                'Group agents into coalitions. Return JSON ONLY: {"coalitions": [{"coalition_id": 0, "members": ["uav_1", "uav_2"]}]}'
            )
        else:
            try:
                prompt = format_prompt(
                    "coalition",
                    subtasks=json.dumps(subtasks),
                    agents=json.dumps(agents),
                    distance_matrix=json.dumps(distance_matrix),
                    cqi_matrix=json.dumps(cqi_matrix),
                    c1=str(th.get("C1", 50.0)),
                    gamma_min=str(th.get("gamma_min", 0.3)),
                )
            except (FileNotFoundError, KeyError):
                prompt = (
                    "Form agent coalitions for subtask execution.\n"
                    f"Context: {json.dumps({'subtasks': subtasks, 'agents': agents, 'D': distance_matrix, 'Q': cqi_matrix})}\n"
                    'Return JSON: {"coalitions": [{"coalition_id": 0, "members": ["id1"]}]}'
                )

        # AutoHMA self-correction: inject prior execution feedback into existing prompt
        if execution_feedback_context:
            prompt = prompt + "\n" + execution_feedback_context

        raw = self.complete(prompt, system="You are a Cloud LLM coalition planner.", caller="form_coalitions")
        if raw == _FAILURE_SENTINEL:
            _log("form_coalitions() degraded: cloud LLM unavailable")
            if self._last_coalitions:
                _log("Falling back to previous valid coalitions")
                return self._last_coalitions
            if self.device_fallback_coalitions is not None:
                try:
                    _log("Falling back to local Device LLM for coalitions")
                    fallback = self.device_fallback_coalitions(
                        subtasks, agents, distance_matrix, cqi_matrix
                    )
                    if fallback:
                        return fallback
                except Exception as e:  # noqa: BLE001
                    _log(f"Device LLM fallback failed: {e}")
            _log("No fallback available -- returning empty coalitions, simulation continues")
            return []

        # Dedicated raw log output for Task 4
        try:
            debug_log_path = Path("logs/llm_coalitions_debug.log")
            debug_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_log_path, "a", encoding="utf-8") as f_debug:
                f_debug.write(
                    f"=== STEP {self.current_step} | SCENARIO {self.experiment_scenario} | "
                    f"RAW LEN {len(raw)} | PROMPT CHARS {len(prompt)} ===\n{raw}\n\n"
                )
        except Exception:
            pass

        valid_agent_ids = {
            str(a.get("id", a.get("agent_id")))
            for a in agents
            if a.get("id") or a.get("agent_id")
        }

        result, fallback_used, stripped_map = self._parse_coalitions_response(raw, agents)
        if fallback_used:
            self.hallucination_stats["form_coalitions_failure_count"] = self.hallucination_stats.get("form_coalitions_failure_count", 0) + 1
            tot = self.hallucination_stats.get("form_coalitions_success_count", 0) + self.hallucination_stats["form_coalitions_failure_count"]
            fail_rate = self.hallucination_stats["form_coalitions_failure_count"] / tot
            if fail_rate > 0.20 and tot >= 5:
                _log(f"[WARNING] form_coalitions failure rate ({fail_rate:.1%}) exceeds 20% threshold! ({self.hallucination_stats['form_coalitions_failure_count']}/{tot})")
        else:
            self.hallucination_stats["form_coalitions_success_count"] = self.hallucination_stats.get("form_coalitions_success_count", 0) + 1

        # --- ADVANCED RECOVERY PIPELINE (TASK 2) ---
        if stripped_map:
            all_stripped = [s for sub in stripped_map.values() for s in sub]
            _log(
                f"[HALLUCINATION DETECTED] Step={self.current_step} "
                f"Scenario={self.experiment_scenario} "
                f"StrippedIDs={all_stripped}"
            )
            self.hallucination_stats["total_events"] += 1
            self.hallucination_stats["events_log"].append({
                "step": self.current_step,
                "scenario": self.experiment_scenario,
                "stripped_ids": all_stripped,
            })

            # Strategy 1: Corrective Re-query to Cloud LLM (capped at 1 retry)
            requery_prompt = (
                "[CORRECTIVE RE-QUERY - PREVIOUS RESPONSE CONTAINED INVALID AGENT IDS]\n"
                f"Your previous coalition response contained invalid agent IDs: {all_stripped}.\n"
                f"THE ONLY VALID FLEET AGENT ROSTER IS: {sorted(list(valid_agent_ids))}.\n"
                "Re-form the coalitions using ONLY valid agent IDs from the roster above.\n"
                'Return JSON: {"coalitions": [{"coalition_id": 0, "members": ["id1"]}]}'
            )
            self.hallucination_stats["retry_attempts"] += 1
            tokens_before = self.usage.tokens
            calls_before = self.usage.api_calls
            raw_retry = self.complete(requery_prompt, system="You are a Cloud LLM coalition planner.", caller="form_coalitions_retry")
            tokens_added = self.usage.tokens - tokens_before
            calls_added = self.usage.api_calls - calls_before
            self.hallucination_stats["recovery_tokens"] = self.hallucination_stats.get("recovery_tokens", 0) + tokens_added
            self.hallucination_stats["recovery_api_calls"] = self.hallucination_stats.get("recovery_api_calls", 0) + calls_added
            total_t = self.usage.tokens
            self.hallucination_stats["recovery_overhead_pct"] = round((self.hallucination_stats["recovery_tokens"] / total_t * 100), 2) if total_t > 0 else 0.0

            if raw_retry != _FAILURE_SENTINEL:
                result_retry, _, stripped_retry = self._parse_coalitions_response(raw_retry, agents)
                if result_retry and not stripped_retry:
                    _log("[RECOVERED VIA RE-QUERY] Corrective re-query succeeded with 0 stripped IDs.")
                    self.hallucination_stats["retry_successes"] += 1
                    result = result_retry
                    stripped_map = {}

        # Strategy 2: Intelligent Role Substitution for remaining under-staffed coalitions
        if stripped_map:
            assigned_members = {m for c in result for m in c.get("members", [])}
            unassigned_agents = [
                a for a in agents
                if str(a.get("id", a.get("agent_id"))) not in assigned_members
            ]
            for cid, stripped_list in stripped_map.items():
                target_coalition = next((c for c in result if c.get("coalition_id") == cid), None)
                if target_coalition is None:
                    target_coalition = {"coalition_id": cid, "members": []}
                    result.append(target_coalition)
                for stripped_id in stripped_list:
                    target_type = stripped_id.split("_")[0].lower() if "_" in stripped_id else ""
                    sub_idx = -1
                    for idx, un_a in enumerate(unassigned_agents):
                        un_id = str(un_a.get("id", un_a.get("agent_id")))
                        un_type = str(un_a.get("type", un_a.get("agent_type", ""))).lower()
                        if target_type in un_type or target_type in un_id:
                            sub_idx = idx
                            break
                    if sub_idx < 0 and unassigned_agents:
                        sub_idx = 0
                    if sub_idx >= 0:
                        sub_agent = unassigned_agents.pop(sub_idx)
                        sub_id = str(sub_agent.get("id", sub_agent.get("agent_id")))
                        target_coalition["members"].append(sub_id)
                        assigned_members.add(sub_id)
                        self.hallucination_stats["substitutions"] += 1
                        _log(
                            f"[ROLE SUBSTITUTION] Step={self.current_step} "
                            f"Coalition={cid}: Substituted missing '{stripped_id}' with idle agent '{sub_id}'"
                        )
                    else:
                        self.hallucination_stats["partial_strips"] += 1

        if not result:
            _log("form_coalitions: no valid coalitions remain, using per-agent singletons")
            self.hallucination_stats["singleton_fallbacks"] += 1
            result = self._singleton_coalitions(agents)

        if result:
            self._last_coalitions = result
        return result

    def _parse_json(self, text: str) -> dict:
        """Extract and robustly parse JSON from text, auto-repairing truncated JSON endings."""
        if not text:
            return {}
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            pass

        if "```" in text:
            import re
            matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            for m in matches:
                try:
                    start = m.index("{")
                    end = m.rindex("}") + 1
                    return json.loads(m[start:end])
                except (ValueError, json.JSONDecodeError):
                    pass

        try:
            start = text.index("{")
            sub = text[start:].strip()
            stack = []
            in_string = False
            escape = False
            repaired_chars = []
            for ch in sub:
                repaired_chars.append(ch)
                if escape:
                    escape = False
                    continue
                if ch == '\\':
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if not in_string:
                    if ch in "{[":
                        stack.append("}" if ch == "{" else "]")
                    elif ch in "}]":
                        if stack and stack[-1] == ch:
                            stack.pop()

            if in_string:
                repaired_chars.append('"')
            while stack:
                repaired_chars.append(stack.pop())

            repaired_text = "".join(repaired_chars)
            return json.loads(repaired_text)
        except (ValueError, json.JSONDecodeError):
            pass

        return {}

    def _parse_json_list(self, text: str) -> list | None:
        """Try to extract a bare JSON list from LLM output."""
        try:
            start = text.index("[")
            end = text.rindex("]") + 1
            result = json.loads(text[start:end])
            if isinstance(result, list):
                return result
        except (ValueError, json.JSONDecodeError):
            pass
        return None

    _COALITION_KEYS = ("coalitions", "groups", "teams", "clusters", "alliances")
    _ASSIGNMENT_KEYS = ("assignments", "task_assignments", "decomposition", "allocation")

    def _parse_coalitions_response(
        self, raw: str, agents: list[dict],
    ) -> tuple[list[dict], bool, dict[int, list[str]]]:
        """Robustly extract coalitions from a real LLM response, returning (coalitions, fallback_used, stripped_map)."""
        parsed = self._parse_json(raw)
        valid_agent_ids = {
            str(a.get("id", a.get("agent_id")))
            for a in agents
            if a.get("id") or a.get("agent_id")
        }
        for key in self._COALITION_KEYS:
            val = parsed.get(key)
            if isinstance(val, list) and val:
                norm, stripped = self._normalize_coalitions(val, valid_agent_ids)
                if norm or stripped:
                    return norm, False, stripped
        for val in parsed.values():
            if isinstance(val, list) and val and isinstance(val[0], dict):
                norm, stripped = self._normalize_coalitions(val, valid_agent_ids)
                if norm or stripped:
                    return norm, False, stripped
        bare = self._parse_json_list(raw)
        if bare and isinstance(bare[0], dict):
            norm, stripped = self._normalize_coalitions(bare, valid_agent_ids)
            if norm or stripped:
                return norm, False, stripped
        _log("form_coalitions: could not extract valid coalitions from LLM "
             f"response (len={len(raw)}), generating per-agent singletons")
        return self._singleton_coalitions(agents), True, {}

    def _normalize_coalitions(
        self, raw_list: list[dict], valid_agent_ids: set[str] | None = None
    ) -> tuple[list[dict], dict[int, list[str]]]:
        """Ensure every coalition dict has 'coalition_id' and 'members', returning (clean_coalitions, stripped_map)."""
        result = []
        stripped_map = {}
        for i, c in enumerate(raw_list):
            cid = c.get("coalition_id", c.get("id", i))
            members = c.get("members", c.get("agents", c.get("agent_ids", [])))
            if isinstance(members, str):
                members = [members]
            if isinstance(members, list):
                clean_members = []
                stripped_members = []
                for m in members:
                    m_str = str(m)
                    if valid_agent_ids is not None and m_str not in valid_agent_ids:
                        _log(f"[PARSER WARNING] Stripped invalid/hallucinated agent ID '{m_str}' from coalition {cid}")
                        stripped_members.append(m_str)
                        continue
                    clean_members.append(m_str)
                if stripped_members:
                    stripped_map[int(cid)] = stripped_members
                if clean_members:
                    result.append({"coalition_id": int(cid), "members": clean_members})
        return result, stripped_map

    def _singleton_coalitions(self, agents: list[dict]) -> list[dict]:
        """One coalition per agent — minimal structure for distributed planning."""
        coalitions = []
        for i, a in enumerate(agents):
            aid = self._agent_id(a)
            if aid:
                coalitions.append({"coalition_id": i, "members": [aid]})
        return coalitions

    def _parse_assignments_response(self, raw: str) -> dict[str, list[str]]:
        """Robustly extract assignments from a real LLM response."""
        parsed = self._parse_json(raw)
        for key in self._ASSIGNMENT_KEYS:
            val = parsed.get(key)
            if isinstance(val, dict) and val:
                return self._normalize_assignments(val)
        # Any remaining dict-valued key whose values look like agent lists
        for val in parsed.values():
            if isinstance(val, dict) and val:
                first_v = next(iter(val.values()), None)
                if isinstance(first_v, (list, str)):
                    return self._normalize_assignments(val)
        return {}

    @staticmethod
    def _normalize_assignments(raw_map: dict) -> dict[str, list[str]]:
        """Ensure every assignment value is a list of agent-id strings."""
        result: dict[str, list[str]] = {}
        for sid, agents in raw_map.items():
            if isinstance(agents, str):
                agents = [agents]
            if isinstance(agents, list):
                result[str(sid)] = [str(a) for a in agents]
        return result