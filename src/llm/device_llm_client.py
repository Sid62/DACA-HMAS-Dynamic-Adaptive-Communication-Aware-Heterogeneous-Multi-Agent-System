"""Device LLM client for local dispatch and distributed domain-level reasoning."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import time

import psutil
import tracemalloc
import numpy as np

import httpx

from src.communication.models import NodeState, PeerMessage, SharedPlan
from src.config import get_llm_config, project_root


# PAPER INTERPRETATION NOTE (memory_mb):
# The AutoHMA-LLM base paper's "Memory Usage per Device LLM" is measured here as the
# local orchestrator/controller process's own memory footprint (via
# psutil.Process(os.getpid()).memory_info().rss), NOT the memory required to host or
# run the underlying LLM's own weights/inference engine. This interpretation is based
# on: (1) the paper's reported magnitude (40-70 MB) being 2-3 orders of magnitude too
# small to represent hosting memory for either model the paper uses in this role
# (GPT-4, API-only and not independently inspectable; Llama2-70B, which alone requires
# ~140GB in fp16), and (2) the paper's explicit framing of the Device LLM as a
# lightweight "dispatcher" role, distinct from the "heavy reasoning" attributed to the
# Cloud LLM. This is a documented, evidence-based assumption, not a fact confirmed by
# the paper, since the paper never explicitly states what the metric measures at an
# implementation level.


@dataclass
class DeviceLLMUsage:
    tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    retry_tokens: int = 0
    api_calls: int = 0
    device_api_calls: int = 0
    device_inference_calls: int = 0
    logical_requests: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    retried_calls: int = 0
    cache_hits: int = 0
    memory_mb: float = 0.0
    python_heap_delta_mb: float = 0.0
    python_heap_delta_by_device: dict[str, float] = field(default_factory=dict)
    tokens_processed_by_device: dict[str, int] = field(default_factory=dict)
    llm_wait_s: float = 0.0
    device_inference_time_s: float = 0.0
    rss_delta_mb_samples: list[float] = field(default_factory=list)
    heap_delta_mb_samples: list[float] = field(default_factory=list)
    device_llm_memory_mb: dict[str, float] = field(default_factory=dict)
    device_llm_memory_peak_mb: dict[str, float] = field(default_factory=dict)
    device_llm_heap_delta_mb: dict[str, float] = field(default_factory=dict)

    def reset(self) -> None:
        self.tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.retry_tokens = 0
        self.api_calls = 0
        self.device_api_calls = 0
        self.device_inference_calls = 0
        self.logical_requests = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.retried_calls = 0
        self.cache_hits = 0
        self.memory_mb = 0.0
        self.python_heap_delta_mb = 0.0
        self.python_heap_delta_by_device = {}
        self.tokens_processed_by_device = {}
        self.llm_wait_s = 0.0
        self.device_inference_time_s = 0.0
        self.rss_delta_mb_samples = []
        self.heap_delta_mb_samples = []
        self.device_llm_memory_mb = {}
        self.device_llm_memory_peak_mb = {}
        self.device_llm_heap_delta_mb = {}


@dataclass
class DeviceLLMClient:
    """
    One Device LLM per agent-type domain (e.g. uav, vehicle, robot).

    Manages all robots of that type via ``managed_agent_ids``. Performs LLM
    reasoning only — no networking. Legacy per-agent instances (node_id = agent
    id) remain supported until orchestrator migration completes.
    """

    config: dict[str, Any] = field(default_factory=get_llm_config)
    usage: DeviceLLMUsage = field(default_factory=DeviceLLMUsage)
    node_id: str = "device_0"
    managed_agent_ids: list[str] = field(default_factory=list)
    node_state: NodeState | None = None

    def __post_init__(self) -> None:
        if not tracemalloc.is_tracing():
            tracemalloc.start()

    @property
    def domain_id(self) -> str:
        return self.node_id

    def __post_init__(self) -> None:
        if self.node_state is None:
            managed = list(self.managed_agent_ids) or [self.node_id]
            self.node_state = NodeState(node_id=self.node_id, managed_agent_ids=managed)
        else:
            if self.node_state.node_id != self.node_id:
                self.node_state.node_id = self.node_id
            if self.managed_agent_ids:
                self.node_state.managed_agent_ids = list(self.managed_agent_ids)
            elif self.node_state.managed_agent_ids:
                self.managed_agent_ids = list(self.node_state.managed_agent_ids)
            else:
                self.managed_agent_ids = [self.node_id]
                self.node_state.managed_agent_ids = [self.node_id]

    @classmethod
    def for_domain(
        cls,
        domain_id: str,
        managed_agent_ids: list[str],
        config: dict[str, Any] | None = None,
    ) -> DeviceLLMClient:
        """Factory for a domain-scoped Device LLM instance."""
        cfg = config if config is not None else get_llm_config()
        return cls(
            config=cfg,
            node_id=domain_id,
            managed_agent_ids=list(managed_agent_ids),
        )

    def _cache_path(self, prompt: str) -> Path | None:
        if not self.config.get("cache_responses", True):
            return None
        cache_dir = project_root() / self.config.get("cache_dir", ".llm_cache")
        cache_dir.mkdir(exist_ok=True)
        key = hashlib.sha256(f"{self.node_id}:{prompt}".encode()).hexdigest()[:16]
        return cache_dir / f"device_{key}.json"

    current_step: int = 0

    def set_step(self, step: int) -> None:
        self.current_step = step

    def complete(self, prompt: str, caller: str = "device_complete", coalition_id: int = 0) -> str:
        self.usage.logical_requests += 1
        proc = psutil.Process(os.getpid())
        rss_before = proc.memory_info().rss
        snapshot_before = tracemalloc.take_snapshot() if tracemalloc.is_tracing() else None

        t_start = time.perf_counter()
        step = getattr(self, "current_step", 0)
        cache_path = self._cache_path(prompt)
        cache_hit = False
        if cache_path and cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
                before = self.usage.device_api_calls
                # Cache hit: DO NOT add inference tokens or increment actual device API calls!
                self.usage.cache_hits += 1
                self.usage.successful_calls += 1
                after = self.usage.device_api_calls
                cache_hit = True
                elapsed = time.perf_counter() - t_start
                self.usage.llm_wait_s += elapsed
                self.usage.device_inference_time_s += elapsed
                print(f"[COUNTER] metric=device_planning_calls step={step} before={before} after={after} reason={caller} caller=DeviceLLMClient.complete()")
                print(f"[DEVICE_COMPLETE] timestamp={time.time():.4f} step={step} caller={caller} domain={self.node_id} coalition_id={coalition_id} cache=HIT latency={elapsed:.4f}s")
                return data["response"]

        if self.config.get("use_mock", True):
            before = self.usage.device_api_calls
            response = self._mock_response(prompt)
            p_tok = len(prompt.split())
            c_tok = len(response.split())
            t_tok = p_tok + c_tok
            self.usage.prompt_tokens += p_tok
            self.usage.completion_tokens += c_tok
            self.usage.total_tokens += t_tok
            self.usage.tokens = self.usage.total_tokens
            self.usage.device_inference_calls += 1
            self.usage.device_api_calls += 1
            self.usage.api_calls = self.usage.device_api_calls
            self.usage.successful_calls += 1

            elapsed = time.perf_counter() - t_start
            self.usage.llm_wait_s += elapsed
            self.usage.device_inference_time_s += elapsed
            after = before + 1

            rss_after = proc.memory_info().rss
            snapshot_after = tracemalloc.take_snapshot() if tracemalloc.is_tracing() else None

            rss_delta = max(0.0, (rss_after - rss_before) / (1024 * 1024))
            self.usage.rss_delta_mb_samples.append(rss_delta)

            if snapshot_before and snapshot_after:
                stats = snapshot_after.compare_to(snapshot_before, "lineno")
                heap_delta_bytes = sum(stat.size_diff for stat in stats if stat.size_diff > 0)
                heap_delta_mb = heap_delta_bytes / (1024 * 1024)
            else:
                heap_delta_mb = 0.0
            self.usage.heap_delta_mb_samples.append(heap_delta_mb)

            # Whole-process RSS at time of domain call (system-level, kept for backward compatibility)
            self.usage.memory_mb = rss_after / (1024 * 1024)
            self.usage.python_heap_delta_mb = max(self.usage.python_heap_delta_mb, heap_delta_mb)

            if cache_path:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump({"response": response, "tokens": t_tok, "prompt_tokens": p_tok, "completion_tokens": c_tok}, f)
            return response
        else:
            before = self.usage.device_api_calls
            provider = self.config.get("device", {}).get("provider", "ollama")
            start = time.perf_counter()
            if provider == "vllm":
                response, p_tok, c_tok, t_tok = self._vllm_call(prompt)
            else:
                response, p_tok, c_tok, t_tok = self._ollama_call(prompt)
            elapsed = time.perf_counter() - start
            self.usage.prompt_tokens += p_tok
            self.usage.completion_tokens += c_tok
            self.usage.total_tokens += t_tok
            self.usage.tokens = self.usage.total_tokens
            self.usage.device_inference_calls += 1
            self.usage.device_api_calls += 1
            self.usage.api_calls = self.usage.device_api_calls
            self.usage.successful_calls += 1

            self.usage.llm_wait_s += elapsed
            self.usage.device_inference_time_s += elapsed
            after = before + 1

            rss_after = proc.memory_info().rss
            snapshot_after = tracemalloc.take_snapshot() if tracemalloc.is_tracing() else None

            rss_delta = max(0.0, (rss_after - rss_before) / (1024 * 1024))
            self.usage.rss_delta_mb_samples.append(rss_delta)

            if snapshot_before and snapshot_after:
                stats = snapshot_after.compare_to(snapshot_before, "lineno")
                heap_delta_bytes = sum(stat.size_diff for stat in stats if stat.size_diff > 0)
                heap_delta_mb = heap_delta_bytes / (1024 * 1024)
            else:
                heap_delta_mb = 0.0
            self.usage.heap_delta_mb_samples.append(heap_delta_mb)

            # Whole-process RSS at time of domain call (system-level, kept for backward compatibility)
            self.usage.memory_mb = rss_after / (1024 * 1024)
            self.usage.python_heap_delta_mb = max(self.usage.python_heap_delta_mb, heap_delta_mb)

            print(f"[COUNTER] metric=device_planning_calls step={step} before={before} after={after} reason={caller} caller=DeviceLLMClient.complete()")
            print(f"[DEVICE_COMPLETE] timestamp={time.time():.4f} step={step} caller={caller} domain={self.node_id} coalition_id={coalition_id} cache=MISS latency={elapsed:.4f}s")
            if cache_path:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump({"response": response, "tokens": t_tok, "prompt_tokens": p_tok, "completion_tokens": c_tok}, f)
            return response

    def _ollama_call(self, prompt: str) -> tuple[str, int, int, int]:
        device = self.config["device"]
        base_url = device.get("base_url", "http://localhost:11434")
        model = device.get("model", "llama3.1:8b")
        timeout_s = 420.0

        prompt_chars = len(prompt)
        approx_tokens = prompt_chars // 4
        print(f"[OLLAMA-REQUEST] domain={self.node_id} model={model} "
              f"url={base_url}/api/generate timeout={timeout_s}s "
              f"prompt_chars={prompt_chars} approx_tokens={approx_tokens}")

        t_start = time.perf_counter()
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": device.get("temperature", 0.1),
                        "num_gpu": device.get("num_gpu", -1),
                        "num_predict": device.get("max_tokens", 200),
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response", "")
            p_tok = data.get("prompt_eval_count", len(prompt.split()))
            c_tok = data.get("eval_count", len(text.split()))
            t_tok = p_tok + c_tok
        t_elapsed = time.perf_counter() - t_start

        print(f"[OLLAMA-RESPONSE] domain={self.node_id} elapsed={t_elapsed:.2f}s "
              f"response_chars={len(text)} response_tokens={c_tok} "
              f"prompt_eval_count={p_tok} "
              f"eval_duration_ns={data.get('eval_duration')} "
              f"prompt_eval_duration_ns={data.get('prompt_eval_duration')} "
              f"load_duration_ns={data.get('load_duration')}")
        return text, p_tok, c_tok, t_tok

    def _vllm_call(self, prompt: str) -> tuple[str, int, int, int]:
        """OpenAI-compatible vLLM server endpoint."""
        device = self.config["device"]
        base_url = device.get("base_url", "http://localhost:8000/v1")
        model = device.get("model", "meta-llama/Llama-3.1-8B-Instruct")
        with httpx.Client(timeout=520.0) as client:
            resp = client.post(
                f"{base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": device.get("max_tokens", 200),
                    "temperature": device.get("temperature", 0.1),
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            p_tok = usage.get("prompt_tokens", len(prompt.split()))
            c_tok = usage.get("completion_tokens", len(text.split()))
            t_tok = usage.get("total_tokens", p_tok + c_tok)
            return text, p_tok, c_tok, t_tok

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

    def _mock_response(self, prompt: str) -> str:
        pl = prompt.lower()
        managed = self.managed_agent_ids or [self.node_id]
        if "decompose" in pl or "task decomposer" in pl:
            agents = self._extract_labeled_json(prompt, "Agents")
            subtasks = self._extract_labeled_json(prompt, "Subtasks")
            if isinstance(agents, list) and isinstance(subtasks, list):
                return json.dumps({"assignments": self._mock_assignments_from_inputs(agents, subtasks)})
        if "coalition" in pl:
            agents = self._extract_labeled_json(prompt, "Agents")
            if isinstance(agents, list):
                return json.dumps({"coalitions": self._mock_coalitions_from_inputs(agents)})
        if "dispatch" in pl:
            coalitions_data = self._extract_labeled_json(prompt, "Coalitions")
            assignments = {}
            if isinstance(coalitions_data, list):
                for i, c in enumerate(coalitions_data):
                    members = c.get("members", [])
                    cid = c.get("coalition_id", c.get("id", i))
                    for m in members:
                        if m in managed:
                            assignments[m] = f"T_{cid}"
            return json.dumps({"dispatched": True, "domain": self.node_id, "assignments": assignments})
        if "plan_local" in pl or "plan locally" in pl:
            assignments = {aid: f"T_{i}" for i, aid in enumerate(managed)}
            return json.dumps({
                "action": "plan_local",
                "domain": self.node_id,
                "assignments": assignments,
                "plan_version": 1,
            })
        if "review" in pl and "peer" in pl:
            return json.dumps({"approved": True, "revision": {}, "comments": "ok"})
        if "merge" in pl and "peer" in pl:
            return json.dumps({
                "merged_plan": {"assignments": {}, "subtasks": []},
                "version": 2,
            })
        if "respond" in pl and "peer" in pl:
            return json.dumps({
                "response_type": "ack",
                "payload": {},
                "local_decision": {},
            })
        if "coordinate" in pl or "realloc" in pl:
            return json.dumps({"action": "reallocate", "status": "ok"})
        return json.dumps({"status": "ack"})

    def decompose(
        self,
        instruction: str,
        agents: list[dict],
        subtasks: list[dict],
        distance_matrix: list[list[float]] | None = None,
    ) -> dict[str, list[str]]:
        """Device-level task decomposition for decentralized mode."""
        prompt = (
            f"Decompose subtask assignments locally for domain {self.node_id}.\n"
            f"Instruction: {instruction}\n"
            f"Agents: {json.dumps(agents)}\n"
            f"Subtasks: {json.dumps(subtasks)}\n"
            f"Distance matrix: {json.dumps(distance_matrix or [])}\n"
            'Return JSON ONLY: {"assignments": {"T_0": ["uav_1"], ...}}'
        )
        raw = self.complete(prompt, caller="device_decompose")
        parsed = self._parse_json_response(raw)
        if isinstance(parsed, dict) and "assignments" in parsed and isinstance(parsed["assignments"], dict):
            return parsed["assignments"]
        if isinstance(parsed, dict) and parsed:
            result = {}
            for k, v in parsed.items():
                if isinstance(k, str):
                    result[k] = v if isinstance(v, list) else [str(v)]
            if result:
                return result
        return self._mock_assignments_from_inputs(agents, subtasks)

    def form_coalitions(
        self,
        subtasks: list[dict],
        agents: list[dict],
        distance_matrix: list[list[float]] | None = None,
        cqi_matrix: list[list[float]] | None = None,
    ) -> list[dict]:
        """Device-level coalition formation for decentralized mode."""
        prompt = (
            f"Form agent coalitions locally for domain {self.node_id}.\n"
            f"Subtasks: {json.dumps(subtasks)}\n"
            f"Agents: {json.dumps(agents)}\n"
            f"Distance matrix: {json.dumps(distance_matrix or [])}\n"
            f"CQI matrix: {json.dumps(cqi_matrix or [])}\n"
            'Return JSON ONLY: {"coalitions": [{"coalition_id": 0, "members": ["uav_1"]}, ...]}'
        )
        raw = self.complete(prompt, caller="device_form_coalitions")
        parsed = self._parse_json_response(raw)
        if isinstance(parsed, dict) and "coalitions" in parsed and isinstance(parsed["coalitions"], list):
            return parsed["coalitions"]
        if isinstance(parsed, list):
            return parsed
        return self._mock_coalitions_from_inputs(agents)

    def _parse_json_response(self, raw: str) -> dict[str, Any]:
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            return {}

    def _observations_payload(
        self, scope: list[str] | None = None
    ) -> dict[str, dict[str, Any]]:
        """Per-agent observations for robots managed by this domain.
        If `scope` is given, restrict to those agent IDs only (used to keep
        a coalition's prompt limited to its own members instead of every
        agent this domain LLM manages)."""
        managed = scope if scope else (
            self.managed_agent_ids or list(self.node_state.local_observations.keys())
        )
        if not managed:
            managed = [self.node_id]
        return {
            aid: dict(self.node_state.local_observations.get(aid, {}))
            for aid in managed
        }

    def _observations_json(self) -> str:
        return json.dumps(self._observations_payload())

    def update_local_state(
        self,
        local_observation: dict[str, Any] | None = None,
        local_observations: dict[str, dict[str, Any]] | None = None,
        current_task: str | None = None,
        shared_plan_version: int = 0,
    ) -> None:
        """Refresh domain state from per-agent or fleet-level observations."""
        if local_observations:
            for aid, obs in local_observations.items():
                if not self.managed_agent_ids or aid in self.managed_agent_ids:
                    self.node_state.local_observations[aid] = obs
        elif local_observation:
            agent_id = str(
                local_observation.get(
                    "agent_id",
                    self.managed_agent_ids[0] if self.managed_agent_ids else self.node_id,
                )
            )
            self.node_state.local_observations[agent_id] = local_observation
            self.node_state.local_observation = local_observation
        if current_task is not None:
            self.node_state.current_task = current_task
        self.node_state.shared_plan_version = shared_plan_version

    def update_from_fleet_observations(
        self,
        fleet_observations: dict[str, dict[str, Any]],
        shared_plan_version: int = 0,
    ) -> None:
        """Pull observations for all managed agents from a fleet-wide obs map."""
        for aid in self.managed_agent_ids:
            if aid in fleet_observations:
                self.node_state.local_observations[aid] = fleet_observations[aid]
        self.node_state.shared_plan_version = shared_plan_version

    def plan_local(
        self,
        coalition_id: int,
        coalition_members: list[str],
        shared_plan: SharedPlan,
        neighbor_messages: list[PeerMessage],
        coalition_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Domain leader/member generates a plan for its managed agents."""
        from src.llm.prompts import format_prompt

        domain_members = [m for m in coalition_members if m in self.managed_agent_ids]
        obs_payload = self._observations_payload(scope=domain_members)
        subtasks_assigned = coalition_state.get("subtasks", [])
        print("\nDEVICE LLM CALLED")
        print(f"- coalition id: {coalition_id}")
        print(f"- participating agents: {coalition_members}")
        print(f"- subtask: {subtasks_assigned}\n")
        try:
            prompt = format_prompt(
                "plan_local",
                node_id=self.node_id,
                coalition_id=str(coalition_id),
                local_observation=json.dumps(obs_payload),
                coalition_members=json.dumps(coalition_members),
                shared_plan_version=str(shared_plan.version),
                shared_plan=json.dumps(shared_plan.to_dict()),
                neighbor_messages=json.dumps([m.to_dict() for m in neighbor_messages]),
                coalition_state=json.dumps({
                    **coalition_state,
                    "domain_id": self.node_id,
                    "managed_agent_ids": self.managed_agent_ids,
                    "domain_members": domain_members,
                    
                }),
            )
        except (FileNotFoundError, KeyError):
            prompt = (
                f"Plan locally for coalition {coalition_id} as domain {self.node_id}.\n"
                f"Managed agents: {self.managed_agent_ids}\n"
                f"Coalition members: {coalition_members}\n"
                "Return JSON plan_local response."
            )

        obs_json_len = len(json.dumps(obs_payload))
        shared_plan_len = len(json.dumps(shared_plan.to_dict()))
        neighbor_msgs_len = len(json.dumps([m.to_dict() for m in neighbor_messages]))
        print("=" * 70)
        print(f"[PLAN_LOCAL] domain={self.node_id} coalition={coalition_id} "
              f"members={len(coalition_members)} managed_agents={len(self.managed_agent_ids)}")
        print(f"  local_observations chars = {obs_json_len} (embedded once)")
        print(f"  shared_plan chars        = {shared_plan_len}")
        print(f"  neighbor_messages chars  = {neighbor_msgs_len} "
              f"(count={len(neighbor_messages)})")
        print(f"  TOTAL prompt length      = {len(prompt)} chars")
        print(f"  Approx tokens            = {len(prompt)//4}")
        print("=" * 70)

        raw = self.complete(prompt, caller="plan_local", coalition_id=coalition_id)
        result = self._parse_json_response(raw)
        if "assignments" in result:
            self.node_state.belief_state["last_plan"] = result
            self.node_state.shared_plan = result.get("merged_plan", result)
        return result

    def review_peer_plan(
        self,
        peer_id: str,
        peer_plan: dict[str, Any],
        shared_plan_version: int,
    ) -> dict[str, Any]:
        """Review a plan proposed by another Device LLM domain."""
        from src.llm.prompts import format_prompt

        try:
            prompt = format_prompt(
                "review_peer_plan",
                node_id=self.node_id,
                peer_id=peer_id,
                peer_plan=json.dumps(peer_plan),
                local_observation=self._observations_json(),
                shared_plan_version=str(shared_plan_version),
            )
        except (FileNotFoundError, KeyError):
            prompt = (
                f"Review peer plan from domain {peer_id} as domain {self.node_id}.\n"
                f"Managed agents: {self.managed_agent_ids}\n"
                f"Plan: {json.dumps(peer_plan)}\n"
                'Return JSON: {{"approved": true, "revision": {{}}}}'
            )
        raw = self.complete(prompt, caller="review_peer_plan", coalition_id=shared_plan_version)
        result = self._parse_json_response(raw)
        self.node_state.neighbor_plans[peer_id] = peer_plan
        return result

    def merge_peer_plan(
        self,
        coalition_id: int,
        leader_plan: dict[str, Any],
        peer_reviews: list[dict[str, Any]],
        shared_plan_version: int,
    ) -> dict[str, Any]:
        """Merge leader domain plan with peer domain reviews."""
        from src.llm.prompts import format_prompt

        try:
            prompt = format_prompt(
                "merge_peer_plan",
                node_id=self.node_id,
                coalition_id=str(coalition_id),
                leader_plan=json.dumps(leader_plan),
                peer_reviews=json.dumps(peer_reviews),
                shared_plan_version=str(shared_plan_version),
            )
        except (FileNotFoundError, KeyError):
            prompt = (
                f"Merge plans for coalition {coalition_id} at domain {self.node_id}.\n"
                f"Leader: {json.dumps(leader_plan)}\n"
                f"Reviews: {json.dumps(peer_reviews)}"
            )
        raw = self.complete(prompt, caller="merge_peer_plan", coalition_id=coalition_id)
        result = self._parse_json_response(raw)
        merged = result.get("merged_plan", result)
        if merged:
            self.node_state.shared_plan = merged
        return result

    def respond_to_peer(
        self,
        peer_id: str,
        message_type: str,
        payload: dict[str, Any],
        shared_plan_version: int,
    ) -> dict[str, Any]:
        """Generate a domain-level response to an incoming peer message."""
        from src.llm.prompts import format_prompt

        try:
            prompt = format_prompt(
                "respond_to_peer",
                node_id=self.node_id,
                peer_id=peer_id,
                message_type=message_type,
                payload=json.dumps(payload),
                local_observation=self._observations_json(),
                shared_plan_version=str(shared_plan_version),
            )
        except (FileNotFoundError, KeyError):
            prompt = (
                f"Domain {self.node_id} responds to domain {peer_id} "
                f"message type {message_type}.\n"
                f"Payload: {json.dumps(payload)}"
            )
        raw = self.complete(prompt, caller="respond_to_peer", coalition_id=shared_plan_version)
        return self._parse_json_response(raw)

    def ingest_messages(self, messages: list[PeerMessage]) -> None:
        self.node_state.received_messages.extend(messages)

    def dispatch(self, coalitions: list[dict], mode: int = 0) -> dict[str, Any]:
        from src.llm.prompts import format_prompt

        try:
            prompt = format_prompt(
                "dispatch",
                node_id=self.node_id,
                mode=str(mode),
                coalitions=json.dumps(coalitions),
            )
        except (FileNotFoundError, KeyError):
            prompt = (
                f"Dispatch agents per coalitions. Mode m={mode}.\n"
                f"Coalitions: {json.dumps(coalitions)}\n"
                "Return JSON dispatch plan."
            )
        raw = self.complete(prompt)
        result = self._parse_json_response(raw)
        return result if result else {"dispatched": True}

    def coordinate_locally(
        self, coalitions: list[dict], local_state: dict[str, Any]
    ) -> dict[str, Any]:
        """Backward-compatible coordination; accepts per-agent or fleet obs dict."""
        if local_state and "agent_id" in local_state:
            self.update_local_state(local_observation=local_state)
        elif local_state:
            self.update_local_state(local_observations=local_state)

        from src.llm.prompts import format_prompt

        try:
            prompt = format_prompt(
                "coordinate",
                coalitions=json.dumps(coalitions),
                local_state=self._observations_json(),
            )
        except (FileNotFoundError, KeyError):
            prompt = (
                "Coordinate agents locally under decentralized mode.\n"
                f"Domain: {self.node_id}, managed: {self.managed_agent_ids}\n"
                f"Coalitions: {json.dumps(coalitions)}\n"
                f"State: {self._observations_json()}\n"
                "Return JSON coordination plan."
            )
        raw = self.complete(prompt)
        result = self._parse_json_response(raw)
        return result if result else {"action": "coordinate"}

    def reallocate_remaining(
        self,
        remaining_subtasks: list[dict],
        agents: list[dict],
        distance_matrix: list[list[float]],
        cqi_matrix: list[list[float]],
        *,
        scope_to_managed: bool = True,
    ) -> list[dict]:
        from src.llm.prompts import format_prompt
        from src.config import get_thresholds

        scoped_agents = agents
        if scope_to_managed and self.managed_agent_ids:
            managed = set(self.managed_agent_ids)
            scoped_agents = [
                a
                for a in agents
                if str(a.get("agent_id", a.get("id", ""))) in managed
            ]

        gamma_min = get_thresholds().get("gamma_min", 0.3)
        try:
            prompt = format_prompt(
                "reallocate",
                remaining_subtasks=json.dumps(remaining_subtasks),
                agents=json.dumps(scoped_agents),
                distance_matrix=json.dumps(distance_matrix),
                cqi_matrix=json.dumps(cqi_matrix),
                gamma_min=str(gamma_min),
            )
        except (FileNotFoundError, KeyError):
            prompt = (
                f"Reallocate remaining subtasks for domain {self.node_id}.\n"
                f"Managed agents: {self.managed_agent_ids}\n"
                f"Remaining: {json.dumps(remaining_subtasks)}\n"
                f"Agents: {json.dumps(scoped_agents)}\n"
                f"D: {json.dumps(distance_matrix)}, Q: {json.dumps(cqi_matrix)}\n"
                'Return JSON: {"coalitions": [...]}'
            )
        raw = self.complete(prompt)
        result = self._parse_json_response(raw)
        return result.get("coalitions", [])

def aggregate_device_usage(device_llms: dict[str, DeviceLLMClient]) -> DeviceLLMUsage:
    total = DeviceLLMUsage()
    for client in device_llms.values():
        total.prompt_tokens += client.usage.prompt_tokens
        total.completion_tokens += client.usage.completion_tokens
        total.total_tokens += client.usage.total_tokens
        total.tokens = total.total_tokens
        total.retry_tokens += client.usage.retry_tokens
        total.logical_requests += client.usage.logical_requests
        total.device_inference_calls += client.usage.device_inference_calls
        total.device_api_calls += client.usage.device_api_calls
        total.api_calls = total.device_api_calls
        total.successful_calls += client.usage.successful_calls
        total.failed_calls += client.usage.failed_calls
        total.retried_calls += client.usage.retried_calls
        total.cache_hits += client.usage.cache_hits
        total.llm_wait_s += client.usage.llm_wait_s
        total.device_inference_time_s += client.usage.device_inference_time_s
    non_zero_readings = [c.usage.memory_mb for c in device_llms.values() if c.usage.memory_mb > 0]
    total.memory_mb = max(non_zero_readings) if non_zero_readings else 0.0

    # Build real per-domain memory dictionaries
    device_llm_memory_mb = {
        domain: float(np.mean(client.usage.rss_delta_mb_samples)) if client.usage.rss_delta_mb_samples else 0.0
        for domain, client in device_llms.items()
    }
    device_llm_memory_peak_mb = {
        domain: float(np.max(client.usage.rss_delta_mb_samples)) if client.usage.rss_delta_mb_samples else 0.0
        for domain, client in device_llms.items()
    }
    device_llm_heap_delta_mb = {
        domain: float(np.mean(client.usage.heap_delta_mb_samples)) if client.usage.heap_delta_mb_samples else 0.0
        for domain, client in device_llms.items()
    }

    total.device_llm_memory_mb = device_llm_memory_mb
    total.device_llm_memory_peak_mb = device_llm_memory_peak_mb
    total.device_llm_heap_delta_mb = device_llm_heap_delta_mb

    total.python_heap_delta_by_device = device_llm_heap_delta_mb
    total.python_heap_delta_mb = max(device_llm_heap_delta_mb.values(), default=0.0)
    # Load proxy per device role; true per-device GPU memory isolation is not possible under a shared vLLM serving instance with continuous batching.
    total.tokens_processed_by_device = {k: c.usage.total_tokens for k, c in device_llms.items()}
    return total
