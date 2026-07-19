"""Device LLM client for local dispatch and distributed domain-level reasoning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import time

import httpx

from src.communication.models import NodeState, PeerMessage, SharedPlan
from src.config import get_llm_config, project_root


@dataclass
class DeviceLLMUsage:
    tokens: int = 0
    api_calls: int = 0
    memory_mb: float = 0.0
    cache_hits: int = 0 


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

    def complete(self, prompt: str) -> str:
        cache_path = self._cache_path(prompt)
        if cache_path and cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
                self.usage.tokens += data.get("tokens", 0)
                self.usage.api_calls += 1
                self.usage.cache_hits += 1
                print(f"[COMPLETE] domain={self.node_id} CACHE HIT "
                      f"prompt_chars={len(prompt)}")
                return data["response"]

        if self.config.get("use_mock", True):
            response = self._mock_response(prompt)
            tokens = len(prompt.split()) + len(response.split())
            self.usage.memory_mb = 4096.0
        else:
            provider = self.config.get("device", {}).get("provider", "ollama")
            start = time.perf_counter()
            if provider == "vllm":
                response, tokens = self._vllm_call(prompt)
            else:
                response, tokens = self._ollama_call(prompt)
            elapsed = time.perf_counter() - start
            print(f"[COMPLETE] domain={self.node_id} provider={provider} "
                  f"elapsed={elapsed:.2f}s prompt_chars={len(prompt)} "
                  f"approx_prompt_tokens={len(prompt)//4} response_tokens={tokens}")
            self.usage.memory_mb = self.config.get("device", {}).get("memory_mb", 8192.0)

        self.usage.tokens += tokens
        self.usage.api_calls += 1
        if cache_path:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"response": response, "tokens": tokens}, f)
        return response

    def _ollama_call(self, prompt: str) -> tuple[str, int]:
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
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response", "")
            tokens = data.get("eval_count", len(text.split()))
        t_elapsed = time.perf_counter() - t_start

        print(f"[OLLAMA-RESPONSE] domain={self.node_id} elapsed={t_elapsed:.2f}s "
              f"response_chars={len(text)} response_tokens={tokens} "
              f"prompt_eval_count={data.get('prompt_eval_count')} "
              f"eval_duration_ns={data.get('eval_duration')} "
              f"prompt_eval_duration_ns={data.get('prompt_eval_duration')} "
              f"load_duration_ns={data.get('load_duration')}")
        return text, tokens

    def _vllm_call(self, prompt: str) -> tuple[str, int]:
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
                    "max_tokens": device.get("max_tokens", 512),
                    "temperature": device.get("temperature", 0.1),
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", len(text.split()))
            return text, tokens

    def _mock_response(self, prompt: str) -> str:
        pl = prompt.lower()
        managed = self.managed_agent_ids or [self.node_id]
        if "dispatch" in pl:
            return json.dumps({"dispatched": True, "assignments": {}})
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

    def _parse_json_response(self, raw: str) -> dict[str, Any]:
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            return {}

    def _observations_payload(self) -> dict[str, dict[str, Any]]:
        """Per-agent observations for all robots managed by this domain."""
        managed = self.managed_agent_ids or list(self.node_state.local_observations.keys())
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
        obs_payload = self._observations_payload()
        try:
            prompt = format_prompt(
                "plan_local",
                node_id=self.node_id,
                coalition_id=str(coalition_id),
                local_observation=self._observations_json(),
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

        raw = self.complete(prompt)
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
        raw = self.complete(prompt)
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
        raw = self.complete(prompt)
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
        raw = self.complete(prompt)
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
        total.tokens += client.usage.tokens
        total.api_calls += client.usage.api_calls
        total.memory_mb += client.usage.memory_mb
        total.cache_hits += client.usage.cache_hits   # NEW
    return total
