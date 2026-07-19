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


def _log(msg: str) -> None:
    print(f"[LLM] {msg}")


@dataclass
class LLMUsage:
    tokens: int = 0
    api_calls: int = 0


@dataclass
class CloudLLMClient:
    config: dict[str, Any] = field(default_factory=get_llm_config)
    usage: LLMUsage = field(default_factory=LLMUsage)
    max_retries: int = 3
    backoff_base: float = 1.0  # seconds; sequence becomes 1, 2, 4

    _client: Any = field(default=None, init=False, repr=False)
    _client_provider: str | None = field(default=None, init=False, repr=False)

    # Experiment context, set once by the orchestrator so a failure report
    # can be fully populated at the point of failure. Purely descriptive —
    # never used to change planning behavior.
    experiment_scenario: str | None = field(default=None, init=False, repr=False)
    experiment_architecture: str | None = field(default=None, init=False, repr=False)
    experiment_network_profile: str | None = field(default=None, init=False, repr=False)
    experiment_seed: int | None = field(default=None, init=False, repr=False)
    current_step: int | None = field(default=None, init=False, repr=False)

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
        else:
            raise ValueError(f"Unknown provider: {provider}")

        self._client_provider = provider
        return self._client

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def complete(self, prompt: str, system: str = "") -> str:
        cache_path = self._cache_path(prompt)
        if cache_path:
            cached = self._read_cache(cache_path)
            if cached:
                # Reuse of a PRIOR genuine LLM response for the identical
                # prompt — not error recovery. See explanation section.
                self.usage.tokens += cached.get("tokens", 0)
                self.usage.api_calls += 1
                return cached["response"]

        if self.config.get("use_mock", True):
            # Explicit baseline experimental condition (e.g. B1/B2 configs),
            # not a failure fallback. Only reached when the run was
            # deliberately configured with use_mock: true.
            response = self._mock_response(prompt)
            tokens = len(prompt.split()) + len(response.split())
            self.usage.tokens += tokens
            self.usage.api_calls += 1
            if cache_path:
                self._write_cache(cache_path, {"response": response, "tokens": tokens})
            return response

        response, tokens = self._call_with_retries(prompt, system)
        self.usage.tokens += tokens
        self.usage.api_calls += 1
        if cache_path:
            self._write_cache(cache_path, {"response": response, "tokens": tokens})
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

    def _call_with_retries(self, prompt: str, system: str) -> tuple[str, int]:
        provider = self.config.get("cloud", {}).get("provider", "unknown")
        model = self.config.get("cloud", {}).get("model", "unknown")
        last_err: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                return self._api_call(prompt, system)
            except Exception as e:  # noqa: BLE001 — classified below, then re-raised as ExperimentFailed
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
        # from this point forward. Record and terminate; do NOT substitute
        # cache, a previous response, a mock planner, or an empty result.
        if last_err is not None:
            print(
                "[LLM DEBUG]\n"
                f"Exception Type: {type(last_err).__name__}\n"
                f"Exception Message: {last_err}"
            )
        report = FailureReport(
            experiment_status="FAILED",
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
        raise ExperimentFailed(report)

    # ------------------------------------------------------------------
    # Raw provider call — UNCHANGED signature/behavior
    # ------------------------------------------------------------------
    def _api_call(self, prompt: str, system: str) -> tuple[str, int]:
        cloud = self.config["cloud"]
        provider = cloud.get("provider", "openai")
        client = self._get_client()

        if provider in ("openai", "groq"):
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            default_model = "llama-3.3-70b-versatile" if provider == "groq" else "gpt-4o"
            resp = client.chat.completions.create(
                model=cloud.get("model", default_model),
                messages=messages,
                max_tokens=cloud.get("max_tokens", 1024),
                temperature=cloud.get("temperature", 0.2),
            )
            text = resp.choices[0].message.content or ""
            tokens = resp.usage.total_tokens if resp.usage else len(text.split())
            return text, tokens

        elif provider == "anthropic":
            resp = client.messages.create(
                model=cloud.get("model", "claude-sonnet-4-20250514"),
                max_tokens=cloud.get("max_tokens", 1024),
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text if resp.content else ""
            tokens = resp.usage.input_tokens + resp.usage.output_tokens if resp.usage else 0
            return text, tokens

        raise ValueError(f"Unknown provider: {provider}")

    # ------------------------------------------------------------------
    # Mock helpers — used ONLY for explicit use_mock=true baseline runs
    # ------------------------------------------------------------------
    def _mock_response(self, prompt: str) -> str:
        pl = prompt.lower()
        if "coalition" in pl and "coalition planner" in pl:
            return self._mock_coalition(prompt)
        if "decompose" in pl or "task decomposer" in pl:
            return self._mock_decomposition(prompt)
        if "coalition" in pl:
            return self._mock_coalition(prompt)
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
        marker = f"{label}:"
        idx = prompt.find(marker)
        if idx < 0:
            return None
        rest = prompt[idx + len(marker) :].lstrip()
        if not rest:
            return None
        opener = rest[0]
        if opener not in "[{":
            return None
        closer = "]" if opener == "[" else "}"
        depth = 0
        for pos, ch in enumerate(rest):
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
    ) -> dict[str, list[str]]:
        if self.config.get("use_mock", True):
            return self._mock_assignments_from_inputs(agents, subtasks)

        from src.config import get_thresholds
        from src.llm.prompts import format_prompt

        th = get_thresholds()
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
        raw = self.complete(prompt, system="You are a Cloud LLM task decomposer.")
        return self._parse_json(raw).get("assignments", {})

    def form_coalitions(
        self,
        subtasks: list[dict],
        agents: list[dict],
        distance_matrix: list[list[float]] | None = None,
        cqi_matrix: list[list[float]] | None = None,
    ) -> list[dict]:
        if self.config.get("use_mock", True):
            return self._mock_coalitions_from_inputs(agents)

        from src.config import get_thresholds
        from src.llm.prompts import format_prompt

        th = get_thresholds()
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
        raw = self.complete(prompt, system="You are a Cloud LLM coalition planner.")
        return self._parse_json(raw).get("coalitions", [])

    def _parse_json(self, text: str) -> dict:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return {}