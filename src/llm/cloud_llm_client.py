"""Cloud LLM client for global task decomposition and coalition formation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import get_llm_config, project_root


@dataclass
class LLMUsage:
    tokens: int = 0
    api_calls: int = 0


@dataclass
class CloudLLMClient:
    config: dict[str, Any] = field(default_factory=get_llm_config)
    usage: LLMUsage = field(default_factory=LLMUsage)

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

    def complete(self, prompt: str, system: str = "") -> str:
        cache_path = self._cache_path(prompt)
        if cache_path:
            cached = self._read_cache(cache_path)
            if cached:
                self.usage.tokens += cached.get("tokens", 0)
                self.usage.api_calls += 1
                return cached["response"]

        if self.config.get("use_mock", True):
            response = self._mock_response(prompt)
            tokens = len(prompt.split()) + len(response.split())
        else:
            response, tokens = self._api_call(prompt, system)

        self.usage.tokens += tokens
        self.usage.api_calls += 1
        if cache_path:
            self._write_cache(cache_path, {"response": response, "tokens": tokens})
        return response

    def _api_call(self, prompt: str, system: str) -> tuple[str, int]:
        cloud = self.config["cloud"]
        provider = cloud.get("provider", "openai")
        if provider == "openai":
            from openai import OpenAI

            client = OpenAI(api_key=os.environ.get(cloud.get("api_key_env", "OPENAI_API_KEY")))
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = client.chat.completions.create(
                model=cloud.get("model", "gpt-4o"),
                messages=messages,
                max_tokens=cloud.get("max_tokens", 1024),
                temperature=cloud.get("temperature", 0.2),
            )
            text = resp.choices[0].message.content or ""
            tokens = resp.usage.total_tokens if resp.usage else len(text.split())
            return text, tokens
        elif provider == "anthropic":
            import anthropic

            client = anthropic.Anthropic(
                api_key=os.environ.get(cloud.get("api_key_env", "ANTHROPIC_API_KEY"))
            )
            resp = client.messages.create(
                model=cloud.get("model", "claude-sonnet-4-20250514"),
                max_tokens=cloud.get("max_tokens", 1024),
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text if resp.content else ""
            tokens = resp.usage.input_tokens + resp.usage.output_tokens if resp.usage else 0
            return text, tokens
        raise ValueError(f"Unknown provider: {provider}")

    def _mock_response(self, prompt: str) -> str:
        if "decompose" in prompt.lower() or "subtask" in prompt.lower():
            return self._mock_decomposition(prompt)
        if "coalition" in prompt.lower():
            return self._mock_coalition(prompt)
        return json.dumps({"status": "ok"})

    def _mock_decomposition(self, prompt: str) -> str:
        try:
            start = prompt.index("{")
            ctx = json.loads(prompt[start:].split("\n")[0] if "\n" in prompt[start:] else prompt[start:])
            subtasks = ctx.get("subtasks", [])
            agents = ctx.get("agents", [])
            assignments = {}
            for i, st in enumerate(subtasks):
                if agents:
                    assignments[st.get("id", f"T_{i}")] = [
                        agents[i % len(agents)]["id"]
                    ]
            return json.dumps({"assignments": assignments})
        except (ValueError, json.JSONDecodeError):
            return json.dumps({"assignments": {}})

    def _mock_coalition(self, prompt: str) -> str:
        try:
            start = prompt.index("{")
            end = prompt.rindex("}") + 1
            ctx = json.loads(prompt[start:end])
            agents = ctx.get("agents", [])
            coalitions = []
            for i in range(0, len(agents), 2):
                group = [a["id"] for a in agents[i : i + 2]]
                if group:
                    coalitions.append({"coalition_id": len(coalitions), "members": group})
            return json.dumps({"coalitions": coalitions})
        except (ValueError, json.JSONDecodeError):
            return json.dumps({"coalitions": []})

    def decompose(
        self,
        instruction: str,
        agents: list[dict],
        subtasks: list[dict],
        distance_matrix: list[list[float]] | None = None,
    ) -> dict[str, list[str]]:
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
