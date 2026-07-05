"""Device LLM client for local dispatch and coordination."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from src.config import get_llm_config, project_root


@dataclass
class DeviceLLMUsage:
    tokens: int = 0
    api_calls: int = 0
    memory_mb: float = 0.0


@dataclass
class DeviceLLMClient:
    config: dict[str, Any] = field(default_factory=get_llm_config)
    usage: DeviceLLMUsage = field(default_factory=DeviceLLMUsage)
    node_id: str = "device_0"

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
                return data["response"]

        if self.config.get("use_mock", True):
            response = self._mock_response(prompt)
            tokens = len(prompt.split()) + len(response.split())
            self.usage.memory_mb = 4096.0
        else:
            provider = self.config.get("device", {}).get("provider", "ollama")
            if provider == "vllm":
                response, tokens = self._vllm_call(prompt)
            else:
                response, tokens = self._ollama_call(prompt)
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
        with httpx.Client(timeout=120.0) as client:
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
            return text, tokens

    def _vllm_call(self, prompt: str) -> tuple[str, int]:
        """OpenAI-compatible vLLM server endpoint."""
        device = self.config["device"]
        base_url = device.get("base_url", "http://localhost:8000/v1")
        model = device.get("model", "meta-llama/Llama-3.1-8B-Instruct")
        with httpx.Client(timeout=120.0) as client:
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
        if "dispatch" in prompt.lower():
            return json.dumps({"dispatched": True, "assignments": {}})
        if "coordinate" in prompt.lower() or "realloc" in prompt.lower():
            return json.dumps({"action": "reallocate", "status": "ok"})
        return json.dumps({"status": "ack"})

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
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            return {"dispatched": True}

    def coordinate_locally(
        self, coalitions: list[dict], local_state: dict
    ) -> dict[str, Any]:
        from src.llm.prompts import format_prompt

        try:
            prompt = format_prompt(
                "coordinate",
                coalitions=json.dumps(coalitions),
                local_state=json.dumps(local_state),
            )
        except (FileNotFoundError, KeyError):
            prompt = (
                "Coordinate agents locally under decentralized mode.\n"
                f"Coalitions: {json.dumps(coalitions)}\n"
                f"State: {json.dumps(local_state)}\n"
                "Return JSON coordination plan."
            )
        raw = self.complete(prompt)
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            return {"action": "coordinate"}

    def reallocate_remaining(
        self,
        remaining_subtasks: list[dict],
        agents: list[dict],
        distance_matrix: list[list[float]],
        cqi_matrix: list[list[float]],
    ) -> list[dict]:
        from src.llm.prompts import format_prompt
        from src.config import get_thresholds

        gamma_min = get_thresholds().get("gamma_min", 0.3)
        try:
            prompt = format_prompt(
                "reallocate",
                remaining_subtasks=json.dumps(remaining_subtasks),
                agents=json.dumps(agents),
                distance_matrix=json.dumps(distance_matrix),
                cqi_matrix=json.dumps(cqi_matrix),
                gamma_min=str(gamma_min),
            )
        except (FileNotFoundError, KeyError):
            prompt = (
                "Reallocate remaining subtasks to feasible coalitions locally.\n"
                f"Remaining: {json.dumps(remaining_subtasks)}\n"
                f"Agents: {json.dumps(agents)}\n"
                f"D: {json.dumps(distance_matrix)}, Q: {json.dumps(cqi_matrix)}\n"
                'Return JSON: {"coalitions": [...]}'
            )
        raw = self.complete(prompt)
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return json.loads(raw[start:end]).get("coalitions", [])
        except (ValueError, json.JSONDecodeError):
            return []
