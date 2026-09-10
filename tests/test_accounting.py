"""Unit and integration tests for LLM API and inference accounting."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.llm.device_llm_client import DeviceLLMClient, DeviceLLMUsage
from src.llm.cloud_llm_client import CloudLLMClient, LLMUsage


def test_device_cache_hit(tmp_path: Path):
    """TEST 1: Device cache hit must increment cache_hits and logical_requests, but NOT device_api_calls/device_inference_calls."""
    config = {
        "use_mock": True,
        "cache_responses": True,
        "cache_dir": str(tmp_path),
    }
    client = DeviceLLMClient(node_id="device_test", config=config)

    prompt = "Unit test prompt for device cache hit"
    cache_path = client._cache_path(prompt)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"response": "cached response", "tokens": 10}, f)

    res = client.complete(prompt)
    assert res == "cached response"
    assert client.usage.cache_hits == 1
    assert client.usage.logical_requests == 1
    assert client.usage.device_api_calls == 0
    assert client.usage.device_inference_calls == 0


def test_device_cache_miss(tmp_path: Path):
    """TEST 2: Device cache miss must increment device_api_calls and device_inference_calls by 1."""
    config = {
        "use_mock": True,
        "cache_responses": True,
        "cache_dir": str(tmp_path),
    }
    client = DeviceLLMClient(node_id="device_test", config=config)

    prompt = "Unit test prompt for device cache miss"
    res = client.complete(prompt)
    assert res is not None
    assert client.usage.cache_hits == 0
    assert client.usage.logical_requests == 1
    assert client.usage.device_api_calls == 1
    assert client.usage.device_inference_calls == 1


def test_cloud_cache_hit(tmp_path: Path):
    """TEST 3: Cloud cache hit must increment cache_hits and logical_requests, but NOT cloud_api_calls or cloud_network_calls."""
    config = {
        "use_mock": True,
        "cache_responses": True,
        "cache_dir": str(tmp_path),
    }
    client = CloudLLMClient(config=config)

    prompt = "Unit test prompt for cloud cache hit"
    cache_path = client._cache_path(prompt)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"response": "cached cloud response", "tokens": 10}, f)

    res = client.complete(prompt)
    assert res == "cached cloud response"
    assert client.usage.cache_hits == 1
    assert client.usage.logical_requests == 1
    assert client.usage.cloud_api_calls == 0
    assert client.usage.cloud_network_calls == 0


def test_cloud_cache_miss(tmp_path: Path):
    """TEST 4: Cloud cache miss must increment cloud_api_calls and cloud_network_calls by 1."""
    config = {
        "use_mock": True,
        "cache_responses": True,
        "cache_dir": str(tmp_path),
    }
    client = CloudLLMClient(config=config)

    prompt = "Unit test prompt for cloud cache miss"
    res = client.complete(prompt)
    assert res is not None
    assert client.usage.cache_hits == 0
    assert client.usage.logical_requests == 1
    assert client.usage.cloud_api_calls == 1
    assert client.usage.cloud_network_calls == 1


def test_repeated_identical_request(tmp_path: Path):
    """TEST 5: Repeated identical requests: 1st call = backend inference, 2nd call = cache hit."""
    config = {
        "use_mock": True,
        "cache_responses": True,
        "cache_dir": str(tmp_path),
    }
    client = DeviceLLMClient(node_id="device_test", config=config)

    prompt = "Repeated identical prompt"

    res1 = client.complete(prompt)
    assert client.usage.logical_requests == 1
    assert client.usage.device_inference_calls == 1
    assert client.usage.device_api_calls == 1
    assert client.usage.cache_hits == 0

    res2 = client.complete(prompt)
    assert res1 == res2
    assert client.usage.logical_requests == 2
    assert client.usage.device_inference_calls == 1
    assert client.usage.device_api_calls == 1
    assert client.usage.cache_hits == 1


def test_retry_accounting(tmp_path: Path):
    """TEST 6: Retry accounting: 1 logical request with 2 backend network attempts."""
    config = {
        "use_mock": False,
        "cache_responses": False,
        "cloud": {"provider": "openai", "model": "gpt-4"},
    }
    client = CloudLLMClient(config=config, max_retries=2, backoff_base=0.01)

    mock_api = MagicMock()
    # 1st attempt fails, 2nd attempt succeeds
    mock_api.side_effect = [RuntimeError("API error"), ("success response", 10, 10, 20)]

    with patch.object(client, "_api_call", mock_api):
        res = client.complete("Prompt with retry")
        assert res == "success response"
        assert client.usage.logical_requests == 1
        assert client.usage.cloud_api_calls == 1
        assert client.usage.cloud_network_calls == 2
        assert client.usage.cloud_failed_attempts == 1
        assert client.usage.retried_calls == 1


def test_deterministic_integration_accounting(tmp_path: Path):
    """
    TEST 7 (Integration Test):
    Sequence: Request A (miss), Request A (hit), Request B (miss), Request B (hit).
    Verify: logical_requests = 4, cache_hits = 2, actual inference calls = 2.
    Also verify exact number of mock invocations.
    """
    config = {
        "use_mock": True,
        "cache_responses": True,
        "cache_dir": str(tmp_path),
    }

    # Test Device Client
    device_client = DeviceLLMClient(node_id="device_test", config=config)
    mock_dev_gen = MagicMock(side_effect=device_client._mock_response)
    with patch.object(device_client, "_mock_response", mock_dev_gen):
        resA1 = device_client.complete("Request A")
        resA2 = device_client.complete("Request A")
        resB1 = device_client.complete("Request B")
        resB2 = device_client.complete("Request B")

        assert resA1 == resA2
        assert resB1 == resB2
        assert device_client.usage.logical_requests == 4
        assert device_client.usage.cache_hits == 2
        assert device_client.usage.device_inference_calls == 2
        assert device_client.usage.device_api_calls == 2
        assert mock_dev_gen.call_count == 2

    # Test Cloud Client
    cloud_client = CloudLLMClient(config=config)
    mock_cloud_gen = MagicMock(side_effect=cloud_client._mock_response)
    with patch.object(cloud_client, "_mock_response", mock_cloud_gen):
        resA1 = cloud_client.complete("Request A")
        resA2 = cloud_client.complete("Request A")
        resB1 = cloud_client.complete("Request B")
        resB2 = cloud_client.complete("Request B")

        assert resA1 == resA2
        assert resB1 == resB2
        assert cloud_client.usage.logical_requests == 4
        assert cloud_client.usage.cache_hits == 2
        assert cloud_client.usage.cloud_api_calls == 2
        assert cloud_client.usage.cloud_network_calls == 2
        assert mock_cloud_gen.call_count == 2
