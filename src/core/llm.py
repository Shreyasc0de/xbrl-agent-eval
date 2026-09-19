"""Model backends, and the caching layer in front of them.

Three rules run through this module, all from CLAUDE.md and none of them style
preferences.

**No silent fallbacks.** A failed model call raises. It never returns an empty
string, a default answer, or a partially parsed response. The task that needed
the call is recorded as failed and the run continues with one fewer result,
which is a smaller lie than a result that was never computed.

**Every call is cached by content.** The caching wrapper is the only thing that
talks to a backend in a real run, so a rerun is exact and free.

**Work is counted.** ``calls``, ``cache_hits`` and ``cache_misses`` are tracked
so the manifest can assert the run did something before results are written.

The backend protocol is deliberately small: one method, text in and text out
with usage attached. Anything richer would push provider-shaped details into the
agents, and the point of the ablation is that Agent A and Agent B are identical
across backends.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from core.cache import LLMCache, ModelCall, call_key

__all__ = [
    "CachingBackend",
    "Completion",
    "LLMBackend",
    "ModelCallError",
    "OllamaBackend",
    "ScriptedBackend",
]


class ModelCallError(RuntimeError):
    """A model call failed. The task that needed it fails with it."""


@dataclass(frozen=True)
class Completion:
    """What a backend returns. Usage is required, not optional.

    A backend that cannot report token counts reports zeros and says so in
    ``notes``, rather than leaving the field absent and making a cost figure
    silently incomplete.
    """

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float = 0.0
    notes: str = ""


class LLMBackend(Protocol):
    """One model, callable with a prompt and parameters."""

    @property
    def model_id(self) -> str: ...

    def complete(self, prompt: str, params: dict[str, Any]) -> Completion: ...


@dataclass
class ScriptedBackend:
    """A backend that returns prepared responses, for tests.

    Raises when it runs out rather than repeating the last response, because a
    test that silently reuses a response is testing the harness rather than the
    agent.
    """

    responses: list[str]
    model_id_value: str = "scripted"
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    @property
    def model_id(self) -> str:
        return self.model_id_value

    def complete(self, prompt: str, params: dict[str, Any]) -> Completion:
        self.calls.append((prompt, dict(params)))
        if not self.responses:
            raise ModelCallError(
                f"scripted backend exhausted after {len(self.calls)} calls; "
                "the agent asked for more turns than the test prepared"
            )
        text = self.responses.pop(0)
        return Completion(
            text=text,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            latency_ms=0.0,
        )


class OllamaBackend:
    """Local models through Ollama.

    Reachable only from the machine running Ollama. Thinking is disabled and the
    token budget is capped, because an uncapped small model will happily spend
    its whole context reasoning aloud and never emit the structured answer the
    grader needs.

    Cost is zero and recorded as zero. That is the honest figure for a local
    model and it is what makes the cost-per-correct-answer comparison against a
    hosted backend meaningful rather than undefined.
    """

    def __init__(
        self,
        model_id: str,
        base_url: str = "http://localhost:11434",
        *,
        timeout_seconds: float = 120.0,
        num_predict: int = 1024,
    ) -> None:
        self._model_id = model_id
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._num_predict = num_predict

    @property
    def model_id(self) -> str:
        return self._model_id

    def complete(self, prompt: str, params: dict[str, Any]) -> Completion:
        import requests

        options: dict[str, Any] = {
            "temperature": params.get("temperature", 0.0),
            "num_predict": params.get("num_predict", self._num_predict),
        }
        if params.get("seed") is not None:
            options["seed"] = params["seed"]
        if params.get("stop"):
            options["stop"] = params["stop"]

        payload = {
            "model": self._model_id,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": options,
        }

        started = time.perf_counter()
        try:
            response = requests.post(
                f"{self._base_url}/api/generate", json=payload, timeout=self._timeout
            )
        except Exception as error:
            raise ModelCallError(f"ollama request failed: {error}") from error
        latency_ms = (time.perf_counter() - started) * 1000.0

        if response.status_code != 200:
            raise ModelCallError(
                f"ollama returned HTTP {response.status_code}: {response.text[:300]}"
            )

        try:
            body = response.json()
        except ValueError as error:
            raise ModelCallError(f"ollama returned non-JSON: {error}") from error

        text = body.get("response")
        if text is None:
            raise ModelCallError(f"ollama response has no 'response' field: {sorted(body)}")

        return Completion(
            text=text,
            prompt_tokens=int(body.get("prompt_eval_count", 0)),
            completion_tokens=int(body.get("eval_count", 0)),
            latency_ms=latency_ms,
            cost_usd=0.0,
            notes="local model, cost is zero by definition",
        )


class CachingBackend:
    """Wraps a backend so every call is content-addressed and counted.

    This is what a run actually talks to. A cache hit still counts as a model
    call for the purpose of the manifest's work assertion, because the work was
    genuinely done once and reproduced exactly; what it does not count as is a
    cache miss, and the manifest records both so the difference is visible.
    """

    def __init__(self, backend: LLMBackend, cache: LLMCache) -> None:
        self._backend = backend
        self._cache = cache
        self.calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost_usd = 0.0

    @property
    def model_id(self) -> str:
        return self._backend.model_id

    def complete(self, prompt: str, params: dict[str, Any]) -> Completion:
        key = call_key(self._backend.model_id, prompt, params)
        self.calls += 1

        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            self._account(cached.prompt_tokens, cached.completion_tokens, cached.cost_usd)
            return Completion(
                text=cached.response,
                prompt_tokens=cached.prompt_tokens,
                completion_tokens=cached.completion_tokens,
                latency_ms=cached.latency_ms,
                cost_usd=cached.cost_usd,
                notes="served from cache",
            )

        self.cache_misses += 1
        completion = self._backend.complete(prompt, params)

        self._cache.put(
            ModelCall(
                key=key,
                model_id=self._backend.model_id,
                prompt=prompt,
                params_json=json.dumps(params, sort_keys=True, separators=(",", ":")),
                response=completion.text,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                latency_ms=completion.latency_ms,
                cost_usd=completion.cost_usd,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        )
        self._account(completion.prompt_tokens, completion.completion_tokens, completion.cost_usd)
        return completion

    def _account(self, prompt_tokens: int, completion_tokens: int, cost: float) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_cost_usd += cost
