"""EDGAR fetcher: compliance, caching, and failure behavior.

No test here touches the network. The transport is injected, so the suite runs
in CI and in a sandbox with SEC unreachable.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from ingest.edgar import (
    EdgarClient,
    EdgarFetchError,
    HttpResponse,
    RateLimiter,
    build_user_agent,
)

UA = "xbrl-agent-eval someone@example.com"


class FakeTransport:
    def __init__(self, *responses: HttpResponse) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, str]]] = []

    def __call__(self, url: str, headers: Mapping[str, str]) -> HttpResponse:
        self.calls.append((url, dict(headers)))
        if len(self._responses) == 1:
            return self._responses[0]
        return self._responses.pop(0)


def ok(body: bytes = b"<html>filing</html>") -> HttpResponse:
    return HttpResponse(status_code=200, content=body)


def client(tmp_path: Path, transport: FakeTransport, **kwargs: object) -> EdgarClient:
    return EdgarClient(
        UA,
        tmp_path / "cache",
        transport=transport,
        rate_limiter=RateLimiter(1000.0, monotonic=lambda: 0.0, sleep=lambda _: None),
        sleep=lambda _: None,
        **kwargs,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------
# Compliance
# --------------------------------------------------------------------------


def test_user_agent_requires_a_real_contact_email() -> None:
    with pytest.raises(ValueError):
        build_user_agent("xbrl-agent-eval", "not-an-email")


def test_user_agent_rejects_an_empty_app_name() -> None:
    with pytest.raises(ValueError):
        build_user_agent("   ", "someone@example.com")


def test_build_user_agent_produces_the_expected_shape() -> None:
    assert build_user_agent("xbrl-agent-eval", "a@b.co") == "xbrl-agent-eval a@b.co"


def test_client_refuses_a_user_agent_without_a_contact(tmp_path: Path) -> None:
    """SEC blocks anonymous scrapers. Failing here is better than getting the

    contact address blocked partway through a run.
    """
    with pytest.raises(ValueError):
        EdgarClient("python-requests/2.32", tmp_path, transport=FakeTransport(ok()))


def test_the_contact_is_actually_sent(tmp_path: Path) -> None:
    transport = FakeTransport(ok())
    client(tmp_path, transport).get("https://www.sec.gov/x")
    _, headers = transport.calls[0]
    assert headers["User-Agent"] == UA


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------


def test_rate_limiter_spaces_consecutive_calls() -> None:
    slept: list[float] = []
    limiter = RateLimiter(10.0, monotonic=lambda: 0.0, sleep=slept.append)
    limiter.acquire()
    limiter.acquire()
    assert slept == [pytest.approx(0.1)]


def test_rate_limiter_does_not_sleep_when_enough_time_has_passed() -> None:
    slept: list[float] = []
    clock = iter([0.0, 5.0])
    limiter = RateLimiter(10.0, monotonic=lambda: next(clock), sleep=slept.append)
    limiter.acquire()
    limiter.acquire()
    assert slept == []


def test_rate_limiter_rejects_a_nonpositive_rate() -> None:
    with pytest.raises(ValueError):
        RateLimiter(0.0)


def test_default_rate_sits_under_the_sec_limit() -> None:
    from ingest.edgar import DEFAULT_MAX_REQUESTS_PER_SECOND

    assert DEFAULT_MAX_REQUESTS_PER_SECOND < 10.0


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------


def test_a_second_get_does_not_touch_the_network(tmp_path: Path) -> None:
    transport = FakeTransport(ok(b"body"))
    edgar = client(tmp_path, transport)
    first = edgar.get("https://www.sec.gov/a")
    second = edgar.get("https://www.sec.gov/a")
    assert first == second == b"body"
    assert len(transport.calls) == 1
    assert edgar.network_fetches == 1
    assert edgar.cache_hits == 1


def test_a_fresh_client_reuses_the_cache_on_disk(tmp_path: Path) -> None:
    """This is what makes a rerun reproducible and free."""
    first_transport = FakeTransport(ok(b"body"))
    client(tmp_path, first_transport).get("https://www.sec.gov/a")

    second_transport = FakeTransport(ok(b"different"))
    result = client(tmp_path, second_transport).get("https://www.sec.gov/a")

    assert result == b"body"
    assert second_transport.calls == []


def test_different_urls_do_not_collide(tmp_path: Path) -> None:
    transport = FakeTransport(ok(b"a"), ok(b"b"))
    edgar = client(tmp_path, transport)
    assert edgar.get("https://www.sec.gov/a") == b"a"
    assert edgar.get("https://www.sec.gov/b") == b"b"


# --------------------------------------------------------------------------
# Failure is failure
# --------------------------------------------------------------------------


def test_a_404_raises_rather_than_returning_empty_bytes(tmp_path: Path) -> None:
    transport = FakeTransport(HttpResponse(404, b""))
    with pytest.raises(EdgarFetchError):
        client(tmp_path, transport).get("https://www.sec.gov/missing")


def test_a_failed_fetch_caches_nothing(tmp_path: Path) -> None:
    """A cached empty body would be indistinguishable on a later run from a

    filing that legitimately has no content, which is exactly the kind of
    silent fallback that invalidates a benchmark.
    """
    edgar = client(tmp_path, FakeTransport(HttpResponse(404, b"")))
    with pytest.raises(EdgarFetchError):
        edgar.get("https://www.sec.gov/missing")

    cache = tmp_path / "cache"
    written = list(cache.rglob("*")) if cache.exists() else []
    assert [p for p in written if p.is_file()] == []


def test_a_404_is_not_retried(tmp_path: Path) -> None:
    transport = FakeTransport(HttpResponse(404, b""))
    with pytest.raises(EdgarFetchError):
        client(tmp_path, transport).get("https://www.sec.gov/missing")
    assert len(transport.calls) == 1


def test_a_403_is_not_retried(tmp_path: Path) -> None:
    """SEC returns 403 for an undeclared automated tool. That is a statement

    about the request, not a transient condition, and retrying it is what gets
    a contact address blocked.
    """
    transport = FakeTransport(HttpResponse(403, b"Undeclared automated tool"))
    with pytest.raises(EdgarFetchError, match="403"):
        client(tmp_path, transport).get("https://www.sec.gov/a")
    assert len(transport.calls) == 1


def test_an_error_message_carries_the_response_body_and_user_agent(
    tmp_path: Path,
) -> None:
    """SEC explains a refusal in the body. Discarding it turns a diagnosable

    problem into a bare status code.
    """
    transport = FakeTransport(HttpResponse(403, b"Your request originated from an undeclared tool"))
    with pytest.raises(EdgarFetchError) as caught:
        client(tmp_path, transport).get("https://www.sec.gov/a")
    message = str(caught.value)
    assert "undeclared" in message
    assert "someone@example.com" in message


def test_a_429_is_retried_then_gives_up(tmp_path: Path) -> None:
    transport = FakeTransport(HttpResponse(429, b""))
    with pytest.raises(EdgarFetchError):
        client(tmp_path, transport, max_attempts=3).get("https://www.sec.gov/a")
    assert len(transport.calls) == 3


def test_a_429_followed_by_success_succeeds(tmp_path: Path) -> None:
    transport = FakeTransport(HttpResponse(429, b""), ok(b"body"))
    assert client(tmp_path, transport).get("https://www.sec.gov/a") == b"body"


def test_a_partial_cache_file_is_not_left_behind(tmp_path: Path) -> None:
    edgar = client(tmp_path, FakeTransport(ok(b"body")))
    edgar.get("https://www.sec.gov/a")
    assert list((tmp_path / "cache").rglob("*.partial")) == []
