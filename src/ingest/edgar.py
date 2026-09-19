"""Rate-limited, caching EDGAR fetcher.

SEC requires a descriptive User-Agent containing a real contact email and
enforces a limit of roughly ten requests per second. Both are handled here, and
neither is optional: a client constructed without a contact email raises rather
than sending a request that SEC will reject or, worse, that gets the contact
blocked.

Every response is cached to disk on first fetch. That is what makes runs
reproducible, and it is why the fetcher is safe to call from a loop that may be
re-run: the second run touches the network zero times.

No silent fallbacks. A non-success status raises and nothing is written to the
cache, so a failed fetch can never be mistaken on a later run for a filing that
happens to be empty.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

__all__ = [
    "DERA_QUARTER_URL",
    "EdgarClient",
    "client_from_env",
    "EdgarFetchError",
    "HttpResponse",
    "RateLimiter",
    "build_user_agent",
]

# SEC publishes a limit of 10 requests per second. Sitting slightly under it
# leaves room for clock jitter without tripping the limiter.
DEFAULT_MAX_REQUESTS_PER_SECOND: Final[float] = 8.0

# Retried, with backoff, a bounded number of times. Everything else raises on
# the first response.
_RETRYABLE_STATUSES: Final[frozenset[int]] = frozenset({403, 429, 500, 502, 503, 504})
_DEFAULT_MAX_ATTEMPTS: Final[int] = 4

_EMAIL_RE: Final[re.Pattern[str]] = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")

# Quarterly Financial Statement Data Sets published by SEC DERA.
DERA_QUARTER_URL: Final[str] = (
    "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{quarter}.zip"
)


class EdgarFetchError(RuntimeError):
    """A fetch failed. The task that needed it fails with it."""


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    content: bytes
    headers: Mapping[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300


Transport = Callable[[str, Mapping[str, str]], HttpResponse]


def build_user_agent(app_name: str, contact_email: str) -> str:
    """Build the User-Agent SEC requires, rejecting anything without a contact."""
    if not app_name.strip():
        raise ValueError("app_name must not be empty")
    if not _EMAIL_RE.fullmatch(contact_email.strip()):
        raise ValueError(
            f"contact_email must be a real address, got {contact_email!r}. "
            "SEC rejects requests without one."
        )
    return f"{app_name.strip()} ({contact_email.strip()})"


class RateLimiter:
    """Spaces calls so that no more than ``max_per_second`` are issued.

    The clock and sleep function are injectable so the spacing can be tested
    without spending real seconds.
    """

    def __init__(
        self,
        max_per_second: float = DEFAULT_MAX_REQUESTS_PER_SECOND,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_per_second <= 0:
            raise ValueError(f"max_per_second must be positive, got {max_per_second}")
        self._min_interval = 1.0 / max_per_second
        self._monotonic = monotonic
        self._sleep = sleep
        self._last_call: float | None = None

    def acquire(self) -> None:
        now = self._monotonic()
        if self._last_call is not None:
            elapsed = now - self._last_call
            remaining = self._min_interval - elapsed
            if remaining > 0:
                self._sleep(remaining)
                now = self._monotonic() + remaining
        self._last_call = now


def _cache_path(cache_dir: Path, url: str) -> Path:
    """Content-addressed by URL, so the cache key is stable across runs."""
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return cache_dir / digest[:2] / f"{digest}.bin"


class EdgarClient:
    """Fetches EDGAR documents once, then serves them from disk forever."""

    def __init__(
        self,
        user_agent: str,
        cache_dir: Path,
        *,
        transport: Transport | None = None,
        rate_limiter: RateLimiter | None = None,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not _EMAIL_RE.search(user_agent):
            raise ValueError(
                f"user_agent must contain a contact email, got {user_agent!r}. "
                "Use build_user_agent()."
            )
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
        self.user_agent = user_agent
        self.cache_dir = cache_dir
        self._transport = transport if transport is not None else _requests_transport
        self._limiter = rate_limiter if rate_limiter is not None else RateLimiter()
        self._max_attempts = max_attempts
        self._sleep = sleep
        self.network_fetches = 0
        self.cache_hits = 0

    def get(self, url: str) -> bytes:
        """Return the document at ``url``, from cache when present.

        Raises ``EdgarFetchError`` on any failure. Nothing is cached on failure.
        """
        path = _cache_path(self.cache_dir, url)
        if path.exists():
            self.cache_hits += 1
            return path.read_bytes()

        content = self._fetch(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary sibling and move into place, so an interrupted
        # run cannot leave a truncated file that a later run trusts as cached.
        temporary = path.with_suffix(".partial")
        temporary.write_bytes(content)
        temporary.replace(path)
        return content

    def download(self, url: str, destination: Path) -> Path:
        """Fetch a large file straight to ``destination``, skipping it if present.

        Separate from ``get`` because a quarterly DERA archive runs to tens of
        megabytes and putting it in the response cache as well as on disk would
        store it twice. The skip-if-present check is what makes re-running the
        data pull cheap rather than a fresh multi-gigabyte download.

        The file is written to a temporary sibling and moved into place, so an
        interrupted download cannot leave a truncated archive that a later run
        treats as complete.
        """
        if destination.exists() and destination.stat().st_size > 0:
            self.cache_hits += 1
            return destination

        content = self._fetch(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        temporary.write_bytes(content)
        temporary.replace(destination)
        return destination

    def _fetch(self, url: str) -> bytes:
        headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
        last_status: int | None = None

        for attempt in range(1, self._max_attempts + 1):
            self._limiter.acquire()
            self.network_fetches += 1
            response = self._transport(url, headers)

            if response.ok:
                return response.content

            last_status = response.status_code
            if response.status_code not in _RETRYABLE_STATUSES:
                raise EdgarFetchError(f"{url} returned HTTP {response.status_code}")
            if attempt < self._max_attempts:
                self._sleep(2.0 ** (attempt - 1))

        raise EdgarFetchError(
            f"{url} still returning HTTP {last_status} after {self._max_attempts} attempts"
        )


def _requests_transport(url: str, headers: Mapping[str, str]) -> HttpResponse:
    import requests

    response = requests.get(url, headers=dict(headers), timeout=30)
    return HttpResponse(
        status_code=response.status_code,
        content=response.content,
        headers=dict(response.headers),
    )


def client_from_env(cache_dir: Path) -> EdgarClient:
    """Build a client from ``.env``, refusing to run without a contact address.

    SEC requires a real contact in the User-Agent. Defaulting to a placeholder
    would get requests throttled or the address blocked partway through a run,
    so a missing value stops the run before the first request.
    """
    import os

    from dotenv import load_dotenv

    load_dotenv()
    email = os.environ.get("SEC_CONTACT_EMAIL", "").strip()
    app = os.environ.get("SEC_USER_AGENT_APP", "xbrl-agent-eval").strip()
    if not email:
        raise ValueError(
            "SEC_CONTACT_EMAIL is not set. Copy .env.example to .env and put a real "
            "contact address in it; SEC rejects requests without one."
        )
    return EdgarClient(build_user_agent(app, email), cache_dir)
