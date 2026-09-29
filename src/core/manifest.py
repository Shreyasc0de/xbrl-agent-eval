"""Run manifests.

A run without a complete manifest is not a valid run. The manifest is what lets
someone else, or you in six months, say exactly which model, which prompt, which
questions and which commit produced a number, and it is what
``verify_results.py`` reads when it recomputes the README.

Two fields carry most of the weight. ``prompt_template_hash`` ties results to
the exact prompt text, so an edited prompt cannot quietly inherit an old run's
numbers. ``question_set_hash`` ties them to the exact question set, so a
regenerated set that dropped forty questions cannot be compared against an older
run as though nothing changed.

``ZeroModelCallsError`` exists because the most dangerous failure in a pipeline
like this is not a crash, it is a run that completes having called nothing and
writes a results file full of defaults.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "RunManifest",
    "ZeroModelCallsError",
    "git_commit",
    "prompt_template_hash",
]


class ZeroModelCallsError(RuntimeError):
    """A run reached the end without making a single model call."""


def prompt_template_hash(*templates: str) -> str:
    digest = hashlib.sha256()
    for template in templates:
        digest.update(template.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


def git_commit() -> str:
    """The commit a run was produced from, or a loud marker if unavailable.

    Returns ``"unknown"`` rather than raising, because a run from a tarball with
    no git metadata is still a real run. It is recorded as unknown rather than
    omitted so a reader can tell the difference between a commit that was not
    recorded and one that was.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"

    commit = result.stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if dirty.returncode == 0 and dirty.stdout.strip():
        return f"{commit}-dirty"
    return commit


@dataclass
class RunManifest:
    """Everything needed to say what a run was."""

    run_id: str
    system: str
    model_id: str
    backend: str
    temperature: float
    seed: int | None
    prompt_template_hash: str
    question_set_hash: str
    question_count: int
    # Present when the run covered a sample rather than the whole set. Without
    # the seed a sampled run cannot be reproduced, and a run that cannot be
    # reproduced is not evidence.
    sample_size: int | None = None
    sample_seed: int | None = None
    git_commit: str = field(default_factory=git_commit)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str = ""
    wall_clock_seconds: float = 0.0
    model_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    notes: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def assert_work_happened(self, *, allow_full_cache: bool = True) -> None:
        """Refuse to write results for a run that did no work.

        A run served entirely from cache is legitimate and is the normal case on
        a rerun, so cache hits count as work. A run with no calls of any kind is
        not: it means the pipeline short-circuited, and the results file it
        would write describes nothing.
        """
        if self.model_calls <= 0:
            raise ZeroModelCallsError(
                f"run {self.run_id} made zero model calls. Refusing to write results, "
                "because a results file from a run that called nothing is fabricated "
                "output regardless of what it contains."
            )
        if not allow_full_cache and self.cache_misses <= 0:
            raise ZeroModelCallsError(
                f"run {self.run_id} was served entirely from cache but was declared "
                "a fresh run. Either the cache is stale or the run type is wrong."
            )

    def complete(self, wall_clock_seconds: float) -> None:
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.wall_clock_seconds = wall_clock_seconds

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["total_tokens"] = self.total_tokens
        return payload

    def write(self, path: Path) -> Path:
        """Write the manifest, after checking the run actually happened."""
        self.assert_work_happened()
        if not self.finished_at:
            raise ValueError(
                "manifest written before complete() was called; wall clock would be zero"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True), encoding="utf-8")
        return path
