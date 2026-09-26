"""Where the data lives, and the fact that everything agrees about it.

These exist because the four entry points and the Makefile once disagreed: the
Makefile was moved off the Desktop and the CLIs were not, so running a module
directly failed looking for a database that no longer existed. The defaults are
now defined once and asserted here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.paths import DEFAULT_DB_ENV, default_db_path, default_filings_dir


def test_the_default_database_is_outside_any_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A checkout on the Desktop sits inside a synced folder, and SQLite on one

    dies with a bus error partway through a multi-gigabyte write.
    """
    monkeypatch.delenv(DEFAULT_DB_ENV, raising=False)
    path = default_db_path()
    assert path.is_absolute()
    assert "Desktop" not in str(path)
    assert path.name == "xbrl.db"


def test_the_environment_variable_overrides_the_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(DEFAULT_DB_ENV, str(tmp_path / "custom.db"))
    assert default_db_path() == tmp_path / "custom.db"


def test_a_tilde_in_the_override_is_expanded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DEFAULT_DB_ENV, "~/elsewhere/xbrl.db")
    assert str(default_db_path()).startswith(str(Path.home()))


def test_an_empty_override_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(DEFAULT_DB_ENV, "   ")
    assert default_db_path().name == "xbrl.db"


def test_filings_live_beside_the_database(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Cached filings are fetched artifacts, not source, and they run to

    hundreds of megabytes. They follow the database rather than the checkout.
    """
    monkeypatch.setenv(DEFAULT_DB_ENV, str(tmp_path / "db" / "xbrl.db"))
    assert default_filings_dir() == tmp_path / "db" / "filings"


def test_every_entry_point_uses_the_shared_default() -> None:
    """The regression this file exists for: four places, one answer."""
    import inspect

    from eval import run as eval_run
    from ingest import dera_loader, filings
    from questions import generate

    for module in (dera_loader, filings, generate, eval_run):
        source = inspect.getsource(module)
        assert 'Path("data/xbrl.db")' not in source, (
            f"{module.__name__} hard-codes a database path instead of using "
            "core.paths.default_db_path"
        )


def test_every_source_package_is_type_checked_in_ci() -> None:
    """CI runs a bare ``mypy``, which checks only the packages listed in config.

    A package missing from that list is silently unchecked: it passes locally
    under ``mypy --strict src`` and is skipped entirely on every push. That is
    how ``annotate`` went a commit without type checking.

    Read with a regex rather than tomllib, which is 3.11 and up while this
    project supports 3.10. A test that cannot run on the oldest supported
    interpreter does not guard the build on it.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    section = text.split("[tool.mypy]", 1)
    assert len(section) == 2, "pyproject.toml has no [tool.mypy] section"
    match = re.search(r"^packages\s*=\s*\[(.*?)\]", section[1], re.MULTILINE | re.DOTALL)
    assert match, "[tool.mypy] declares no packages list"
    declared = set(re.findall(r'"([^"]+)"', match.group(1)))

    on_disk = {
        path.name
        for path in (root / "src").iterdir()
        if path.is_dir() and (path / "__init__.py").exists()
    }
    assert on_disk <= declared, f"not type checked in CI: {sorted(on_disk - declared)}"


def test_the_makefile_never_looks_for_data_inside_the_checkout() -> None:
    """The same divergence as above, one level down.

    ``make doctor`` checked ``data/filings/`` and ``data/questions/`` in the
    repository while the pipeline writes both beside the database, so it
    reported them missing when they existed. It stayed invisible because
    doctor only ever prints: nothing fails when a status line lies.
    """
    import re
    from pathlib import Path

    makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text(encoding="utf-8")
    # The hyphen in the lookbehind matters: the default database path is
    # ~/xbrl-data/xbrl.db, which is outside the checkout and must not match.
    pattern = re.compile(r"(?<![\w/$(){}-])data/(filings|questions|xbrl\.db)")
    offending = [
        line
        for line in makefile.splitlines()
        if not line.lstrip().startswith("#") and pattern.search(line)
    ]
    # A guard that cannot fail is not a guard. These are the exact line the
    # bug was, and the exact line it must not flag.
    assert pattern.search("\t@if [ -f data/filings/manifest.jsonl ]; then \\")
    assert not pattern.search("DB ?= $(HOME)/xbrl-data/xbrl.db")

    assert not offending, (
        "the Makefile points at data/ inside the checkout; it must derive "
        f"these from DB, as core.paths does: {offending}"
    )


def test_no_pipeline_target_runs_a_bare_python() -> None:
    """A bare ``python`` in a recipe is whatever the shell resolves.

    On a machine with conda on PATH that is conda's interpreter, which has none
    of this project's packages, so the run dies with ModuleNotFoundError: No
    module named 'ingest'. That reads like a broken checkout rather than a
    forgotten activate, and it cost a load that had been running for half an
    hour. Every pipeline target names the venv's interpreter by path.
    """
    import re
    from pathlib import Path

    makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text(encoding="utf-8")
    # Only module invocations at a command position. Echoed help text and the
    # python3 probes in doctor are not runs of the project's code.
    pattern = re.compile(r"(?:^\t|[;&|]\s*)python\s+-m\b")
    offending = [line for line in makefile.splitlines() if pattern.search(line)]

    # A guard that cannot fail is not a guard.
    assert pattern.search("\tpython -m ingest.dera_loader --db $(DB)")
    assert not pattern.search("\t$(VENV_PY) -m ingest.dera_loader --db $(DB)")

    assert not offending, f"recipes run a bare python instead of $(VENV_PY): {offending}"
