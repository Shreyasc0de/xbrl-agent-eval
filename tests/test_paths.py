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
