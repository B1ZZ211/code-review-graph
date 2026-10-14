"""Uninstall must leave strict JSON behind in every hook and MCP config it edits.

Regression for issue #1068, fixed by PR #1078. Clients such as Claude Code read
``settings.json`` as strict JSON and skip the whole file on a trailing comma,
which silently disables the user's own hooks and permission rules. Uninstall
removes several sibling entries in one splice, so these tests seed the user's
own settings, run the real installer and uninstaller, and read the result back
with plain ``json.loads`` rather than the tolerant JSONC reader.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import pytest

from code_review_graph import skills, uninstall

_USER_PERMISSIONS = {"allow": ["Bash(npm test)"], "deny": ["Bash(git push:*)"]}
_USER_PRE_TOOL = [{
    "matcher": "Bash",
    "hooks": [{"type": "command", "command": "echo MY-OWN-HOOK"}],
}]


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home dir ü"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HERMES_HOME", str(home / ".hermes"))
    return home


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo with spaces ü"
    (repo / ".git" / "hooks").mkdir(parents=True)
    return repo


def _strict(path: Path) -> Any:
    """Parse exactly as a strict-JSON client would."""
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _claude(repo: Path, home: Path) -> tuple[Path, Callable[[], object]]:
    return repo / ".claude" / "settings.json", lambda: skills.install_hooks(repo)


def _qoder(repo: Path, home: Path) -> tuple[Path, Callable[[], object]]:
    return (
        repo / ".qoder" / "settings.json",
        lambda: skills.install_hooks(repo, platform="qoder"),
    )


def _codebuddy(repo: Path, home: Path) -> tuple[Path, Callable[[], object]]:
    return (
        repo / ".codebuddy" / "settings.json",
        lambda: skills.install_codebuddy_hooks(repo),
    )


def _codex(repo: Path, home: Path) -> tuple[Path, Callable[[], object]]:
    return home / ".codex" / "hooks.json", lambda: skills.install_codex_hooks(repo)


def _gemini(repo: Path, home: Path) -> tuple[Path, Callable[[], object]]:
    return (
        repo / ".gemini" / "settings.json",
        lambda: skills.install_gemini_cli_hooks(repo),
    )


_NESTED_TARGETS = {
    "claude": _claude,
    "qoder": _qoder,
    "codebuddy": _codebuddy,
    "codex": _codex,
    "gemini": _gemini,
}


def _nested_original(same_events: bool) -> dict[str, Any]:
    hooks: dict[str, Any] = {"PreToolUse": copy.deepcopy(_USER_PRE_TOOL)}
    if same_events:
        # The user already listens on the events the installer also uses, so
        # removal happens inside existing arrays rather than whole members.
        for event in ("PostToolUse", "SessionStart", "AfterTool"):
            hooks[event] = [{
                "matcher": "Edit",
                "hooks": [{"type": "command", "command": f"echo user-{event}"}],
            }]
    return {"permissions": copy.deepcopy(_USER_PERMISSIONS), "hooks": hooks}


@pytest.mark.parametrize("same_events", [False, True], ids=["own-events", "shared-events"])
@pytest.mark.parametrize("target", sorted(_NESTED_TARGETS))
def test_install_uninstall_leaves_strict_json_with_user_hooks(
    target: str, same_events: bool, fake_repo: Path, fake_home: Path
) -> None:
    settings, install = _NESTED_TARGETS[target](fake_repo, fake_home)
    original = _nested_original(same_events)
    _write_json(settings, original)

    install()
    installed = _strict(settings)
    assert installed != original, "installer did not add its hooks"

    report = uninstall.run(repo=fake_repo, keep_data=True)

    assert not report.errors
    assert _strict(settings) == original


def test_cursor_flat_hooks_uninstall_leaves_strict_json(
    fake_repo: Path, fake_home: Path
) -> None:
    hooks_json = fake_home / ".cursor" / "hooks.json"
    original = {
        "version": 1,
        "hooks": {"beforeSubmitPrompt": [{"command": "./my-own-hook.sh"}]},
    }
    _write_json(hooks_json, original)

    skills.install_cursor_hooks()
    assert set(_strict(hooks_json)["hooks"]) > {"beforeSubmitPrompt"}

    report = uninstall.run(repo=fake_repo, keep_data=True)

    assert not report.errors
    assert _strict(hooks_json) == original


def test_continue_duplicate_registrations_at_the_end_leave_strict_json(
    fake_repo: Path, fake_home: Path
) -> None:
    config = fake_home / ".continue" / "config.json"
    other = {"name": "other", "command": "other-server"}
    # Earlier releases could stack several registrations side by side.
    _write_json(config, {
        "models": [{"title": "mine"}],
        "mcpServers": [
            other,
            {"name": "code-review-graph", "command": "code-review-graph"},
            {"name": "code-review-graph", "command": "uvx"},
        ],
    })

    report = uninstall.run(repo=fake_repo, keep_data=True)

    assert not report.errors
    assert _strict(config) == {"models": [{"title": "mine"}], "mcpServers": [other]}


def test_object_mcp_entry_registered_last_leaves_strict_json(
    fake_repo: Path, fake_home: Path
) -> None:
    config = fake_repo / ".mcp.json"
    _write_json(config, {"mcpServers": {
        "other": {"command": "other"},
        "code-review-graph": {"command": "code-review-graph"},
    }})

    uninstall.run(repo=fake_repo, keep_data=True)

    assert _strict(config) == {"mcpServers": {"other": {"command": "other"}}}


def test_hooks_written_by_an_earlier_release_leave_strict_json(
    fake_repo: Path, fake_home: Path
) -> None:
    settings = fake_repo / ".claude" / "settings.json"
    legacy = sorted(uninstall._legacy_repo_hook_commands(fake_repo))
    original = {"permissions": _USER_PERMISSIONS, "hooks": {"PreToolUse": _USER_PRE_TOOL}}
    seeded = copy.deepcopy(original)
    seeded["hooks"]["PostToolUse"] = [
        {"matcher": "Edit|Write", "hooks": [{"type": "command", "command": legacy[1]}]},
    ]
    seeded["hooks"]["SessionStart"] = [
        {"matcher": "", "hooks": [{"type": "command", "command": legacy[0]}]},
    ]
    _write_json(settings, seeded)

    uninstall.run(repo=fake_repo, keep_data=True)

    assert _strict(settings) == original


def test_owned_hooks_appended_inside_a_user_group_leave_strict_json(
    fake_repo: Path, fake_home: Path
) -> None:
    settings = fake_repo / ".claude" / "settings.json"
    owned = [
        hook["command"]
        for entries in skills.generate_hooks_config(fake_repo)["hooks"].values()
        for entry in entries
        for hook in entry["hooks"]
    ]
    user_hook = {"type": "command", "command": "echo mine"}
    _write_json(settings, {
        "permissions": _USER_PERMISSIONS,
        "hooks": {"PostToolUse": [{
            "matcher": "Edit|Write",
            "hooks": [user_hook] + [{"type": "command", "command": cmd} for cmd in owned],
        }]},
    })

    uninstall.run(repo=fake_repo, keep_data=True)

    assert _strict(settings) == {
        "permissions": _USER_PERMISSIONS,
        "hooks": {"PostToolUse": [{"matcher": "Edit|Write", "hooks": [user_hook]}]},
    }


def test_second_uninstall_and_reinstall_cycles_stay_strict(
    fake_repo: Path, fake_home: Path
) -> None:
    settings = fake_repo / ".claude" / "settings.json"
    original = _nested_original(same_events=False)
    _write_json(settings, original)

    for _ in range(2):
        skills.install_hooks(fake_repo)
        uninstall.run(repo=fake_repo, keep_data=True)
        assert _strict(settings) == original
        settled = settings.read_bytes()
        uninstall.run(repo=fake_repo, keep_data=True)
        assert settings.read_bytes() == settled


def test_dry_run_leaves_the_installed_settings_byte_for_byte(
    fake_repo: Path, fake_home: Path
) -> None:
    settings = fake_repo / ".claude" / "settings.json"
    _write_json(settings, _nested_original(same_events=False))
    skills.install_hooks(fake_repo)
    before = settings.read_bytes()

    uninstall.run(repo=fake_repo, keep_data=True, dry_run=True)

    assert settings.read_bytes() == before


def test_cli_install_then_uninstall_reproduction_from_issue_1068(
    fake_repo: Path, fake_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from code_review_graph import cli

    settings = fake_repo / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    # Byte-for-byte the file from the issue's reproduction steps.
    settings.write_text(
        '{ "permissions": { "allow": ["Bash(npm test)"], "deny": ["Bash(git push:*)"] },\n'
        '  "hooks": { "PreToolUse": [ { "matcher": "Bash", "hooks": [ { "type": '
        '"command", "command": "echo MY-OWN-HOOK" } ] } ] } }\n',
        encoding="utf-8",
    )
    original = _strict(settings)

    argv = ["code-review-graph", "install", "--repo", str(fake_repo), "-y",
            "--platform", "claude-code"]
    with patch.object(sys, "argv", argv):
        cli.main()
    assert {"PostToolUse", "SessionStart"} <= set(_strict(settings)["hooks"])

    argv = ["code-review-graph", "uninstall", "--repo", str(fake_repo), "-y"]
    with patch.object(sys, "argv", argv):
        cli.main()
    capsys.readouterr()

    assert _strict(settings) == original
