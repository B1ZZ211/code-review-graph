"""Strict JSON and comment preservation regressions for issue #1068."""

import itertools
import json
from pathlib import Path

import pytest

from code_review_graph import jsonc
from code_review_graph.jsonc import remove_paths
from code_review_graph.skills import _strip_jsonc


@pytest.mark.parametrize("raw,paths,expected", [
    ('{"keep": 1, "a": 2, "b": 3}', [("a",), ("b",)], {"keep": 1}),
    ('{"a": 1, "b": 2, "keep": 3}', [("a",), ("b",)], {"keep": 3}),
    ('{"items": [1, 2, 3]}', [("items", 1), ("items", 2)], {"items": [1]}),
    ('{"items": [1, 2, 3]}', [("items", 0), ("items", 1)], {"items": [3]}),
    ('{"a": 1, "b": 2}', [("a",), ("b",)], {}),
])
def test_multi_removal_preserves_strict_json(raw, paths, expected):
    assert json.loads(remove_paths(raw, paths)) == expected


def test_removal_preserves_comments_and_string_commas():
    raw = '{"keep": "literal,}", /* user comment */ "a": 2, "b": 3}'
    rewritten = remove_paths(raw, [("a",), ("b",)])
    assert '"literal,}"' in rewritten
    assert "/* user comment */" in rewritten
    assert json.loads(rewritten.replace("/* user comment */", "")) == {"keep": "literal,}"}


def test_existing_jsonc_trailing_comma_is_preserved():
    raw = '{"keep": {"value": 1, /* keep comma */}, "a": 2, "b": 3}'
    rewritten = remove_paths(raw, [("a",), ("b",)])
    assert '{"value": 1, /* keep comma */}' in rewritten


def test_install_uninstall_preserves_strict_claude_settings(tmp_path, monkeypatch):
    from code_review_graph import skills, uninstall

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    repo = tmp_path / "repo"
    (repo / ".git" / "hooks").mkdir(parents=True)
    settings = repo / ".claude" / "settings.json"
    settings.parent.mkdir()
    original = {
        "permissions": {"allow": ["Bash(npm test)"], "deny": ["Bash(git push:*)"]},
        "hooks": {"PreToolUse": [{
            "matcher": "Bash", "hooks": [{"type": "command", "command": "echo MY-OWN-HOOK"}],
        }]},
    }
    settings.write_text(json.dumps(original, indent=2), encoding="utf-8")
    skills.install_hooks(repo)
    json.loads(settings.read_text(encoding="utf-8"))
    report = uninstall.run(repo=repo, keep_data=True)
    assert not report.errors
    assert json.loads(settings.read_text(encoding="utf-8")) == original


# --- Edge cases added while hardening PR #1078 ---------------------------------

def _exposed_trailing_commas(text: str) -> list[int]:
    """Offsets of commas that sit directly before a closing bracket."""
    tokens = jsonc.tokenize(text)
    return [
        token.start
        for token, following in zip(tokens, tokens[1:])
        if token.kind == "," and following.kind in ("}", "]")
    ]


def _subset_cases():
    for size in range(1, 6):
        for removed in itertools.chain.from_iterable(
            itertools.combinations(range(size), count) for count in range(size + 1)
        ):
            yield size, removed


@pytest.mark.parametrize("indent", [None, 2])
def test_every_removal_subset_leaves_strict_json(indent):
    """Any subset of object members or array elements, adjacent or not."""
    for size, removed in _subset_cases():
        obj = {f"k{index}": index for index in range(size)}
        raw = json.dumps({"obj": obj, "tail": 0}, indent=indent)
        rewritten = remove_paths(raw, [("obj", f"k{index}") for index in removed])
        expected = {key: value for key, value in obj.items() if int(key[1:]) not in removed}
        assert json.loads(rewritten) == {"obj": expected, "tail": 0}, (size, removed)

        arr = list(range(size))
        raw = json.dumps({"arr": arr}, indent=indent)
        rewritten = remove_paths(raw, [("arr", index) for index in removed])
        expected_arr = [value for value in arr if value not in removed]
        assert json.loads(rewritten) == {"arr": expected_arr}, (size, removed)


def test_every_removal_subset_keeps_comments_and_exposes_no_comma():
    """Commented JSONC: comments survive and no new trailing comma appears."""
    for size, removed in _subset_cases():
        lines = ["{"]
        for index in range(size):
            comma = "," if index < size - 1 else ""
            lines.append(f'  "k{index}": {index}{comma} // note-{index}')
        lines.append("  /* closing note */")
        lines.append("}")
        raw = "\n".join(lines) + "\n"
        rewritten = remove_paths(raw, [(f"k{index}",) for index in removed])
        for index in range(size):
            assert f"// note-{index}" in rewritten, (size, removed)
        assert "/* closing note */" in rewritten
        assert _exposed_trailing_commas(rewritten) == [], (size, removed)
        assert json.loads(_strip_jsonc(rewritten)) == {
            f"k{index}": index for index in range(size) if index not in removed
        }


def test_no_paths_returns_the_document_unchanged():
    raw = '{\n  "a": [1, 2,],\n  // note\n  "b": {"c": 3,},\n}\n'
    assert remove_paths(raw, []) == raw


def test_surviving_member_keeps_its_own_jsonc_trailing_comma():
    raw = '{\n  "a": 1,\n  "keep": 2,\n}\n'
    rewritten = remove_paths(raw, [("a",)])
    assert '"keep": 2,\n}' in rewritten
    assert json.loads(_strip_jsonc(rewritten)) == {"keep": 2}


def test_removed_tail_with_jsonc_trailing_comma_leaves_strict_json():
    raw = '{"keep": 1, "a": 2, "b": 3,}'
    rewritten = remove_paths(raw, [("a",), ("b",)])
    assert json.loads(rewritten) == {"keep": 1}

    raw = '{"keep": 1, "b": 3,}'
    assert json.loads(remove_paths(raw, [("b",)])) == {"keep": 1}


@pytest.mark.parametrize("paths", [
    [("a",), ("a",), ("b",)],
    [("b",), ("a",), ("b",)],
    [("a",), ("b",), ("a",), ("b",)],
])
def test_duplicate_paths_are_removed_once(paths):
    raw = '{"keep": 1, "a": 2, "b": 3}'
    assert json.loads(remove_paths(raw, paths)) == {"keep": 1}


def test_parent_and_child_paths_overlap_cleanly():
    raw = json.dumps(
        {"keep": 1, "hooks": {"PostToolUse": [1], "SessionStart": [2]}}, indent=2
    )
    rewritten = remove_paths(
        raw, [("hooks", "PostToolUse"), ("hooks",), ("hooks", "SessionStart", 0)]
    )
    assert json.loads(rewritten) == {"keep": 1}


def test_removals_in_several_containers_at_once():
    raw = json.dumps(
        {"x": {"k": 1, "a": 2, "b": 3}, "y": [1, 2, 3], "z": {"n": [4, 5, 6]}, "w": 0},
        indent=2,
    )
    rewritten = remove_paths(raw, [
        ("x", "a"), ("x", "b"), ("y", 1), ("y", 2), ("z", "n", 1), ("z", "n", 2), ("w",),
    ])
    assert json.loads(rewritten) == {"x": {"k": 1}, "y": [1], "z": {"n": [4]}}


def test_nested_hook_entries_removed_at_the_end_of_a_group():
    raw = json.dumps({"hooks": {"PostToolUse": [{
        "matcher": "Edit|Write",
        "hooks": [{"command": "user"}, {"command": "crg-1"}, {"command": "crg-2"}],
    }]}}, indent=2)
    rewritten = remove_paths(raw, [
        ("hooks", "PostToolUse", 0, "hooks", 1),
        ("hooks", "PostToolUse", 0, "hooks", 2),
    ])
    assert json.loads(rewritten) == {"hooks": {"PostToolUse": [{
        "matcher": "Edit|Write", "hooks": [{"command": "user"}],
    }]}}


def test_unicode_escapes_and_windows_paths_before_the_edit_keep_offsets_right():
    original = {
        "ключ": "値,}",
        "emoji": "😀 \",] \\",
        "path": "C:\\Users\\Ünïcode dir\\settings.json,}",
        "a": 1,
        "b": [2],
    }
    for ensure_ascii in (True, False):
        raw = json.dumps(original, indent=2, ensure_ascii=ensure_ascii)
        rewritten = remove_paths(raw, [("a",), ("b",)])
        expected = {key: value for key, value in original.items() if key not in ("a", "b")}
        assert json.loads(rewritten) == expected


def test_crlf_line_endings_survive_and_stay_strict():
    raw = json.dumps({"keep": 1, "a": 2, "b": 3}, indent=2).replace("\n", "\r\n")
    rewritten = remove_paths(raw, [("a",), ("b",)])
    assert json.loads(rewritten) == {"keep": 1}
    assert "\n" not in rewritten.replace("\r\n", "")


def test_large_document_tail_and_interleaved_removals():
    size = 300
    raw = json.dumps({f"k{index}": index for index in range(size)}, indent=2)
    rewritten = remove_paths(raw, [(f"k{index}",) for index in range(size // 2, size)])
    assert json.loads(rewritten) == {f"k{index}": index for index in range(size // 2)}

    raw = json.dumps({"a": list(range(size))})
    rewritten = remove_paths(raw, [("a", index) for index in range(1, size, 2)])
    assert json.loads(rewritten) == {"a": list(range(0, size, 2))}


@pytest.mark.parametrize("raw,paths,error", [
    ('{"a": 1}', [("missing",)], KeyError),
    ('{"a": [1]}', [("a", 3)], IndexError),
    ('{"a": [1]}', [("a", -1)], IndexError),
    ('{"a": 1}', [()], ValueError),
    ("", [("a",)], ValueError),
    ('{"a": 1, "b": 2}', [("b",), ("missing",)], KeyError),
])
def test_unresolvable_paths_still_raise(raw, paths, error):
    with pytest.raises(error):
        remove_paths(raw, paths)
