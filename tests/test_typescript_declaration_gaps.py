"""TypeScript interfaces and type aliases are not test gaps (issue #1109, PR #1112).

They are stored as Class nodes but have no runtime behavior, so a changed
interface must not appear in test_gaps.

The later tests pin the edges of that rule: where the declarations can sit
(TSX, ``.mts``/``.cts``/``.d.ts``, namespaces, ambient blocks, Vue and Svelte
``<script lang="ts">``), what must stay a gap (enums, classes, components,
other languages' interfaces, merged class/interface declarations, a class of
the same name elsewhere), the real build/update path, and graphs written
before the parser recorded ``typescript_kind``.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from code_review_graph.changes import analyze_changes
from code_review_graph.graph import GraphStore
from code_review_graph.incremental import full_build, incremental_update
from code_review_graph.parser import CodeParser, NodeInfo


def test_parser_tags_interfaces_and_type_aliases_only():
    source = (
        b"export interface WidgetProps { title: string }\n"
        b"export type Size = { w: number };\n"
        b"export class Widget { render() { return 1; } }\n"
    )
    nodes, _ = CodeParser().parse_bytes(Path("widget.ts"), source)
    by_name = {n.name: n for n in nodes if n.kind == "Class"}

    assert by_name["WidgetProps"].extra["typescript_kind"] == "interface_declaration"
    assert by_name["Size"].extra["typescript_kind"] == "type_alias_declaration"
    assert "typescript_kind" not in by_name["Widget"].extra


class TestTypeScriptDeclarationGaps:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = GraphStore(self.tmp.name)

    def teardown_method(self):
        self.store.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def _add_class(self, name: str, extra: dict, line_start: int) -> None:
        node = NodeInfo(
            kind="Class",
            name=name,
            file_path="widget.ts",
            line_start=line_start,
            line_end=line_start + 2,
            language="typescript",
            is_test=False,
            extra=extra,
        )
        self.store.upsert_node(node, file_hash="abc")
        self.store.commit()

    def test_interface_is_not_a_gap_but_a_real_class_still_is(self):
        self._add_class("WidgetProps", {"typescript_kind": "interface"}, 1)
        self._add_class("Widget", {}, 10)

        result = analyze_changes(
            self.store,
            changed_files=["widget.ts"],
            changed_ranges={"widget.ts": [(1, 20)]},
        )
        gap_names = {g["name"] for g in result["test_gaps"]}

        assert "WidgetProps" not in gap_names
        assert "Widget" in gap_names


# ---------------------------------------------------------------------------
# Parser: where the tag is set, and where it must not be
# ---------------------------------------------------------------------------


def _classes(path: str, source: str) -> dict[str, NodeInfo]:
    nodes, _ = CodeParser().parse_bytes(Path(path), source.encode("utf-8"))
    return {n.name: n for n in nodes if n.kind == "Class"}


def test_tsx_tags_declarations_but_not_components_or_classes():
    """The shape from issue #1109: a .tsx file with props, a component, a class."""
    nodes, _ = CodeParser().parse_bytes(
        Path("widget.tsx"),
        b"export interface WidgetProps { title: string }\n"
        b"export type Size = { w: number };\n"
        b"export function Widget({ title }: WidgetProps) {\n"
        b"  return <h1>{title}</h1>;\n"
        b"}\n"
        b"export class Panel extends React.Component { render() { return null; } }\n",
    )
    by_name = {n.name: n for n in nodes if n.kind != "File"}

    assert by_name["WidgetProps"].language == "tsx"
    assert by_name["WidgetProps"].extra["typescript_kind"] == "interface_declaration"
    assert by_name["Size"].extra["typescript_kind"] == "type_alias_declaration"
    assert by_name["Widget"].kind == "Function"
    assert "typescript_kind" not in by_name["Widget"].extra
    assert "typescript_kind" not in by_name["Panel"].extra
    assert "typescript_kind" not in by_name["render"].extra


@pytest.mark.parametrize("suffix", [".mts", ".cts", ".d.ts"])
def test_typescript_module_and_declaration_files_are_tagged(suffix):
    classes = _classes(
        f"types{suffix}",
        "export interface Decl { a: number }\nexport type Alias = string;\n",
    )
    assert classes["Decl"].extra["typescript_kind"] == "interface_declaration"
    assert classes["Alias"].extra["typescript_kind"] == "type_alias_declaration"


@pytest.mark.parametrize(
    ("source", "name", "kind"),
    [
        ("export default interface Props { a: string }\n",
         "Props", "interface_declaration"),
        ("export namespace NS { export interface Inner { a: number } }\n",
         "Inner", "interface_declaration"),
        ("declare global { interface Window { crg: string } }\n",
         "Window", "interface_declaration"),
        ("declare module 'x' { export interface Ambient { b: number } }\n",
         "Ambient", "interface_declaration"),
        ("export declare interface Declared { c: number }\n",
         "Declared", "interface_declaration"),
        ("export interface Box<T> { v: T }\n", "Box", "interface_declaration"),
        ("export interface Svc { run(): void; go: () => void }\n",
         "Svc", "interface_declaration"),
        ("export type Fn<T> = (x: T) => void;\n", "Fn", "type_alias_declaration"),
        ("export type Tone = 'light' | 'dark';\n", "Tone", "type_alias_declaration"),
    ],
)
def test_wrapped_and_ambient_declarations_are_tagged(source, name, kind):
    classes = _classes("decls.ts", source)
    assert classes[name].extra["typescript_kind"] == kind


def test_interface_method_signatures_do_not_become_function_gaps():
    """Method signatures inside an interface are not Function nodes at all."""
    nodes, _ = CodeParser().parse_bytes(
        Path("svc.ts"), b"export interface Svc { run(): void; go: () => void }\n",
    )
    assert [n.name for n in nodes if n.kind == "Function"] == []


def test_unicode_declaration_names_are_tagged():
    classes = _classes(
        "unicode.ts",
        "export interface \u00dcn\u00efc\u00f6d\u00e9 { a: string }\n"
        "export type \u578b = number;\n",
    )
    assert classes["\u00dcn\u00efc\u00f6d\u00e9"].extra["typescript_kind"] == (
        "interface_declaration"
    )
    assert classes["\u578b"].extra["typescript_kind"] == "type_alias_declaration"


@pytest.mark.parametrize(
    ("source", "name"),
    [
        ("export enum Color { Red, Green }\n", "Color"),
        ("export const enum Flag { A }\n", "Flag"),
        ("export class Widget { render() { return 1; } }\n", "Widget"),
        ("export namespace NS { export class C { m() { return 1; } } }\n", "C"),
    ],
)
def test_runtime_typescript_declarations_are_not_tagged(source, name):
    """Enums and classes emit runtime code, so they stay gap candidates."""
    assert "typescript_kind" not in _classes("runtime.ts", source)[name].extra


@pytest.mark.parametrize(
    ("path", "source", "name"),
    [
        ("Widget.js", "export class Widget { render() { return 1; } }\n", "Widget"),
        ("Runner.java", "public interface Runner { void run(); }\n", "Runner"),
        ("Runner.cs", "public interface IRunner { void Run(); }\n", "IRunner"),
    ],
)
def test_other_languages_are_not_tagged(path, source, name):
    """The tag is TypeScript-only, even where another grammar reuses the
    ``interface_declaration`` node type."""
    assert "typescript_kind" not in _classes(path, source)[name].extra


@pytest.mark.parametrize(
    ("path", "source", "language"),
    [
        (
            "Comp.vue",
            "<template><div/></template>\n"
            "<script lang=\"ts\">\n"
            "export interface VProps { a: string }\n"
            "export type VSize = number;\n"
            "export class VC { m() { return 1; } }\n"
            "</script>\n",
            "vue",
        ),
        (
            "Comp.svelte",
            "<script lang=\"ts\">\n"
            "  export interface VProps { a: string }\n"
            "  export type VSize = number;\n"
            "  export class VC { m() { return 1; } }\n"
            "</script>\n"
            "<div/>\n",
            "svelte",
        ),
    ],
)
def test_single_file_component_typescript_declarations_are_tagged(path, source, language):
    """Vue and Svelte relabel script nodes with their own language; the tag
    is set while the block is parsed as TypeScript and must survive that."""
    classes = _classes(path, source)
    assert classes["VProps"].language == language
    assert classes["VProps"].extra["typescript_kind"] == "interface_declaration"
    assert classes["VSize"].extra["typescript_kind"] == "type_alias_declaration"
    assert "typescript_kind" not in classes["VC"].extra


# ---------------------------------------------------------------------------
# Real build and update path
# ---------------------------------------------------------------------------

_WIDGET = (
    "export interface WidgetProps { title: string }\n"
    "export type Size = { w: number };\n"
    "export enum Tone { Light, Dark }\n"
    "export function Widget({ title }: WidgetProps) {\n"
    "  return <h1>{title}</h1>;\n"
    "}\n"
    "export class Panel { render() { return 1; } }\n"
)
_WIDGET_EDITED = _WIDGET.replace("title: string }", "title: string; sub?: string }")
_RUNTIME_GAPS = {"Tone", "Widget", "Panel", "render"}
_DECLARATIONS = {"WidgetProps", "Size"}


class TestBuildAndUpdatePath:
    """Parse -> SQLite -> analyze_changes, through full_build and
    incremental_update, in a repository path with spaces and non-ASCII."""

    @pytest.fixture
    def repo(self, tmp_path):
        root = tmp_path / "my repo \u00fc"
        (root / "src dir").mkdir(parents=True)
        (root / ".git").mkdir()
        (root / "src dir" / "widget.tsx").write_text(_WIDGET, encoding="utf-8")
        store = GraphStore(root / "graph.db")
        try:
            yield root, store
        finally:
            store.close()

    @staticmethod
    def _build(root: Path, store: GraphStore) -> None:
        with patch(
            "code_review_graph.incremental.get_all_tracked_files",
            return_value=["src dir/widget.tsx"],
        ):
            result = full_build(root, store)
        assert result["errors"] == []

    @staticmethod
    def _widget_path(root: Path, store: GraphStore) -> str:
        (path,) = [
            f for f in store.get_all_files() if f.endswith("widget.tsx")
        ]
        return path

    def _gaps(self, root: Path, store: GraphStore, ranges: bool = True) -> dict:
        path = self._widget_path(root, store)
        return analyze_changes(
            store,
            changed_files=[path],
            changed_ranges={path: [(1, 7)]} if ranges else None,
        )

    def test_full_build_drops_declarations_but_keeps_runtime_gaps(self, repo):
        root, store = repo
        self._build(root, store)
        result = self._gaps(root, store)

        assert {g["name"] for g in result["test_gaps"]} == _RUNTIME_GAPS
        # Other analyses still see the declarations: they are changed nodes
        # with a risk score, only the gap report leaves them out.
        changed = {f["name"]: f for f in result["changed_functions"]}
        assert _DECLARATIONS <= set(changed)
        assert all("risk_score" in changed[n] for n in _DECLARATIONS)
        assert f"{len(_RUNTIME_GAPS)} test gap(s)" in result["summary"]
        untested_line = next(
            line for line in result["summary"].splitlines() if "Untested:" in line
        )
        assert not any(name in untested_line for name in _DECLARATIONS)

    def test_whole_file_fallback_also_drops_declarations(self, repo):
        """No line ranges: every node of the changed file is a candidate."""
        root, store = repo
        self._build(root, store)
        result = self._gaps(root, store, ranges=False)
        assert {g["name"] for g in result["test_gaps"]} == _RUNTIME_GAPS

    def test_change_touching_only_an_interface_reports_no_gap(self, repo):
        root, store = repo
        self._build(root, store)
        path = self._widget_path(root, store)
        result = analyze_changes(
            store, changed_files=[path], changed_ranges={path: [(1, 2)]},
        )
        assert {f["name"] for f in result["changed_functions"]} == _DECLARATIONS
        assert result["test_gaps"] == []

    def test_graph_from_an_earlier_release_heals_when_the_file_is_reparsed(self, repo):
        """A graph built before this change has no ``typescript_kind``.

        Until the file is parsed again its declarations are still reported
        (the documented limitation). The normal edit -> update path reparses
        the file, and from then on they drop out; running the update or a
        full rebuild again changes nothing.
        """
        root, store = repo
        self._build(root, store)
        path = self._widget_path(root, store)
        for node in store.get_nodes_by_file(path):
            if node.extra.get("typescript_kind"):
                legacy = {k: v for k, v in node.extra.items() if k != "typescript_kind"}
                store._conn.execute(
                    "UPDATE nodes SET extra = ? WHERE id = ?",
                    (json.dumps(legacy), node.id),
                )
        store.commit()

        stale = {g["name"] for g in self._gaps(root, store)["test_gaps"]}
        assert stale == _RUNTIME_GAPS | _DECLARATIONS

        (root / "src dir" / "widget.tsx").write_text(_WIDGET_EDITED, encoding="utf-8")
        incremental_update(root, store, changed_files=["src dir/widget.tsx"])
        assert {g["name"] for g in self._gaps(root, store)["test_gaps"]} == _RUNTIME_GAPS

        incremental_update(root, store, changed_files=["src dir/widget.tsx"])
        assert {g["name"] for g in self._gaps(root, store)["test_gaps"]} == _RUNTIME_GAPS

        self._build(root, store)
        assert {g["name"] for g in self._gaps(root, store)["test_gaps"]} == _RUNTIME_GAPS


# ---------------------------------------------------------------------------
# Name collisions, missing metadata and scale
# ---------------------------------------------------------------------------


class TestCollisionsAndScale:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = GraphStore(self.tmp.name)

    def teardown_method(self):
        self.store.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def _store_parsed(self, path: str, source: str) -> None:
        nodes, _ = CodeParser().parse_bytes(Path(path), source.encode("utf-8"))
        for node in nodes:
            self.store.upsert_node(node, file_hash="abc")
        self.store.commit()

    def _gap_names(self, path: str, start: int, end: int) -> set[str]:
        result = analyze_changes(
            self.store,
            changed_files=[path],
            changed_ranges={path: [(start, end)]},
        )
        return {g["name"] for g in result["test_gaps"]}

    def test_class_of_the_same_name_in_another_file_is_still_a_gap(self):
        self._store_parsed("/r/types.ts", "export interface Widget { a: string }\n")
        self._store_parsed("/r/widget.ts", "export class Widget { render() { return 1; } }\n")

        assert self._gap_names("/r/types.ts", 1, 1) == set()
        assert self._gap_names("/r/widget.ts", 1, 1) == {"Widget", "render"}

    def test_interface_merged_into_a_class_keeps_the_class_methods_as_gaps(self):
        """Declaration merging gives the class and the interface one
        qualified name, so whichever comes last owns the stored node. The
        class's methods are separate nodes and must stay gaps either way."""
        self._store_parsed(
            "/r/after.ts",
            "export class Emitter {\n"
            "  emit() { return 1; }\n"
            "}\n"
            "export interface Emitter {\n"
            "  on(e: string): this;\n"
            "}\n",
        )
        assert "emit" in self._gap_names("/r/after.ts", 1, 6)

        self._store_parsed(
            "/r/before.ts",
            "export interface Mixed {\n"
            "  on(e: string): this;\n"
            "}\n"
            "export class Mixed {\n"
            "  emit() { return 1; }\n"
            "}\n",
        )
        # The class is stored last here, so it keeps its own class-level gap.
        assert self._gap_names("/r/before.ts", 1, 6) == {"Mixed", "emit"}

    def test_null_extra_from_an_old_row_is_still_a_gap(self):
        self._store_parsed("/r/widget.ts", "export class Widget { render() { return 1; } }\n")
        self.store._conn.execute(
            "UPDATE nodes SET extra = NULL WHERE name = 'Widget'"
        )
        self.store.commit()
        assert self._gap_names("/r/widget.ts", 1, 1) == {"Widget", "render"}

    def test_many_declarations_in_one_file_leave_only_runtime_gaps(self):
        # 402 changed nodes: under the default CRG_MAX_CHANGED_FUNCS cap of
        # 500, which is applied before the gap filter.
        lines = []
        for i in range(200):
            lines.append(f"export interface I{i} {{ v{i}: number }}")
            lines.append(f"export type T{i} = I{i} | null;")
        lines.append("export class Big { run() { return 1; } }")
        self._store_parsed("/r/big.ts", "\n".join(lines) + "\n")

        result = analyze_changes(
            self.store,
            changed_files=["/r/big.ts"],
            changed_ranges={"/r/big.ts": [(1, len(lines))]},
        )
        assert {g["name"] for g in result["test_gaps"]} == {"Big", "run"}
        assert len(result["changed_functions"]) == 402
