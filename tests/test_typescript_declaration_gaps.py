"""TypeScript interfaces and type aliases are not test gaps (issue #1109).

They are stored as Class nodes but have no runtime behavior, so a changed
interface must not appear in test_gaps.
"""

import tempfile
from pathlib import Path

from code_review_graph.changes import analyze_changes
from code_review_graph.graph import GraphStore
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
