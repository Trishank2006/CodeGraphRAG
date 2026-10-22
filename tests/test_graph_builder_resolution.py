from collections import defaultdict

import pytest

from graph.builder import GraphBuilder, _absolute_module
from parser.core_parser import parse_source_code
from parser.models import CodeEntity, CodeRelationship, ParsedFile

REPOSITORY = "test-repo"
PREFIX = f"repo:{REPOSITORY}:"


def entity(file_path, name, kind="function", start=0, end=5):
    return CodeEntity(
        id=f"{file_path}:{kind}:{name}",
        type=kind,
        name=name,
        file_path=file_path,
        start_line=start,
        end_line=end,
    )


def parsed_file(file_path, entities, imports=(), calls=(), language="python"):
    relationships = [
        CodeRelationship(source_id=file_path, target_id=e.id, type="CONTAINS")
        for e in entities
    ]
    relationships += [
        CodeRelationship(source_id=file_path, target_id=f"module:{m}", type="IMPORTS")
        for m in imports
    ]
    relationships += [
        CodeRelationship(source_id=caller.id, target_id=f"symbol:{target}", type="CALLS")
        for caller, target in calls
    ]
    return ParsedFile(
        file_path=file_path,
        language=language,
        entities=list(entities),
        relationships=relationships,
    )


def build_graph(*files):
    return GraphBuilder(REPOSITORY).build_from_parsed_files(list(files))


def call_targets(edges):
    targets = defaultdict(set)
    for edge in edges:
        if edge.type.value == "CALLS":
            targets[edge.source_id.removeprefix(PREFIX)].add(
                edge.target_id.removeprefix(PREFIX)
            )
    return targets


# --- False edges -----------------------------------------------------------

@pytest.mark.parametrize(
    "language, extension",
    [("python", "py"), ("c", "c"), ("javascript", "js"), ("go", "go")],
)
def test_external_call_is_not_linked_to_unrelated_repository_function(language, extension):
    repo_open = entity(f"lib/files.{extension}", "open")
    load = entity(f"app/loader.{extension}", "load")

    _, edges = build_graph(
        parsed_file(f"lib/files.{extension}", [repo_open], language=language),
        parsed_file(f"app/loader.{extension}", [load], calls=[(load, "open")], language=language),
    )

    assert call_targets(edges)[load.id] == {"symbol:open"}


def test_same_named_functions_in_two_imported_modules_stay_unresolved():
    a_search = entity("a.py", "search")
    b_search = entity("b.py", "search")
    run = entity("c.py", "run")

    _, edges = build_graph(
        parsed_file("a.py", [a_search]),
        parsed_file("b.py", [b_search]),
        parsed_file("c.py", [run], imports=["a", "b"], calls=[(run, "search")]),
    )

    assert call_targets(edges)[run.id] == {"symbol:search"}


def test_call_through_a_variable_stays_a_placeholder():
    search = entity("a.py", "search")
    run = entity("b.py", "run")

    nodes, edges = build_graph(
        parsed_file("a.py", [search]),
        parsed_file("b.py", [run], imports=["a"], calls=[(run, "pipeline.search")]),
    )

    assert call_targets(edges)[run.id] == {"symbol:pipeline.search"}
    placeholder = next(n for n in nodes if n.id == PREFIX + "symbol:pipeline.search")
    assert placeholder.name == "search"
    assert placeholder.properties["receiver"] == "pipeline"


# --- Correct resolution ----------------------------------------------------

def test_imported_function_resolves():
    read_file = entity("utils/io.py", "read_file")
    run = entity("app/main.py", "run")

    _, edges = build_graph(
        parsed_file("utils/io.py", [read_file]),
        parsed_file("app/main.py", [run], imports=["utils.io"], calls=[(run, "read_file")]),
    )

    assert call_targets(edges)[run.id] == {read_file.id}


def test_self_call_resolves_to_method_of_same_class():
    service = entity("service.py", "Service", kind="class", start=0, end=30)
    run = entity("service.py", "run", start=2, end=10)
    helper = entity("service.py", "helper", start=12, end=20)

    _, edges = build_graph(
        parsed_file("service.py", [service, run, helper], calls=[(run, "self.helper")]),
    )

    assert call_targets(edges)[run.id] == {helper.id}


def test_class_method_call_prefers_the_imported_class():
    user = entity("models/user.py", "User", kind="class", start=0, end=20)
    create = entity("models/user.py", "create", start=2, end=5)
    legacy_user = entity("legacy/user.py", "User", kind="class", start=0, end=20)
    legacy_create = entity("legacy/user.py", "create", start=2, end=5)
    run = entity("app.py", "run")

    _, edges = build_graph(
        parsed_file("models/user.py", [user, create]),
        parsed_file("legacy/user.py", [legacy_user, legacy_create]),
        parsed_file("app.py", [run], imports=["models.user"], calls=[(run, "User.create")]),
    )

    assert call_targets(edges)[run.id] == {create.id}


def test_go_call_resolves_within_the_same_package_only():
    helper = entity("pkg/a.go", "helper")
    other_helper = entity("other/c.go", "helper")
    run = entity("pkg/b.go", "run")

    _, edges = build_graph(
        parsed_file("pkg/a.go", [helper], language="go"),
        parsed_file("other/c.go", [other_helper], language="go"),
        parsed_file("pkg/b.go", [run], calls=[(run, "helper")], language="go"),
    )

    assert call_targets(edges)[run.id] == {helper.id}


# --- Module lookup -----------------------------------------------------------

def test_exact_module_path_wins_over_same_named_nested_module():
    make = entity("models.py", "make")
    test_make = entity("tests/models.py", "make")
    run = entity("app.py", "run")

    _, edges = build_graph(
        parsed_file("models.py", [make]),
        parsed_file("tests/models.py", [test_make]),
        parsed_file("app.py", [run], imports=["models"], calls=[(run, "make")]),
    )

    assert call_targets(edges)[run.id] == {make.id}


def test_src_layout_import_resolves_through_unique_suffix():
    helper = entity("src/pkg/mod.py", "helper")
    run = entity("src/pkg/app.py", "run")

    _, edges = build_graph(
        parsed_file("src/pkg/mod.py", [helper]),
        parsed_file("src/pkg/app.py", [run], imports=["pkg.mod"], calls=[(run, "helper")]),
    )

    assert call_targets(edges)[run.id] == {helper.id}


# --- Relative imports --------------------------------------------------------

@pytest.mark.parametrize(
    "module, file_path, expected",
    [
        (".models", "app/service.py", "app.models"),
        (".models", "app/__init__.py", "app.models"),
        ("..graph.schema", "app/sub/x.py", "app.graph.schema"),
        (".", "app/service.py", "app"),
        ("..x", "top.py", ""),
        ("os.path", "a.py", "os.path"),
    ],
)
def test_absolute_module(module, file_path, expected):
    assert _absolute_module(module, file_path) == expected


def test_relative_import_resolves_and_is_normalized():
    make_node = entity("graph/schema.py", "make_node")
    builder_fn = entity("graph/builder.py", "build")

    nodes, edges = build_graph(
        parsed_file("graph/schema.py", [make_node]),
        parsed_file("graph/builder.py", [builder_fn], imports=[".schema"], calls=[(builder_fn, "make_node")]),
    )

    assert call_targets(edges)[builder_fn.id] == {make_node.id}
    node_ids = {n.id for n in nodes}
    assert PREFIX + "module:graph.schema" in node_ids
    assert PREFIX + "module:.schema" not in node_ids


def test_parent_relative_import_resolves():
    parse = entity("parser/models.py", "parse")
    run = entity("graph/builder.py", "run")

    _, edges = build_graph(
        parsed_file("parser/models.py", [parse]),
        parsed_file("graph/builder.py", [run], imports=["..parser.models"], calls=[(run, "parse")]),
    )

    assert call_targets(edges)[run.id] == {parse.id}


def test_parser_captures_relative_imports():
    parsed = parse_source_code(
        file_path="graph/builder.py",
        content="from .schema import NodeType\nfrom ..parser import models\n",
        language="python",
    )

    targets = {r.target_id for r in parsed.relationships if r.type == "IMPORTS"}
    assert {"module:.schema", "module:..parser"} <= targets


# --- Graph hygiene -----------------------------------------------------------

def test_relationship_with_missing_source_creates_no_placeholder():
    orphan = ParsedFile(
        file_path="a.py",
        language="python",
        entities=[],
        relationships=[
            CodeRelationship(source_id="a.py:function:ghost", target_id="symbol:foo", type="CALLS")
        ],
    )

    nodes, edges = build_graph(orphan)

    assert not any(n.id.endswith("symbol:foo") for n in nodes)
    assert not edges