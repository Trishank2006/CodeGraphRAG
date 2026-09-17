from generation.context import ContextBuilder, ContextConfig
from generation.models import ContextPackage
from retrieval.models import RetrievalResult


def make_result(
    result_id: str,
    content: str,
    symbol: str | None = None,
    source: str = "hybrid",
) -> RetrievalResult:
    return RetrievalResult(
        id=result_id,
        source=source,
        score=1.0,
        repository="test-repo",
        file_path=f"src/{result_id}.py",
        language="python",
        symbol=symbol or result_id,
        start_line=1,
        end_line=10,
        content=content,
    )


def test_context_builder_preserves_rank_order():
    results = [
        make_result("first", "first code"),
        make_result("second", "second code"),
        make_result("third", "third code"),
    ]

    builder = ContextBuilder()

    package = builder.build(
        "test query",
        results,
    )

    assert [item.id for item in package.items] == [
        "first",
        "second",
        "third",
    ]


def test_context_builder_deduplicates_by_id():
    results = [
        make_result("shared", "same code"),
        make_result("shared", "same code again"),
        make_result("other", "other code"),
    ]

    builder = ContextBuilder()

    package = builder.build(
        "test query",
        results,
    )

    assert [item.id for item in package.items] == [
        "shared",
        "other",
    ]


def test_context_builder_deduplicates_identical_code():
    results = [
        make_result("one", "same implementation", "process"),
        make_result("two", "same implementation", "process"),
    ]

    builder = ContextBuilder()

    package = builder.build(
        "test query",
        results,
    )

    assert len(package.items) == 1
    assert package.items[0].content == "same implementation"


def test_context_builder_respects_max_items():
    results = [
        make_result("one", "code one"),
        make_result("two", "code two"),
        make_result("three", "code three"),
    ]

    builder = ContextBuilder(
        ContextConfig(max_items=2)
    )

    package = builder.build(
        "test query",
        results,
    )

    assert len(package.items) == 2


def test_context_builder_respects_character_budget():
    results = [
        make_result("one", "a" * 100),
        make_result("two", "b" * 100),
    ]

    builder = ContextBuilder(
        ContextConfig(max_characters=200)
    )

    package = builder.build(
        "test query",
        results,
    )

    assert len(package.items) == 1


def test_context_builder_preserves_metadata():
    results = [
        make_result(
            "auth",
            "def login(): pass",
            symbol="AuthService.login",
            source="dense+bm25",
        )
    ]

    builder = ContextBuilder()

    package = builder.build(
        "how does authentication work?",
        results,
    )

    item = package.items[0]

    assert item.id == "auth"
    assert item.symbol == "AuthService.login"
    assert item.file_path == "src/auth.py"
    assert item.start_line == 1
    assert item.end_line == 10
    assert item.source == "dense+bm25"


def test_context_builder_includes_graph_context():
    results = [
        make_result(
            "auth",
            "authentication logic",
            symbol="AuthService.login",
        )
    ]

    def graph_context(entity_id: str, hops: int):
        assert entity_id == "auth"
        assert hops == 1

        return {
            "entity_id": "auth",
            "name": "AuthService.login",
            "type": "function",
            "file_path": "src/auth.py",
            "connections": [
                {
                    "target_id": "token",
                    "target_name": "TokenService.generate",
                    "target_type": "function",
                    "target_file": "src/token.py",
                    "relationship": "CALLS",
                    "distance": 1,
                }
            ],
        }

    builder = ContextBuilder(
        graph_context_getter=graph_context,
    )

    package = builder.build(
        "how does authentication work?",
        results,
    )

    assert package.graph_context == (
        "AuthService.login --CALLS--> "
        "TokenService.generate (src/token.py) [distance=1]",
    )


def test_context_builder_deduplicates_graph_context():
    results = [
        make_result(
            "auth",
            "authentication logic",
            symbol="AuthService.login",
        ),
        make_result(
            "auth2",
            "other authentication logic",
            symbol="AuthService.login",
        ),
    ]

    def graph_context(entity_id: str, hops: int):
        return {
            "connections": [
                {
                    "target_id": "token",
                    "target_name": "TokenService.generate",
                    "target_type": "function",
                    "target_file": "src/token.py",
                    "relationship": "CALLS",
                    "distance": 1,
                }
            ]
        }

    builder = ContextBuilder(
        graph_context_getter=graph_context,
    )

    package = builder.build(
        "authentication",
        results,
    )

    assert package.graph_context == (
        "AuthService.login --CALLS--> "
        "TokenService.generate (src/token.py) [distance=1]",
    )


def test_empty_query_returns_empty_package():
    results = [
        make_result("auth", "authentication logic"),
    ]

    builder = ContextBuilder()

    package = builder.build(
        "   ",
        results,
    )

    assert isinstance(package, ContextPackage)
    assert package.items == ()
    assert package.graph_context == ()


def test_empty_results_returns_empty_package():
    builder = ContextBuilder()

    package = builder.build(
        "authentication",
        [],
    )

    assert package.items == ()
    assert package.graph_context == ()


def test_context_config_rejects_invalid_values():
    invalid_configs = [
        {"max_items": -1},
        {"max_characters": -1},
        {"graph_hops": 0},
    ]

    for values in invalid_configs:
        try:
            ContextConfig(**values)
            assert False
        except ValueError:
            pass