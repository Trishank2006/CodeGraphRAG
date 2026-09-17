from generation.models import (
    Citation,
    ContextItem,
    ContextPackage,
    GeneratedAnswer,
)


def test_citation():
    citation = Citation(
        file_path="src/auth/service.py",
        symbol="AuthService.login",
        start_line=20,
        end_line=48,
    )

    assert citation.file_path == "src/auth/service.py"
    assert citation.symbol == "AuthService.login"
    assert citation.start_line == 20
    assert citation.end_line == 48


def test_context_item():
    item = ContextItem(
        id="auth-1",
        file_path="src/auth/service.py",
        symbol="AuthService.login",
        start_line=20,
        end_line=48,
        content="def login(...): ...",
        source="hybrid",
    )

    assert item.id == "auth-1"
    assert item.file_path == "src/auth/service.py"
    assert item.content == "def login(...): ..."
    assert item.source == "hybrid"


def test_context_package():
    item = ContextItem(
        id="auth-1",
        file_path="src/auth/service.py",
        symbol="AuthService.login",
        start_line=20,
        end_line=48,
        content="def login(...): ...",
        source="hybrid",
    )

    package = ContextPackage(
        query="How does authentication work?",
        items=(item,),
        graph_context=(
            "AuthService.login calls TokenService.generate",
        ),
    )

    assert package.query == "How does authentication work?"
    assert len(package.items) == 1
    assert len(package.graph_context) == 1


def test_generated_answer():
    citation = Citation(
        file_path="src/auth/service.py",
        symbol="AuthService.login",
        start_line=20,
        end_line=48,
    )

    answer = GeneratedAnswer(
        answer="Authentication is handled by AuthService.login.",
        citations=(citation,),
    )

    assert "AuthService.login" in answer.answer
    assert len(answer.citations) == 1


def test_default_context_package_is_empty():
    package = ContextPackage(
        query="test",
    )

    assert package.items == ()
    assert package.graph_context == ()


def test_default_generated_answer_has_no_citations():
    answer = GeneratedAnswer(
        answer="Insufficient context.",
    )

    assert answer.citations == ()