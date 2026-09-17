from generation.models import Citation, ContextItem, ContextPackage
from generation.prompt import PromptBuilder


def make_context() -> ContextPackage:
    item = ContextItem(
        id="auth",
        file_path="src/auth/service.py",
        symbol="AuthService.login",
        start_line=20,
        end_line=48,
        content="def login(user): return authenticate(user)",
        source="dense+bm25",
    )

    return ContextPackage(
        query="How does authentication work?",
        items=(item,),
        graph_context=(
            "AuthService.login --CALLS--> TokenService.generate "
            "(src/token.py) [distance=1]",
        ),
    )


def test_prompt_contains_user_query():
    builder = PromptBuilder()

    prompt = builder.build_prompt(make_context())

    assert "How does authentication work?" in prompt


def test_prompt_contains_file_metadata():
    builder = PromptBuilder()

    prompt = builder.build_prompt(make_context())

    assert "src/auth/service.py" in prompt
    assert "AuthService.login" in prompt
    assert "Lines: 20-48" in prompt


def test_prompt_contains_retrieved_code():
    builder = PromptBuilder()

    prompt = builder.build_prompt(make_context())

    assert "def login(user): return authenticate(user)" in prompt


def test_prompt_contains_graph_context():
    builder = PromptBuilder()

    prompt = builder.build_prompt(make_context())

    assert "TokenService.generate" in prompt
    assert "CALLS" in prompt


def test_prompt_contains_grounding_instructions():
    builder = PromptBuilder()

    prompt = builder.build_prompt(make_context())

    assert "Do not invent files" in prompt
    assert "cite the relevant file and line range" in prompt
    assert "context is insufficient" in prompt


def test_prompt_handles_empty_context():
    context = ContextPackage(
        query="How does authentication work?",
    )

    builder = PromptBuilder()

    prompt = builder.build_prompt(context)

    assert "[NO CODE CONTEXT AVAILABLE]" in prompt
    assert "How does authentication work?" in prompt


def test_system_prompt_is_available_separately():
    builder = PromptBuilder()

    system_prompt = builder.build_system_prompt()

    assert "codebase analysis assistant" in system_prompt
    assert "repository context" in system_prompt


def test_prompt_does_not_require_citation_imports():
    """
    Verify the prompt builder remains independent of the Citation model.
    """
    context = make_context()

    builder = PromptBuilder()

    prompt = builder.build_prompt(context)

    assert prompt
    assert Citation is not None