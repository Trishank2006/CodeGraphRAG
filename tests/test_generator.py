import pytest

from generation.context import ContextBuilder
from generation.generator import AnswerGenerator, answer_query
from generation.llm import MockLLM
from generation.models import Citation, GeneratedAnswer
from generation.prompt import PromptBuilder
from retrieval.models import RetrievalResult


class FakeRetrievalPipeline:
    def __init__(self, results):
        self.results = results
        self.received_queries = []

    def search(self, query: str):
        self.received_queries.append(query)
        return self.results


def make_result(
    result_id: str,
    file_path: str,
    symbol: str,
    content: str,
) -> RetrievalResult:
    return RetrievalResult(
        id=result_id,
        source="hybrid",
        score=1.0,
        repository="test-repo",
        file_path=file_path,
        language="python",
        symbol=symbol,
        start_line=10,
        end_line=20,
        content=content,
    )


def make_generator(results):
    retrieval = FakeRetrievalPipeline(results)
    context_builder = ContextBuilder()
    prompt_builder = PromptBuilder()
    llm = MockLLM(
        "Authentication is handled by AuthService.login."
    )

    generator = AnswerGenerator(
        retrieval_pipeline=retrieval,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm_client=llm,
    )

    return generator, retrieval, llm


def test_answer_query_returns_generated_answer():
    results = [
        make_result(
            "auth",
            "src/auth/service.py",
            "AuthService.login",
            "def login(): pass",
        )
    ]

    generator, _, _ = make_generator(results)

    result = generator.answer_query(
        "How does authentication work?"
    )

    assert isinstance(result, GeneratedAnswer)
    assert "AuthService.login" in result.answer


def test_answer_query_calls_retrieval_pipeline():
    results = [
        make_result(
            "auth",
            "src/auth/service.py",
            "AuthService.login",
            "def login(): pass",
        )
    ]

    generator, retrieval, _ = make_generator(results)

    generator.answer_query(
        "How does authentication work?"
    )

    assert retrieval.received_queries == [
        "How does authentication work?"
    ]


def test_answer_query_passes_grounded_prompt_to_llm():
    results = [
        make_result(
            "auth",
            "src/auth/service.py",
            "AuthService.login",
            "def login(): pass",
        )
    ]

    generator, _, llm = make_generator(results)

    generator.answer_query(
        "How does authentication work?"
    )

    assert len(llm.received_prompts) == 1

    prompt = llm.received_prompts[0]

    assert "How does authentication work?" in prompt
    assert "src/auth/service.py" in prompt
    assert "AuthService.login" in prompt
    assert "def login(): pass" in prompt


def test_answer_query_creates_structured_citation():
    results = [
        make_result(
            "auth",
            "src/auth/service.py",
            "AuthService.login",
            "def login(): pass",
        )
    ]

    generator, _, _ = make_generator(results)

    result = generator.answer_query(
        "How does authentication work?"
    )

    assert result.citations == (
        Citation(
            file_path="src/auth/service.py",
            symbol="AuthService.login",
            start_line=10,
            end_line=20,
        ),
    )


def test_answer_query_deduplicates_citations():
    results = [
        make_result(
            "auth-1",
            "src/auth/service.py",
            "AuthService.login",
            "def login(): pass",
        ),
        make_result(
            "auth-2",
            "src/auth/service.py",
            "AuthService.login",
            "def login(): pass",
        ),
    ]

    generator, _, _ = make_generator(results)

    result = generator.answer_query(
        "How does authentication work?"
    )

    assert len(result.citations) == 1


def test_answer_query_preserves_multiple_citations():
    results = [
        make_result(
            "auth",
            "src/auth/service.py",
            "AuthService.login",
            "def login(): pass",
        ),
        make_result(
            "token",
            "src/token/service.py",
            "TokenService.generate",
            "def generate(): pass",
        ),
    ]

    generator, _, _ = make_generator(results)

    result = generator.answer_query(
        "How does authentication work?"
    )

    assert len(result.citations) == 2
    assert result.citations[0].file_path == (
        "src/auth/service.py"
    )
    assert result.citations[1].file_path == (
        "src/token/service.py"
    )


def test_answer_query_with_no_retrieval_results():
    generator, _, llm = make_generator([])

    result = generator.answer_query(
        "How does authentication work?"
    )

    assert isinstance(result, GeneratedAnswer)
    assert result.citations == ()
    assert len(llm.received_prompts) == 1
    assert "[NO CODE CONTEXT AVAILABLE]" in (
        llm.received_prompts[0]
    )


def test_answer_query_rejects_empty_query():
    generator, _, _ = make_generator([])

    with pytest.raises(ValueError):
        generator.answer_query("   ")


def test_convenience_answer_query_function():
    results = [
        make_result(
            "auth",
            "src/auth/service.py",
            "AuthService.login",
            "def login(): pass",
        )
    ]

    retrieval = FakeRetrievalPipeline(results)
    context_builder = ContextBuilder()
    prompt_builder = PromptBuilder()
    llm = MockLLM("Generated answer.")

    result = answer_query(
        query="How does authentication work?",
        retrieval_pipeline=retrieval,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm_client=llm,
    )

    assert result.answer == "Generated answer."
    assert len(result.citations) == 1