from generation.context import ContextBuilder
from generation.generator import AnswerGenerator
from generation.llm import MockLLM
from generation.models import Citation
from generation.prompt import PromptBuilder
from retrieval.models import RetrievalResult


class FakeRetrievalPipeline:
    def __init__(self, results):
        self.results = results

    def search(self, query: str):
        return self.results


def make_result(
    result_id: str,
    file_path: str,
    symbol: str,
    start_line: int,
    end_line: int,
) -> RetrievalResult:
    return RetrievalResult(
        id=result_id,
        source="hybrid",
        score=1.0,
        repository="test-repo",
        file_path=file_path,
        language="python",
        symbol=symbol,
        start_line=start_line,
        end_line=end_line,
        content=f"implementation for {symbol}",
    )


def make_generator(results):
    return AnswerGenerator(
        retrieval_pipeline=FakeRetrievalPipeline(results),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm_client=MockLLM("Generated answer."),
    )


def test_citation_contains_repository_location():
    generator = make_generator(
        [
            make_result(
                "auth",
                "src/auth/service.py",
                "AuthService.login",
                20,
                48,
            )
        ]
    )

    result = generator.answer_query(
        "How does authentication work?"
    )

    assert result.citations == (
        Citation(
            file_path="src/auth/service.py",
            symbol="AuthService.login",
            start_line=20,
            end_line=48,
        ),
    )


def test_citations_preserve_context_order():
    generator = make_generator(
        [
            make_result(
                "first",
                "src/first.py",
                "First.run",
                1,
                10,
            ),
            make_result(
                "second",
                "src/second.py",
                "Second.run",
                20,
                30,
            ),
        ]
    )

    result = generator.answer_query("How does the system work?")

    assert [
        citation.file_path
        for citation in result.citations
    ] == [
        "src/first.py",
        "src/second.py",
    ]


def test_duplicate_citations_are_removed():
    generator = make_generator(
        [
            make_result(
                "auth-1",
                "src/auth/service.py",
                "AuthService.login",
                20,
                48,
            ),
            make_result(
                "auth-2",
                "src/auth/service.py",
                "AuthService.login",
                20,
                48,
            ),
        ]
    )

    result = generator.answer_query(
        "How does authentication work?"
    )

    assert len(result.citations) == 1


def test_citation_metadata_matches_retrieved_code():
    generator = make_generator(
        [
            make_result(
                "payment",
                "src/payment/service.py",
                "PaymentService.process_payment",
                10,
                35,
            )
        ]
    )

    result = generator.answer_query(
        "How does payment processing work?"
    )

    citation = result.citations[0]

    assert citation.file_path == "src/payment/service.py"
    assert citation.symbol == "PaymentService.process_payment"
    assert citation.start_line == 10
    assert citation.end_line == 35