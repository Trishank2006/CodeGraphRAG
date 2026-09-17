from generation.context import ContextBuilder
from generation.generator import AnswerGenerator
from generation.llm import MockLLM
from generation.models import GeneratedAnswer
from generation.prompt import PromptBuilder
from retrieval.models import RetrievalResult


class FakeRetrievalPipeline:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def search(self, query: str):
        self.queries.append(query)
        return self.results


def make_result(
    result_id: str,
    file_path: str,
    symbol: str,
    content: str,
    source: str = "dense+bm25+graph",
) -> RetrievalResult:
    return RetrievalResult(
        id=result_id,
        source=source,
        score=1.0,
        repository="test-repo",
        file_path=file_path,
        language="python",
        symbol=symbol,
        start_line=10,
        end_line=30,
        content=content,
    )


def test_generation_end_to_end():
    retrieval = FakeRetrievalPipeline(
        [
            make_result(
                "payment",
                "src/payment/service.py",
                "PaymentService.process_payment",
                "def process_payment(order): validate(order)",
            ),
            make_result(
                "validator",
                "src/payment/validator.py",
                "PaymentValidator.validate",
                "def validate(order): return order.is_valid",
            ),
        ]
    )

    llm = MockLLM(
        "Payment processing is handled by "
        "PaymentService.process_payment."
    )

    generator = AnswerGenerator(
        retrieval_pipeline=retrieval,
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm_client=llm,
    )

    result = generator.answer_query(
        "How does payment processing work?"
    )

    assert isinstance(result, GeneratedAnswer)
    assert (
        result.answer
        == "Payment processing is handled by "
        "PaymentService.process_payment."
    )

    assert retrieval.queries == [
        "How does payment processing work?"
    ]

    assert len(result.citations) == 2

    assert result.citations[0].file_path == (
        "src/payment/service.py"
    )

    assert result.citations[1].file_path == (
        "src/payment/validator.py"
    )

    assert len(llm.received_prompts) == 1

    prompt = llm.received_prompts[0]

    assert "How does payment processing work?" in prompt
    assert "PaymentService.process_payment" in prompt
    assert "PaymentValidator.validate" in prompt
    assert "process_payment(order)" in prompt
    assert "validate(order)" in prompt
    assert "Do not invent files" in prompt


def test_generation_end_to_end_with_empty_retrieval():
    retrieval = FakeRetrievalPipeline([])

    llm = MockLLM(
        "I couldn't determine the answer from "
        "the retrieved repository context."
    )

    generator = AnswerGenerator(
        retrieval_pipeline=retrieval,
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm_client=llm,
    )

    result = generator.answer_query(
        "Why does the deployment fail?"
    )

    assert result.citations == ()
    assert "couldn't determine" in result.answer.lower()

    assert len(llm.received_prompts) == 1
    assert "[NO CODE CONTEXT AVAILABLE]" in (
        llm.received_prompts[0]
    )


def test_generation_end_to_end_deduplicates_context_and_citations():
    retrieval = FakeRetrievalPipeline(
        [
            make_result(
                "auth-1",
                "src/auth/service.py",
                "AuthService.login",
                "def login(): authenticate()",
            ),
            make_result(
                "auth-2",
                "src/auth/service.py",
                "AuthService.login",
                "def login(): authenticate()",
            ),
        ]
    )

    llm = MockLLM("Authentication is handled by AuthService.login.")

    generator = AnswerGenerator(
        retrieval_pipeline=retrieval,
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm_client=llm,
    )

    result = generator.answer_query(
        "How does authentication work?"
    )

    assert len(result.citations) == 1
    assert result.citations[0].symbol == "AuthService.login"

    prompt = llm.received_prompts[0]

    assert prompt.count("AuthService.login") >= 1