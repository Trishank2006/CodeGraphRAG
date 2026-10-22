from evaluation.answer_accuracy import (
    cited_files,
    names_file,
    names_symbol,
    score,
    summarize,
)
from evaluation.dataset import EvaluationExample

EXAMPLE = EvaluationExample(
    query="Where is BM25 retrieval implemented?",
    query_type="implementation_lookup",
    expected_files=("retrieval/bm25.py",),
    expected_symbols=("BM25Retriever",),
)


def test_names_file_requires_the_full_path_unless_lenient():
    answer = "It is implemented in bm25.py."

    assert not names_file(answer, ["retrieval/bm25.py"])
    assert names_file(answer, ["retrieval/bm25.py"], lenient=True)


def test_names_symbol_matches_whole_identifiers_only():
    assert names_symbol("See CodeGraphRAGService.search.", ["CodeGraphRAGService.search"])
    assert not names_symbol("See search_code.", ["CodeGraphRAGService.search"])


def test_cited_files_maps_bracket_references_to_citations():
    assert cited_files("Answer [2] and [5].", ["a.py", "b.py"]) == ["b.py"]


def test_score_flags_refusals_and_correct_citations():
    refused = score(EXAMPLE, "Not found in the provided code.", ["retrieval/bm25.py"])
    assert refused["refused"]
    assert not refused["has_reference"]

    good = score(EXAMPLE, "BM25Retriever in `retrieval/bm25.py` [1].", ["retrieval/bm25.py"])
    assert good["names_file"]
    assert good["names_symbol"]
    assert good["cites_expected_file"]


def test_summarize_skips_questions_without_labels():
    base = {
        "refused": False,
        "names_file": True,
        "names_file_lenient": True,
        "has_reference": True,
        "cites_expected_file": True,
    }
    results = [
        dict(base, names_symbol=None),
        dict(base, names_symbol=False, refused=True),
    ]

    summary = summarize(results)

    assert summary["names_symbol"] == {"rate": 0.0, "count": 0, "of": 1}
    assert summary["refused"]["count"] == 1
