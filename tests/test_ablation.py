import pytest

from evaluation.ablation import (
    ABLATION_STAGES,
    ablation_to_dict,
    compare_to_baseline,
    comparison_to_dict,
    create_ablation_result,
    results_to_table,
    validate_ablation_order,
)


def make_results():
    return [
        create_ablation_result(
            "Dense",
            recall=0.50,
            mrr=0.40,
            ndcg=0.45,
            faithfulness=0.60,
            citation_accuracy=0.70,
            latency_ms=20.0,
        ),
        create_ablation_result(
            "Dense + BM25",
            recall=0.60,
            mrr=0.50,
            ndcg=0.55,
            faithfulness=0.65,
            citation_accuracy=0.75,
            latency_ms=25.0,
        ),
        create_ablation_result(
            "Dense + BM25 + RRF",
            recall=0.70,
            mrr=0.60,
            ndcg=0.65,
            faithfulness=0.70,
            citation_accuracy=0.80,
            latency_ms=30.0,
        ),
        create_ablation_result(
            "Dense + BM25 + RRF + Reranker",
            recall=0.75,
            mrr=0.68,
            ndcg=0.72,
            faithfulness=0.76,
            citation_accuracy=0.84,
            latency_ms=40.0,
        ),
        create_ablation_result(
            "Full System",
            recall=0.80,
            mrr=0.72,
            ndcg=0.78,
            faithfulness=0.82,
            citation_accuracy=0.90,
            latency_ms=55.0,
        ),
    ]


def test_ablation_stage_order():
    assert ABLATION_STAGES == (
        "Dense",
        "Dense + BM25",
        "Dense + BM25 + RRF",
        "Dense + BM25 + RRF + Reranker",
        "Full System",
    )


def test_validate_ablation_order():
    validate_ablation_order(ABLATION_STAGES)


def test_invalid_ablation_order():
    with pytest.raises(ValueError):
        validate_ablation_order(
            [
                "Dense",
                "Full System",
                "Dense + BM25",
            ]
        )


def test_create_ablation_result():
    result = create_ablation_result(
        "Dense",
        recall=0.5,
        mrr=0.4,
        ndcg=0.45,
        faithfulness=0.6,
        citation_accuracy=0.7,
        latency_ms=20.0,
    )

    assert result.name == "Dense"
    assert result.recall == 0.5
    assert result.mrr == 0.4
    assert result.ndcg == 0.45
    assert result.faithfulness == 0.6
    assert result.citation_accuracy == 0.7
    assert result.latency_ms == 20.0


@pytest.mark.parametrize(
    "metric",
    [
        "recall",
        "mrr",
        "ndcg",
        "faithfulness",
        "citation_accuracy",
    ],
)
def test_invalid_metric(metric):
    kwargs = {
        "recall": 0.5,
        "mrr": 0.5,
        "ndcg": 0.5,
        "faithfulness": 0.5,
        "citation_accuracy": 0.5,
        "latency_ms": 10.0,
    }

    kwargs[metric] = 1.5

    with pytest.raises(ValueError):
        create_ablation_result(
            "Dense",
            **kwargs,
        )


def test_negative_latency_rejected():
    with pytest.raises(ValueError):
        create_ablation_result(
            "Dense",
            recall=0.5,
            mrr=0.5,
            ndcg=0.5,
            faithfulness=0.5,
            citation_accuracy=0.5,
            latency_ms=-1.0,
        )


def test_compare_to_baseline():
    results = make_results()

    comparisons = compare_to_baseline(
        results,
        baseline_name="Dense",
    )

    assert len(comparisons) == 5

    assert comparisons[0].name == "Dense"
    assert comparisons[0].recall_delta == 0.0
    assert comparisons[0].mrr_delta == 0.0
    assert comparisons[0].latency_delta_ms == 0.0

    assert comparisons[-1].name == "Full System"
    assert comparisons[-1].recall_delta == pytest.approx(0.30)
    assert comparisons[-1].mrr_delta == pytest.approx(0.32)
    assert comparisons[-1].ndcg_delta == pytest.approx(0.33)
    assert comparisons[-1].faithfulness_delta == pytest.approx(0.22)
    assert comparisons[-1].citation_accuracy_delta == pytest.approx(0.20)
    assert comparisons[-1].latency_delta_ms == pytest.approx(35.0)


def test_missing_baseline():
    with pytest.raises(ValueError):
        compare_to_baseline(
            make_results(),
            baseline_name="Missing",
        )


def test_empty_comparison():
    assert compare_to_baseline([], "Dense") == []


def test_ablation_to_dict():
    result = make_results()[0]

    data = ablation_to_dict(result)

    assert data["name"] == "Dense"
    assert data["recall"] == 0.50
    assert data["mrr"] == 0.40
    assert data["ndcg"] == 0.45
    assert data["faithfulness"] == 0.60
    assert data["citation_accuracy"] == 0.70
    assert data["latency_ms"] == 20.0


def test_comparison_to_dict():
    comparison = compare_to_baseline(
        make_results(),
        "Dense",
    )[-1]

    data = comparison_to_dict(comparison)

    assert data["name"] == "Full System"
    assert data["recall_delta"] == pytest.approx(0.30)
    assert data["latency_delta_ms"] == pytest.approx(35.0)


def test_results_to_table():
    results = make_results()

    table = results_to_table(results)

    assert len(table) == 5
    assert table[0]["name"] == "Dense"
    assert table[-1]["name"] == "Full System"