import json

import pytest

from evaluation.results import EvaluationResultsStore


def test_save_result(tmp_path):
    store = EvaluationResultsStore(tmp_path)

    result = {
        "dataset": {
            "size": 31,
            "version": "v1",
        },
        "metrics": {
            "recall": 0.8,
            "mrr": 0.7,
        },
    }

    path = store.save("baseline", result)

    assert path.exists()
    assert path.name == "baseline.json"

    loaded_json = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert loaded_json == result


def test_save_with_json_extension(tmp_path):
    store = EvaluationResultsStore(tmp_path)

    path = store.save(
        "hybrid.json",
        {"recall": 0.9},
    )

    assert path.name == "hybrid.json"


def test_load_result(tmp_path):
    store = EvaluationResultsStore(tmp_path)

    result = {
        "metrics": {
            "recall": 0.75,
            "mrr": 0.65,
        }
    }

    store.save("hybrid", result)

    loaded = store.load("hybrid")

    assert loaded == result


def test_load_missing_result(tmp_path):
    store = EvaluationResultsStore(tmp_path)

    with pytest.raises(FileNotFoundError):
        store.load("missing")


def test_load_invalid_json_object(tmp_path):
    store = EvaluationResultsStore(tmp_path)

    path = tmp_path / "invalid.json"

    path.write_text(
        "[1, 2, 3]",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        store.load("invalid")


def test_exists(tmp_path):
    store = EvaluationResultsStore(tmp_path)

    assert not store.exists("baseline")

    store.save(
        "baseline",
        {"test": True},
    )

    assert store.exists("baseline")
    assert store.exists("baseline.json")


def test_list_results(tmp_path):
    store = EvaluationResultsStore(tmp_path)

    store.save("final", {})
    store.save("baseline", {})
    store.save("hybrid", {})

    assert store.list_results() == [
        "baseline.json",
        "final.json",
        "hybrid.json",
    ]


def test_delete_result(tmp_path):
    store = EvaluationResultsStore(tmp_path)

    store.save(
        "baseline",
        {"test": True},
    )

    assert store.exists("baseline")

    store.delete("baseline")

    assert not store.exists("baseline")


def test_delete_missing_result(tmp_path):
    store = EvaluationResultsStore(tmp_path)

    store.delete("does-not-exist")


def test_invalid_save_name(tmp_path):
    store = EvaluationResultsStore(tmp_path)

    with pytest.raises(ValueError):
        store.save("", {})


def test_invalid_load_name(tmp_path):
    store = EvaluationResultsStore(tmp_path)

    with pytest.raises(ValueError):
        store.load("")


def test_nested_directory_is_created(tmp_path):
    results_path = tmp_path / "nested" / "evaluation"

    store = EvaluationResultsStore(results_path)

    store.save(
        "final",
        {"complete": True},
    )

    assert (results_path / "final.json").exists()