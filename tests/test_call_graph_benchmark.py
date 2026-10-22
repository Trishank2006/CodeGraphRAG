from evaluation.call_graph_benchmark import Case, evaluate, summarize


class FakeStore:
    def __init__(self, rows, exists=True):
        self.rows = rows
        self.exists = exists

    def execute_query(self, query, parameters=None):
        if "count(n)" in query:
            return [{"count": 1 if self.exists else 0}]
        return self.rows


def _case(*expected):
    return Case("callees", "a.py", "run", frozenset(expected))


def test_evaluate_separates_found_missing_unexpected_and_unresolved():
    store = FakeStore([
        {"name": "helper", "file_path": "a.py", "reference": False},
        {"name": "other", "file_path": "c.py", "reference": False},
        {"name": "open", "file_path": "", "reference": True},
    ])

    result = evaluate(store, _case(("helper", "a.py"), ("load", "b.py")))

    assert result["found"] == [["helper", "a.py"]]
    assert result["missing"] == [["load", "b.py"]]
    assert result["unexpected"] == [["other", "c.py"]]
    assert result["unresolved"] == 1


def test_evaluate_reports_a_missing_node():
    result = evaluate(FakeStore([], exists=False), _case(("helper", "a.py")))

    assert result["error"] == "node not found"
    assert result["missing"] == [["helper", "a.py"]]


def test_summarize_computes_micro_precision_and_recall():
    results = [
        {
            "direction": "callees",
            "expected": [["a", "x"], ["b", "x"]],
            "found": [["a", "x"]],
            "unexpected": [["c", "x"]],
            "unresolved": 2,
        },
        {
            "direction": "callers",
            "expected": [["d", "y"]],
            "found": [["d", "y"]],
            "unexpected": [],
            "unresolved": 0,
        },
    ]

    summary = summarize(results)

    assert summary["precision"] == round(2 / 3, 4)
    assert summary["recall"] == round(2 / 3, 4)
    assert summary["unresolved_targets"] == 2
