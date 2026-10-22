from graph.builder import GraphBuilder
from parser.core_parser import parse_source_code

SOURCE = '''
class Service:
    def run(self):
        self.helper()
        pipeline.search()
        build()

    def helper(self):
        def nested():
            inner()
        nested()


class Other:
    def run(self):
        pass
'''


def _calls(parsed):
    return {
        (r.source_id, r.target_id)
        for r in parsed.relationships
        if r.type == "CALLS"
    }


def test_parser_keeps_call_receivers():
    parsed = parse_source_code("svc.py", SOURCE, "python")

    targets = {target for _, target in _calls(parsed)}

    assert "symbol:self.helper" in targets
    assert "symbol:pipeline.search" in targets
    assert "symbol:build" in targets


def test_parser_attributes_calls_to_the_innermost_function():
    parsed = parse_source_code("svc.py", SOURCE, "python")

    sources = {source for source, target in _calls(parsed) if target == "symbol:inner"}

    assert sources == {"svc.py:function:nested"}


def test_parser_gives_duplicate_names_distinct_ids():
    parsed = parse_source_code("svc.py", SOURCE, "python")

    run_ids = [e.id for e in parsed.entities if e.name == "run"]

    assert len(run_ids) == 2
    assert len(set(run_ids)) == 2
    assert "svc.py:function:run" in run_ids


def test_self_calls_resolve_end_to_end():
    parsed = parse_source_code("svc.py", SOURCE, "python")

    _, edges = GraphBuilder("demo").build_from_parsed_files([parsed])
    calls = {
        (e.source_id, e.target_id)
        for e in edges
        if e.type.value == "CALLS"
    }

    assert (
        "repo:demo:svc.py:function:run",
        "repo:demo:svc.py:function:helper",
    ) in calls
