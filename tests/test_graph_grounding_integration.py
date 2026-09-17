from unittest.mock import MagicMock
from retrieval.models import RetrievalResult
from generation.context import ContextBuilder
from graph.grounding import GraphGroundingService


def test_graph_grounding_into_context_builder():
    """Validates Person 2's GraphGroundingService cleanly integrates into Person 1's ContextBuilder."""
    mock_store = MagicMock()
    mock_store.execute_query.return_value = [
        {
            "root_id": "fn:upload_meeting",
            "root_name": "upload_meeting",
            "root_file": "routers/meetings.py",
            "root_start": 53,
            "root_end": 80,
            "connections": [
                {
                    "target_id": "fn:process_meeting_pipeline",
                    "target_name": "process_meeting_pipeline",
                    "target_file": "services/pipeline.py",
                    "relationship": "CALLS",
                    "distance": 1,
                }
            ],
        }
    ]

    grounding_service = GraphGroundingService(mock_store)
    results = [
        RetrievalResult(
            id="fn:upload_meeting",
            source="vector",
            score=0.9,
            repository="repo",
            file_path="routers/meetings.py",
            language="python",
            symbol="upload_meeting",
            start_line=53,
            end_line=80,
            content="def upload_meeting(): pass",
        )
    ]

    # 1. Verify GraphGroundingService extracts evidence and tree formatting
    evidence = grounding_service.get_evidence("upload_meeting")
    assert len(evidence) == 1
    assert evidence[0].relationship == "CALLS"
    assert evidence[0].target_entity == "process_meeting_pipeline"

    tree = grounding_service.format_context_tree("upload_meeting")
    assert "upload_meeting" in tree
    assert "CALLS → process_meeting_pipeline" in tree

    # 2. Pass grounding_service.get_context directly into Person 1's graph_context_getter
    builder = ContextBuilder(graph_context_getter=grounding_service.get_context)
    pkg = builder.build(query="How does meeting upload work?", results=results)

    assert len(pkg.items) == 1
    assert pkg.items[0].symbol == "upload_meeting"
    assert len(pkg.graph_context) == 1
    assert "upload_meeting --CALLS--> process_meeting_pipeline" in pkg.graph_context[0]