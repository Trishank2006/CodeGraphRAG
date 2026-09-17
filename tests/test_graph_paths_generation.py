from unittest.mock import MagicMock
from graph.path import GraphPathFinder


def test_get_structured_path():
    mock_store = MagicMock()
    mock_store.execute_query.return_value = [
        {
            "nodes": [
                {
                    "id": "1",
                    "name": "upload_meeting",
                    "label": "Function",
                    "file_path": "routers/meetings.py",
                    "start_line": 53,
                    "end_line": 80,
                },
                {
                    "id": "2",
                    "name": "process_meeting_pipeline",
                    "label": "Function",
                    "file_path": "services/pipeline.py",
                    "start_line": 52,
                    "end_line": 149,
                },
            ],
            "relationships": ["CALLS"],
            "path_length": 1,
        }
    ]

    finder = GraphPathFinder(mock_store)
    path = finder.get_path("upload_meeting", "process_meeting_pipeline")

    assert path is not None
    assert path.path_length == 1
    assert path.steps[0].source.name == "upload_meeting"
    assert path.steps[0].relationship == "CALLS"
    assert path.steps[0].target.name == "process_meeting_pipeline"