from unittest.mock import MagicMock
from graph.context import GraphContextExtractor


def test_format_hierarchy_tree():
    mock_store = MagicMock()
    mock_store.execute_query.return_value = [
        {
            "root_id": "fn:process",
            "root_name": "process_meeting_pipeline",
            "root_type": "Function",
            "root_file": "services/pipeline.py",
            "root_start": 52,
            "root_end": 149,
            "connections": [
                {
                    "target_id": "fn:transcribe",
                    "target_name": "transcribe",
                    "target_file": "services/asr/whisper_service.py",
                    "target_start": 120,
                    "target_end": 155,
                    "relationship": "CALLS",
                    "distance": 1,
                },
                {
                    "target_id": "fn:get_db",
                    "target_name": "get_db",
                    "target_file": "database.py",
                    "target_start": 16,
                    "target_end": 21,
                    "relationship": "CALLS",
                    "distance": 1,
                },
            ],
        }
    ]

    extractor = GraphContextExtractor(mock_store)
    tree = extractor.format_hierarchy("process_meeting_pipeline")

    assert "process_meeting_pipeline" in tree
    assert "├── CALLS → transcribe" in tree
    assert "└── CALLS → get_db" in tree