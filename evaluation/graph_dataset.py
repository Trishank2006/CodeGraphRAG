from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class GraphEvaluationExample:
    query: str
    query_type: str  # "callers", "callees", "dependencies", "inheritance", "path"
    root_symbol: str
    expected_entities: List[str] = field(default_factory=list)
    expected_edges: List[Tuple[str, str, str]] = field(default_factory=list)
    expected_path: List[str] = field(default_factory=list)
    target_symbol: Optional[str] = None


def create_graph_evaluation_dataset() -> List[GraphEvaluationExample]:
    """Curated structural evaluation dataset for CodeGraphRAG."""
    return [
        GraphEvaluationExample(
            query="Who calls validate_token?",
            query_type="callers",
            root_symbol="validate_token",
            expected_entities=["login", "authenticate_request"],
            expected_edges=[
                ("login", "CALLS", "validate_token"),
                ("authenticate_request", "CALLS", "validate_token"),
            ],
        ),
        GraphEvaluationExample(
            query="What does AuthService.login call?",
            query_type="callees",
            root_symbol="login",
            expected_entities=["validate_token", "get_user_by_email"],
            expected_edges=[
                ("login", "CALLS", "validate_token"),
                ("login", "CALLS", "get_user_by_email"),
            ],
        ),
        GraphEvaluationExample(
            query="What files does api/routers/auth.py import?",
            query_type="dependencies",
            root_symbol="api/routers/auth.py",
            expected_entities=["services/auth_service.py", "schemas/auth.py"],
            expected_edges=[
                ("api/routers/auth.py", "IMPORTS", "services/auth_service.py"),
                ("api/routers/auth.py", "IMPORTS", "schemas/auth.py"),
            ],
        ),
        GraphEvaluationExample(
            query="Which classes inherit from BaseRepository?",
            query_type="inheritance",
            root_symbol="BaseRepository",
            expected_entities=["UserRepository", "MeetingRepository"],
            expected_edges=[
                ("UserRepository", "INHERITS", "BaseRepository"),
                ("MeetingRepository", "INHERITS", "BaseRepository"),
            ],
        ),
        GraphEvaluationExample(
            query="Trace execution path from upload_meeting to transcribe",
            query_type="path",
            root_symbol="upload_meeting",
            target_symbol="transcribe",
            expected_path=["upload_meeting", "process_meeting_pipeline", "transcribe"],
            expected_edges=[
                ("upload_meeting", "CALLS", "process_meeting_pipeline"),
                ("process_meeting_pipeline", "CALLS", "transcribe"),
            ],
        ),
    ]