from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from graph.schema import NodeType, RelationType


class GraphNode(BaseModel):
    id: str
    label: NodeType
    name: str
    file_path: Optional[str] = None
    language: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    type: RelationType
    properties: Dict[str, Any] = Field(default_factory=dict)