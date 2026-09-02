from pydantic import BaseModel
from typing import List

class CodeEntity(BaseModel):
    id: str               
    type: str             
    name: str
    file_path: str
    start_line: int
    end_line: int

class CodeRelationship(BaseModel):
    source_id: str
    target_id: str
    type: str             

class ParsedFile(BaseModel):
    file_path: str
    language: str
    entities: List[CodeEntity]
    relationships: List[CodeRelationship]