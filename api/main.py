from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from application.service import CodeGraphRAGService, RepositoryNotIndexedError


class IndexRequest(BaseModel):
    repository_path: str = Field(min_length=1)
    repository_name: str | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)


def create_app(service: CodeGraphRAGService | None = None) -> FastAPI:
    """Create the REST application; expensive runtime clients are lazy."""
    app = FastAPI(title="CodeGraphRAG API", version="1.0.0")
    active_service = service

    def get_service() -> CodeGraphRAGService:
        nonlocal active_service
        if active_service is None:
            active_service = CodeGraphRAGService()
        return active_service

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/repositories/index")
    def index_repository(request: IndexRequest) -> dict[str, Any]:
        try:
            summary = get_service().index_repository(
                request.repository_path,
                request.repository_name,
            )
        except (NotADirectoryError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return asdict(summary)

    @app.post("/search")
    def search(request: QueryRequest) -> list[dict[str, Any]]:
        try:
            results = get_service().search(request.query)
        except RepositoryNotIndexedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return [asdict(result) for result in results]

    @app.post("/answers")
    def answer(request: QueryRequest) -> dict[str, Any]:
        try:
            result = get_service().answer(request.query)
        except RepositoryNotIndexedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "answer": result.answer,
            "citations": [asdict(citation) for citation in result.citations],
        }

    return app


app = create_app()
