from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from application.service import (
    CodeGraphRAGService,
    NoChunksGeneratedError,
    RepositoryNotIndexedError,
)

REPOSITORY_ROOT_ENV = "CODEGRAPHRAG_API_REPOSITORY_ROOT"


class IndexRequest(BaseModel):
    repository_path: str = Field(min_length=1)
    repository_name: str | None = None
    exclude: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    repository_name: str | None = None


def create_app(
    service: CodeGraphRAGService | None = None,
    repository_root: str | Path | None = None,
) -> FastAPI:
    """Create the REST application; expensive runtime clients are lazy.

    Only directories inside ``repository_root`` can be indexed. It defaults to
    $CODEGRAPHRAG_API_REPOSITORY_ROOT, or the current working directory.
    """
    app = FastAPI(title="CodeGraphRAG API", version="1.0.0")
    active_service = service
    allowed_root = Path(
        repository_root or os.getenv(REPOSITORY_ROOT_ENV) or Path.cwd()
    ).resolve()

    def get_service() -> CodeGraphRAGService:
        nonlocal active_service
        if active_service is None:
            active_service = CodeGraphRAGService()
        return active_service

    def resolve_repository_path(raw_path: str) -> str:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_relative_to(allowed_root):
            raise HTTPException(
                status_code=403,
                detail="repository_path is outside the allowed repository root",
            )
        return str(path)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/repositories/index")
    def index_repository(request: IndexRequest) -> dict[str, Any]:
        repository_path = resolve_repository_path(request.repository_path)
        try:
            summary = get_service().index_repository(
                repository_path,
                request.repository_name,
                exclude=request.exclude,
            )
        except (NotADirectoryError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except NoChunksGeneratedError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return asdict(summary)

    @app.post("/search")
    def search(request: QueryRequest) -> list[dict[str, Any]]:
        try:
            results = get_service().search(request.query, request.repository_name)
        except RepositoryNotIndexedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return [asdict(result) for result in results]

    @app.post("/answers")
    def answer(request: QueryRequest) -> dict[str, Any]:
        try:
            result = get_service().answer(request.query, request.repository_name)
        except RepositoryNotIndexedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "answer": result.answer,
            "citations": [asdict(citation) for citation in result.citations],
        }

    return app


app = create_app()