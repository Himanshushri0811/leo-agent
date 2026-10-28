"""
Knowledge management endpoints (admin-protected).

POST /api/v1/knowledge/sources                - register a new knowledge source.
GET  /api/v1/knowledge/sources                - list knowledge sources by status.
POST /api/v1/knowledge/ingest                 - ingest a URL into the knowledge base.
POST /api/v1/knowledge/sources/{id}/approve   - approve an ingested source for live retrieval.

All endpoints require the X-API-Key header with the admin key.
Sources remain in REVIEW status until explicitly approved by an authorized user.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile

from app.core.security import verify_admin_api_key
from app.schemas.knowledge import (
    ApproveSourceRequest,
    IngestResponse,
    IngestURLRequest,
    KnowledgeSourceItem,
    RegisterSourceRequest,
    RegisterSourceResponse,
)
from app.services.knowledge_service import KnowledgeService, get_knowledge_service

router = APIRouter(
    prefix="/api/v1/knowledge",
    tags=["knowledge"],
    dependencies=[Depends(verify_admin_api_key)],
)


@router.post(
    "/sources",
    response_model=RegisterSourceResponse,
    status_code=201,
    summary="Register a knowledge source",
    description=(
        "Register a potential knowledge source for future ingestion. "
        "The source will be created in PENDING state. "
        "Requires admin API key."
    ),
)
async def register_source(
    body: RegisterSourceRequest,
    service: KnowledgeService = Depends(get_knowledge_service),
) -> RegisterSourceResponse:
    source = await service.register_source(
        name=body.name,
        source_url=str(body.source_url) if body.source_url else None,
        notes=body.notes,
    )
    return RegisterSourceResponse(
        source_id=source.id,
        name=source.name,
        status=source.status,
        message="Source registered. Use /ingest to process it.",
    )


@router.get(
    "/sources",
    response_model=list[KnowledgeSourceItem],
    summary="List knowledge sources",
    description="List all registered sources with optional status filtering.",
)
async def list_sources(
    status: str | None = Query(
        None, description="Filter by status (pending, review, approved, etc.)"
    ),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> list[KnowledgeSourceItem]:
    sources = await service.list_sources(status=status)
    return [
        KnowledgeSourceItem(
            id=s.id,
            name=s.name,
            source_type=s.source_type,
            source_url=s.source_url,
            status=s.status,
            version=s.version,
            created_at=s.created_at,
            approved_at=s.approved_at,
            approved_by=s.approved_by,
        )
        for s in sources
    ]


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest a URL",
    description=(
        "Fetch, parse, chunk, and embed a URL. "
        "The source will be placed in REVIEW status pending approval. "
        "Requires admin API key."
    ),
)
async def ingest(
    body: IngestURLRequest,
    service: KnowledgeService = Depends(get_knowledge_service),
) -> IngestResponse:
    result = await service.ingest_url_source(
        url=str(body.source_url),
        name=body.name,
    )

    return IngestResponse(
        source_id=result.source_id,
        source_name=result.source_name,
        chunks_created=result.chunks_created,
        is_duplicate=result.is_duplicate,
        status=result.status,
        message=(
            "Duplicate detected - source already exists."
            if result.is_duplicate
            else f"Ingested {result.chunks_created} chunks. Awaiting review and approval."
        ),
    )


@router.post(
    "/ingest-file",
    response_model=IngestResponse,
    summary="Ingest a document file",
    description=(
        "Upload, parse, chunk, and embed a document file (Markdown, TXT, PDF, DOCX). "
        "The source will be placed in REVIEW status (or APPROVED if auto_approve=True). "
        "Requires admin API key."
    ),
)
async def ingest_file(
    file: UploadFile = File(..., description="Document file (.md, .txt, .pdf, .docx)"),
    name: str | None = Form(None, description="Custom name for the knowledge source"),
    auto_approve: bool = Form(False, description="Whether to immediately approve this source"),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> IngestResponse:
    filename = file.filename or "uploaded_document"
    doc_name = name or Path(filename).stem.replace("_", " ").title()

    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        content = await file.read()
        tmp.write(content)

    try:
        result = await service.ingest_file_source(
            path=tmp_path,
            name=doc_name,
            auto_approve=auto_approve,
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return IngestResponse(
        source_id=result.source_id,
        source_name=result.source_name,
        chunks_created=result.chunks_created,
        is_duplicate=result.is_duplicate,
        status=result.status,
        message=(
            "Duplicate detected - identical document content already exists."
            if result.is_duplicate
            else (
                f"Ingested and approved {result.chunks_created} chunks."
                if result.status == "approved"
                else f"Ingested {result.chunks_created} chunks. Awaiting review and approval."
            )
        ),
    )


@router.post(
    "/sources/{source_id}/approve",
    response_model=KnowledgeSourceItem,
    summary="Approve a knowledge source",
    description=(
        "Transition a source from REVIEW to APPROVED status, enabling it "
        "for semantic vector retrieval in customer conversations. "
        "Requires admin API key."
    ),
)
async def approve_source(
    source_id: uuid.UUID,
    body: ApproveSourceRequest,
    service: KnowledgeService = Depends(get_knowledge_service),
) -> KnowledgeSourceItem:
    source = await service.approve_source(
        source_id=source_id,
        approved_by=body.approved_by,
    )
    return KnowledgeSourceItem(
        id=source.id,
        name=source.name,
        source_type=source.source_type,
        source_url=source.source_url,
        status=source.status,
        version=source.version,
        created_at=source.created_at,
        approved_at=source.approved_at,
        approved_by=source.approved_by,
    )
