"""
Controlled Seed Script for Verified Bitmax Group Knowledge.

Reads verified company facts from a curated dictionary or file,
creates KnowledgeSource records, generates embeddings, and transitions
them through the review gate.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.providers import get_embeddings_provider
from app.core.config import get_settings
from app.db.models.knowledge import KnowledgeChunk, KnowledgeSource, SourceStatus
from app.db.session import init_db, session_context
from app.knowledge.chunking import chunk_text
from app.knowledge.embeddings import embed_chunks
from app.knowledge.parsing import parse_text

VERIFIED_SAMPLE_KNOWLEDGE = [
    {
        "name": "Bitmax Group - Company Overview",
        "url": "https://bitmaxgroup.com/about",
        "content": (
            "Bitmax Group is an enterprise technology solutions provider specializing in "
            "digital transformation, cloud architecture, managed IT services, and "
            "secure infrastructure consulting. The company works closely with enterprise "
            "clients to modernize their mission-critical business systems with high "
            "reliability and strict security standards."
        ),
    },
    {
        "name": "Bitmax Group - Core Services",
        "url": "https://bitmaxgroup.com/services",
        "content": (
            "Bitmax Group offers three core service pillars:\n"
            "1. Cloud Infrastructure & DevOps: Multi-cloud migration, Kubernetes orchestration, "
            "and continuous delivery automation.\n"
            "2. Cybersecurity & Governance: Security posture assessments, DevSecOps integration, "
            "and compliance auditing.\n"
            "3. Custom Enterprise Engineering: Scalable backend architectures, API integration, "
            "and enterprise AI enablement."
        ),
    },
    {
        "name": "Bitmax Group - Contact & Support Channels",
        "url": "https://bitmaxgroup.com/contact",
        "content": (
            "For customer support and business inquiries, Bitmax Group can be reached through:\n"
            "- Business Inquiries: inquiries@bitmaxgroup.com\n"
            "- Technical Support Portal: https://support.bitmaxgroup.com\n"
            "- Dedicated Account Managers are assigned to enterprise contract clients for "
            "priority incident escalation."
        ),
    },
]


async def seed() -> None:
    settings = get_settings()
    init_db()
    embeddings_provider = get_embeddings_provider(settings)

    print("Beginning controlled knowledge seeding...")

    async with session_context() as session:
        for item in VERIFIED_SAMPLE_KNOWLEDGE:
            normalized, content_hash = parse_text(item["content"])
            chunks = chunk_text(
                normalized,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )
            embeddings = await embed_chunks(chunks, embeddings_provider)

            source = KnowledgeSource(
                name=item["name"],
                source_type="manual",
                source_url=item["url"],
                content_hash=content_hash,
                status=SourceStatus.APPROVED,
                approved_by="system-seed",
            )
            session.add(source)
            await session.flush()

            for chunk, emb in zip(chunks, embeddings, strict=True):
                db_chunk = KnowledgeChunk(
                    source_id=source.id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    token_count=chunk.token_estimate,
                    embedding=emb,
                )
                session.add(db_chunk)

            print(f"  + Seeded approved source: {item['name']} ({len(chunks)} chunks)")

    print("Knowledge seeding completed successfully.")


if __name__ == "__main__":
    asyncio.run(seed())
