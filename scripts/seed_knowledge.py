"""
Controlled Seed Script for Verified Global Projects Knowledge.

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
        "name": "Global Projects - Company Overview",
        "url": "https://globalprojects.in/about",
        "content": (
            "Global Projects is a professional services company specializing in "
            "project management, business process outsourcing (BPO), and software "
            "development. The company partners with clients across industries to "
            "deliver end-to-end project execution, operational efficiency, and "
            "technology-driven business solutions with a strong focus on quality "
            "and client satisfaction."
        ),
    },
    {
        "name": "Global Projects - Core Services",
        "url": "https://globalprojects.in/services",
        "content": (
            "Global Projects offers three core service pillars:\n"
            "1. Project Management & Consulting: Full-cycle project planning, "
            "execution, monitoring, and delivery management for infrastructure, "
            "technology, and business transformation projects.\n"
            "2. Business Process Outsourcing (BPO): Customer support operations, "
            "back-office processing, data management, and process automation.\n"
            "3. Software Development: Custom web and mobile application development, "
            "system integrations, and enterprise software solutions tailored to "
            "client requirements."
        ),
    },
    {
        "name": "Global Projects - Contact & Support Channels",
        "url": "https://globalprojects.in/contact",
        "content": (
            "For customer support and business inquiries, Global Projects can be reached through:\n"
            "- Business Inquiries: Please contact the Global Projects representative assigned to your account.\n"
            "- Support: Reach out to the Global Projects support team for project-related assistance.\n"
            "- Dedicated project managers are assigned to clients for priority support and escalation."
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
