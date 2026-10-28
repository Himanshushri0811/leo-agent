"""
CLI Command for indexing Bitmax Group company knowledge documents into PostgreSQL + pgvector.

Usage:
    # Ingest all files in the company knowledge folder and approve them
    python scripts/ingest_knowledge.py --dir knowledge_sources/bitmax_group --approve

    # Ingest a single document file
    python scripts/ingest_knowledge.py --file path/to/doc.md --name "Custom Source Name" --approve
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.providers import get_embeddings_provider
from app.core.config import get_settings
from app.db.session import dispose_db, init_db, session_context
from app.knowledge.ingestion import ingest_document, reindex_all_sources

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf", ".docx"}


async def process_file(
    path: Path,
    name: str | None,
    auto_approve: bool,
    session,
    embeddings_provider,
    settings,
) -> None:
    if not path.is_file():
        print(f"[-] Skipping non-file path: {path}")
        return

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"[-] Skipping unsupported file extension ({path.suffix}): {path.name}")
        return

    doc_name = name or path.stem.replace("_", " ").title()
    try:
        result = await ingest_document(
            path=path,
            name=doc_name,
            session=session,
            embeddings_provider=embeddings_provider,
            settings=settings,
            auto_approve=auto_approve,
        )

        if result.is_duplicate:
            print(
                f"[=] DUPLICATE: '{result.source_name}' (ID: {result.source_id}) "
                f"— identical content already indexed, no new chunks created."
            )
        else:
            print(
                f"[+] INDEXED: '{result.source_name}' (ID: {result.source_id}) "
                f"— {result.chunks_created} chunks stored | Status: {result.status}"
            )
    except Exception as exc:
        print(f"[!] FAILED to ingest '{path.name}': {exc}", file=sys.stderr)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index company knowledge documents into PostgreSQL with pgvector embeddings."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", "-f", type=Path, help="Path to a single document to index.")
    group.add_argument(
        "--dir", "-d", type=Path, help="Directory containing documents to index recursively."
    )
    group.add_argument(
        "--reindex",
        "-r",
        action="store_true",
        help="Re-index all existing knowledge sources with current embeddings provider.",
    )
    parser.add_argument(
        "--name", "-n", type=str, default=None, help="Custom source name (only with --file)."
    )
    parser.add_argument(
        "--approve",
        "-a",
        action="store_true",
        default=False,
        help="Mark ingested sources as APPROVED for immediate retrieval.",
    )

    args = parser.parse_args()

    settings = get_settings()
    init_db()
    embeddings_provider = get_embeddings_provider(settings)

    print("=" * 70)
    print("LEO AI Knowledge Ingestion CLI")
    print(f"Database: PostgreSQL ({settings.database_url.split('@')[-1]})")
    print(f"Embeddings: {settings.embeddings_provider} ({settings.embeddings_dimensions} dims)")
    print(f"Auto-Approve: {args.approve}")
    print("=" * 70)

    try:
        async with session_context() as session:
            if args.reindex:
                print(f"[*] Re-indexing all existing database sources using {embeddings_provider.provider_name}...")
                res = await reindex_all_sources(session=session, embeddings_provider=embeddings_provider, settings=settings)
                print(
                    f"[+] Re-indexing complete: {res['sources_reindexed']} sources reindexed, "
                    f"{res['chunks_reembedded']} chunks re-embedded."
                )
            elif args.file:
                target_file = args.file.resolve()
                if not target_file.exists():
                    print(f"[!] File not found: {target_file}", file=sys.stderr)
                    sys.exit(1)
                await process_file(
                    path=target_file,
                    name=args.name,
                    auto_approve=args.approve,
                    session=session,
                    embeddings_provider=embeddings_provider,
                    settings=settings,
                )
            elif args.dir:
                target_dir = args.dir.resolve()
                if not target_dir.exists():
                    print(f"[!] Directory not found: {target_dir}", file=sys.stderr)
                    sys.exit(1)

                files = sorted(
                    f
                    for f in target_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
                )
                if not files:
                    print(f"[!] No supported files found in: {target_dir}")
                    return

                print(f"Processing {len(files)} files from {target_dir}...")
                for file_path in files:
                    await process_file(
                        path=file_path,
                        name=None,
                        auto_approve=args.approve,
                        session=session,
                        embeddings_provider=embeddings_provider,
                        settings=settings,
                    )

        print("=" * 70)
        print("Ingestion run completed successfully.")
    finally:
        await dispose_db()


if __name__ == "__main__":
    asyncio.run(main())
