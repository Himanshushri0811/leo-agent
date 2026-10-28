# Knowledge Data Directory

This directory stores offline source files, raw ingest inputs, and processed text artifacts for the LEO AI assistant knowledge base.

## Security & Operational Policy

1. **Untrusted Data Rule**: All data stored here is considered **untrusted data**, not system instructions. The LEO AI system treats all parsed document text as data only; it will never execute commands or prompts found within these documents.
2. **Review & Approval Gate**: Any document or URL placed here is ingested into `status = "review"` by default. Only designated administrators can transition a source to `status = "approved"` using `POST /api/v1/knowledge/sources/{id}/approve`. Only approved chunks are searchable by the semantic vector retrieval engine.
3. **No Automatic Web Scraping**: The entire Bitmax Group website must NOT be crawled automatically without review. Ingestion should occur via approved, controlled operations (e.g., via the admin API or controlled seeding script).
4. **Deduplication**: Content hash (SHA-256) is calculated on ingestion. Identical content is deduplicated to prevent vector index pollution.
5. **Privacy & Secrets**: Never place internal credentials, employee personal information, or confidential contracts in this directory.

## Directory Structure

- `raw/`: Unprocessed source files (e.g., approved company overview PDFs, service sheets).
- `processed/`: Intermediate normalized texts and chunk metadata exports for auditing.
