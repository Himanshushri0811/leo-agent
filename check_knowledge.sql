SELECT
    COUNT(*) AS total_sources
FROM knowledge_sources;

SELECT
    status,
    COUNT(*) AS source_count
FROM knowledge_sources
GROUP BY status;

SELECT
    COUNT(*) AS total_chunks,
    COUNT(embedding) AS chunks_with_embeddings,
    COUNT(*) FILTER (WHERE embedding IS NULL) AS chunks_without_embeddings
FROM knowledge_chunks;
