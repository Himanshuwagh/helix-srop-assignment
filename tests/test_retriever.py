"""
Unit tests for RAG retrieval.
Requires the vector store to be seeded first (run ingest.py on docs/).
"""
import pytest

from app.rag.ingest import chunk_markdown


@pytest.mark.asyncio
async def test_search_docs_returns_results_with_chunk_ids(monkeypatch):
    """search_docs must return chunk IDs and scores in [0, 1]."""
    from app.agents.tools.search_docs import search_docs

    class FakeCollection:
        def query(self, query_embeddings: list[list[float]], n_results: int, where=None):
            return {
                "ids": [["chunk_a", "chunk_b", "chunk_c"]],
                "documents": [["doc a", "doc b", "doc c"]],
                "metadatas": [[{"source": "a"}, {"source": "b"}, {"source": "c"}]],
                "distances": [[0.1, 0.2, 0.4]],
            }

    monkeypatch.setattr("app.agents.tools.search_docs._get_collection", lambda: FakeCollection())
    monkeypatch.setattr("app.agents.tools.search_docs._embed_query_sync", lambda query: [0.1, 0.2])

    results = await search_docs("how to rotate a deploy key", k=3)
    assert len(results) > 0
    assert all(result.chunk_id for result in results)
    assert all(0.0 <= result.score <= 1.0 for result in results)


def test_chunker_produces_non_empty_chunks():
    """Chunker must not produce empty strings."""
    text = "# Header\n\nSome content.\n\n## Section 2\n\nMore content here."
    chunks = chunk_markdown(text, chunk_size=40, overlap=10)
    assert len(chunks) > 0
    assert all(chunk.strip() for chunk in chunks)
