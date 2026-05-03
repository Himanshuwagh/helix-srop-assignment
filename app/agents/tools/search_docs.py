"""
search_docs tool — used by KnowledgeAgent.

Queries the vector store for relevant documentation chunks.
Returns chunk IDs, scores, and content so the agent can cite sources.

TODO for candidate: implement this tool.
Wire it to your chosen vector store (Chroma, LanceDB, FAISS, etc.).
"""
import asyncio
from dataclasses import dataclass

from app.settings import settings


@dataclass
class DocChunk:
    chunk_id: str
    score: float
    content: str
    metadata: dict  # e.g. {"product_area": "security", "source": "deploy-keys.md"}


async def search_docs(query: str, k: int = 5, product_area: str | None = None) -> list[DocChunk]:
    """
    Search the vector store for top-k relevant chunks.

    Args:
        query: natural language query from the user
        k: number of chunks to return
        product_area: optional metadata filter (e.g. "security", "ci-cd")

    Returns:
        List of DocChunk ordered by descending similarity score.

    Design considerations:
    - How do you embed the query? Same model as at ingest time.
    - Do you apply a score threshold to filter low-quality results?
    - How do you format chunks for the agent? Include chunk_id so agent can cite.
    """
    collection = await asyncio.to_thread(_get_collection)
    query_embedding = await asyncio.to_thread(_embed_query_sync, query)
    where = {"product_area": product_area} if product_area else None
    result = await asyncio.to_thread(
        collection.query,
        query_embeddings=[query_embedding],
        n_results=k,
        where=where,
    )

    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    chunks: list[DocChunk] = []
    for index, chunk_id in enumerate(ids):
        distance = float(distances[index]) if index < len(distances) else 1.0
        chunks.append(
            DocChunk(
                chunk_id=chunk_id,
                score=max(0.0, min(1.0, 1.0 - distance)),
                content=documents[index] if index < len(documents) else "",
                metadata=metadatas[index] if index < len(metadatas) else {},
            )
        )
    return chunks


def _get_collection():
    import chromadb

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client.get_or_create_collection(
        name="helix_docs",
        metadata={"hnsw:space": "cosine"},
    )


def _embed_query_sync(query: str) -> list[float]:
    import google.generativeai as genai

    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is required for query embeddings")
    genai.configure(api_key=settings.google_api_key)
    result = genai.embed_content(
        model=settings.gemini_embedding_model,
        content=query,
        task_type="retrieval_query",
    )
    embedding = result["embedding"]
    if embedding and isinstance(embedding[0], list):
        return embedding[0]
    return embedding
