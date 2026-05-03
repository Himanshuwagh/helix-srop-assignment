"""
RAG ingest CLI.

Usage:
    python -m app.rag.ingest --path docs/
    python -m app.rag.ingest --path docs/ --chunk-size 512 --chunk-overlap 64

Reads markdown files, chunks them, embeds, and writes to the vector store.

TODO for candidate: implement chunking and embedding logic.
"""
import argparse
import asyncio
import hashlib
import re
import time
from pathlib import Path

from app.settings import settings


def chunk_markdown(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Split markdown text into overlapping chunks.

    Design considerations:
    - Simple character splitting is fast but breaks mid-sentence.
    - Sentence-aware splitting is better for retrieval quality.
    - Heading-aware splitting (split on ## / ###) keeps sections coherent.
    - Overlap helps preserve context at chunk boundaries.

    Choose an approach and document why in the README.
    """
    sections = re.split(r"\n(?=#{2,3} )", text)
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for section in sections:
        clean_section = section.strip()
        if not clean_section:
            continue
        if len(clean_section) <= chunk_size:
            chunks.append(clean_section)
            continue
        start = 0
        while start < len(clean_section):
            piece = clean_section[start : start + chunk_size].strip()
            if piece:
                chunks.append(piece)
            start += step
    return chunks


def extract_metadata(file_path: Path, text: str) -> dict:
    """
    Extract metadata from a markdown file's frontmatter.

    Expected frontmatter format:
        ---
        title: Deploy Keys
        product_area: security
        tags: [keys, secrets]
        ---

    Returns a dict suitable for vector store metadata filtering.
    """
    metadata: dict[str, str] = {
        "source": file_path.name,
        "path": str(file_path),
    }
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if match is None:
        return metadata
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key in {"product_area", "title"} and value:
            metadata[key] = value
    return metadata


async def ingest_directory(docs_path: Path, chunk_size: int, chunk_overlap: int) -> None:
    """
    Walk docs_path, chunk and embed every .md file, upsert into vector store.

    Design considerations:
    - Generate a stable chunk_id (e.g. sha256(file + chunk_index)) for deduplication.
    - Run embeddings in batches to avoid rate limiting.
    - Print progress so the user can see what's happening.
    """
    md_files = _collect_md_files(docs_path)
    print(f"Found {len(md_files)} markdown files in {docs_path}")

    for file_path in md_files:
        text = file_path.read_text(encoding="utf-8")
        metadata = extract_metadata(file_path, text)
        chunks = chunk_markdown(_strip_frontmatter(text), chunk_size, chunk_overlap)
        print(f"  {file_path.name}: {len(chunks)} chunks")
        if not chunks:
            continue

        relative_path = _relative_to_root(file_path, docs_path)
        chunk_ids = [_make_chunk_id(relative_path, chunk_index) for chunk_index in range(len(chunks))]
        embeddings = await _embed_documents_throttled(chunks)
        metadatas = [
            {
                **metadata,
                "path": relative_path,
                "chunk_index": chunk_index,
            }
            for chunk_index in range(len(chunks))
        ]
        collection = await asyncio.to_thread(_get_collection)
        await asyncio.to_thread(
            collection.upsert,
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

    print("Ingest complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs into the vector store")
    parser.add_argument("--path", type=Path, required=True, help="Directory containing .md files")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    args = parser.parse_args()

    asyncio.run(ingest_directory(args.path, args.chunk_size, args.chunk_overlap))


def _strip_frontmatter(text: str) -> str:
    return re.sub(r"^---\n.*?\n---\n", "", text, count=1, flags=re.DOTALL)


def _make_chunk_id(relative_path: str, chunk_index: int) -> str:
    raw = f"{relative_path}::{chunk_index}"
    return "chunk_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _relative_to_root(file_path: Path, root: Path) -> str:
    if root.is_file():
        return root.name
    return str(file_path.relative_to(root))


def _collect_md_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() == ".md" else []
    return list(path.rglob("*.md"))


def _get_collection():
    import chromadb

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client.get_or_create_collection(
        name="helix_docs",
        metadata={"hnsw:space": "cosine"},
    )


def _embed_documents_sync(chunks: list[str]) -> list[list[float]]:
    import google.generativeai as genai

    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is required for document embeddings")
    genai.configure(api_key=settings.google_api_key)
    embeddings: list[list[float]] = []
    batch_size = 20
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        result = genai.embed_content(
            model=settings.gemini_embedding_model,
            content=batch,
            task_type="retrieval_document",
        )
        batch_embeddings = result["embedding"]
        if batch_embeddings and isinstance(batch_embeddings[0], list):
            embeddings.extend(batch_embeddings)
        else:
            embeddings.append(batch_embeddings)
    return embeddings


async def _embed_documents_throttled(chunks: list[str]) -> list[list[float]]:
    """
    Throttle and retry embedding calls to stay under Gemini free-tier per-minute quotas.

    Note: embed_content() is synchronous; we run it in a thread and enforce spacing between requests.
    """
    # Free-tier embed quota is ~100 requests/minute => ~0.6s/request. Add margin.
    min_delay_seconds = 0.75
    last_request_at = 0.0

    all_embeddings: list[list[float]] = []
    batch_size = 20
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]

        now = time.monotonic()
        elapsed = now - last_request_at
        if elapsed < min_delay_seconds:
            await asyncio.sleep(min_delay_seconds - elapsed)

        while True:
            try:
                batch_embeddings = await asyncio.to_thread(_embed_documents_sync, batch)
                all_embeddings.extend(batch_embeddings)
                last_request_at = time.monotonic()
                break
            except Exception as exc:
                message = str(exc)
                if "ResourceExhausted" in message or "Quota exceeded" in message or "429" in message:
                    # The API usually tells you to retry in ~30-60s; use a safe default.
                    await asyncio.sleep(60)
                    continue
                raise

    return all_embeddings


if __name__ == "__main__":
    main()
