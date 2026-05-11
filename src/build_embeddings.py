"""
Build a dense vector embedding index over the chunked codebase.

This is the "semantic search" half of our hybrid retrieval system.
While BM25 matches exact keywords, embeddings capture meaning. A query
like "how does Sentry prevent duplicate errors" will match chunks about
grouping and fingerprinting even if those exact words don't appear in
the code.

How it works:
1. Load each chunk's `contextualized_text` (the one with file summary prepended)
2. Run it through a sentence-transformer model to get a 384-dim vector
3. Store all vectors in ChromaDB (a local vector database)
4. At query time: embed the query, find the closest chunk vectors

Model choice: BAAI/bge-small-en-v1.5
- 33MB, runs on CPU comfortably
- Specifically trained for retrieval tasks (not just similarity)
- Consistently top-ranked on the MTEB retrieval benchmark for its size
- Free, no API key needed

Usage:
    python build_embeddings.py data/chunks_ctx.jsonl data/chroma

Time estimate:
    ~15-30 minutes on CPU (MacBook Air M-series: ~8 min)
"""

import json
import sys
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# This model is the sweet spot of quality vs size for retrieval tasks.
# It produces 384-dimensional vectors, small enough to be fast on CPU.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# How many chunks to embed in one batch. Larger = faster but more RAM.
# 64 is safe for 8GB RAM. Drop to 32 if you see memory errors.
BATCH_SIZE = 64

# ChromaDB collection name. We'll use the same name at query time.
COLLECTION_NAME = "sentry_code"

# BGE models work best with this prefix on queries (not on documents).
# We store this so the retriever knows to add it at query time.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open() as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def batched(items: list, size: int):
    """Split a list into batches of a given size."""
    for i in range(0, len(items), size):
        yield items[i:i + size]


def main():
    if len(sys.argv) != 3:
        print("Usage: python build_embeddings.py <chunks_ctx.jsonl> <chroma_dir>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    chroma_dir = Path(sys.argv[2])

    # Load chunks
    print(f"Loading chunks from {input_path}...")
    chunks = load_chunks(input_path)
    print(f"  loaded {len(chunks)} chunks")

    # Load the embedding model
    # First run downloads it (~33MB). Subsequent runs load from cache.
    print(f"\nLoading embedding model '{EMBEDDING_MODEL}'...")
    print("  (first run downloads ~33MB, subsequent runs are instant)")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  model loaded. embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Set up ChromaDB
    # PersistentClient saves to disk so you don't re-embed on every run.
    print(f"\nSetting up ChromaDB at '{chroma_dir}'...")
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )

    # Delete existing collection if rebuilding
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  deleted existing collection (rebuilding from scratch)")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        # cosine distance is standard for sentence transformers
        metadata={"hnsw:space": "cosine"},
    )

    # Embed and index in batches
    print(f"\nEmbedding {len(chunks)} chunks in batches of {BATCH_SIZE}...")
    print("This takes 15-30 min on CPU, ~8 min on Apple Silicon.\n")

    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    start_time = time.time()

    for batch_num, batch in enumerate(batched(chunks, BATCH_SIZE), 1):
        # Use contextualized_text for embedding (has the file summary prepended).
        # This is the key difference from indexing raw code: semantic search
        # benefits from the extra context the LLM added.
        texts = [c.get("contextualized_text") or c["text"] for c in batch]

        # BGE models don't need a prefix for documents, only for queries.
        # encode() returns a numpy array of shape (batch_size, 384)
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,  # required for cosine similarity
            show_progress_bar=False,
        ).tolist()

        # ChromaDB needs: ids, embeddings, documents, metadatas
        # We store the original text (not contextualized) as the document
        # because that's what we'll show to the LLM as retrieved context.
        ids = [c["id"] for c in batch]
        documents = [c["text"] for c in batch]
        metadatas = [
            {
                "file_path": c["file_path"],
                "symbol_name": c["symbol_name"],
                "symbol_type": c["symbol_type"],
                "parent_class": c.get("parent_class") or "",
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "file_summary": c.get("file_summary") or "",
            }
            for c in batch
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        # Progress and time estimate
        elapsed = time.time() - start_time
        rate = batch_num / elapsed  # batches per second
        remaining = (total_batches - batch_num) / rate if rate > 0 else 0

        print(
            f"  batch {batch_num:3d}/{total_batches} "
            f"| {batch_num * BATCH_SIZE:5d}/{len(chunks)} chunks "
            f"| {elapsed:5.0f}s elapsed "
            f"| ~{remaining:.0f}s remaining"
        )

    total_time = time.time() - start_time
    print(f"\nIndexing complete in {total_time:.0f}s")
    print(f"ChromaDB saved to '{chroma_dir}'")
    print(f"Collection '{COLLECTION_NAME}' contains {collection.count()} vectors")

    # Save model config so the retriever knows which model and prefix to use
    config = {
        "embedding_model": EMBEDDING_MODEL,
        "query_prefix": QUERY_PREFIX,
        "collection_name": COLLECTION_NAME,
        "chroma_dir": str(chroma_dir),
        "num_chunks": len(chunks),
    }
    config_path = chroma_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2))
    print(f"Config saved to '{config_path}'")

    print("\nQuick smoke test...")
    smoke_test(model, collection)


def smoke_test(model: SentenceTransformer, collection):
    """
    Run two semantic queries and print top results.
    These should return conceptually relevant chunks even without
    exact keyword matches, that's the whole point of dense retrieval.
    """
    test_queries = [
        "how are similar errors grouped together",
        "user login and session handling",
    ]

    for query in test_queries:
        # Add BGE query prefix for retrieval
        prefixed = QUERY_PREFIX + query
        query_embedding = model.encode(
            [prefixed],
            normalize_embeddings=True,
        ).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=3,
            include=["metadatas", "distances"],
        )

        print(f"\nQuery: '{query}'")
        for i, (meta, dist) in enumerate(
            zip(results["metadatas"][0], results["distances"][0]), 1
        ):
            # Cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: 1 - distance
            similarity = 1 - dist
            print(f"  {i}. [{meta['symbol_type']}] {meta['symbol_name']}")
            print(f"     {meta['file_path']}:{meta['start_line']}-{meta['end_line']}")
            print(f"     similarity: {similarity:.3f}")


if __name__ == "__main__":
    main()