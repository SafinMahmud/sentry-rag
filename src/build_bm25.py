"""
Build a BM25 keyword search index over the chunked codebase.

BM25 is the same algorithm that powers most search engines (including
Elasticsearch's default). It scores documents by how often the query
terms appear, weighted by how rare those terms are across the whole
corpus. This makes it excellent at matching exact symbol names like
`GroupingConfig` or `EventManager` that semantic search often misses.

We index two things per chunk:
  1. The raw source code text (catches function bodies, variable names)
  2. The symbol name with extra repetition (boosts exact name matches)

Usage:
    python build_bm25.py data/chunks_ctx.jsonl data/bm25.pkl

Output:
    data/bm25.pkl  -- a pickle file containing (bm25_index, chunks_list)
                      loaded at query time by the retriever
"""

import json
import pickle
import re
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """
    Convert text into a list of tokens for BM25 indexing.

    We do four things:
    1. Lowercase everything so "EventManager" matches "eventmanager"
    2. Split camelCase and PascalCase into separate tokens
       so "EventManager" becomes ["event", "manager"] and matches
       a query like "event manager" or "manage events"
    3. Split on non-alphanumeric characters (spaces, dots, underscores,
       brackets etc.) to handle snake_case, dotted paths, and code syntax
    4. Drop very short tokens (single chars, empty strings) that add noise

    Example:
        "def get_user_by_email(email: str)" ->
        ["def", "get", "user", "by", "email", "email", "str"]
    """
    # Step 1: split camelCase/PascalCase before lowercasing
    # Insert a space before any uppercase letter that follows a lowercase letter
    # "EventManager" -> "Event Manager", "getUserById" -> "get User By Id"
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # Step 2: lowercase
    text = text.lower()

    # Step 3: split on anything that isn't a letter or digit
    tokens = re.split(r"[^a-z0-9]+", text)

    # Step 4: drop tokens shorter than 2 chars (removes "a", "i", "", etc.)
    tokens = [t for t in tokens if len(t) >= 2]

    return tokens


def build_document(chunk: dict) -> str:
    """
    Build the text we'll index for a chunk.

    We combine:
    - The raw source code (catches all variable names, logic, comments)
    - The symbol name repeated 3 times (gives it extra weight so exact
      symbol name queries rank this chunk highly)
    - The file path (so queries like "api endpoints" find files in src/api/)
    - The docstring separately if present (plain English, good for BM25)

    Repeating the symbol name is a simple but effective trick. BM25 scores
    by term frequency, so if someone searches for "EventManager", a chunk
    whose document contains "EventManager EventManager EventManager" will
    score higher than one that mentions it only once in passing.
    """
    parts = []

    # Symbol name repeated for extra weight
    symbol = chunk.get("symbol_name", "")
    if symbol:
        parts.append(" ".join([symbol] * 3))

    # File path (helps match queries about specific subsystems)
    parts.append(chunk.get("file_path", ""))

    # Docstring (plain English description, very BM25-friendly)
    if chunk.get("docstring"):
        parts.append(chunk["docstring"])

    # Full source code
    parts.append(chunk.get("text", ""))

    return " ".join(parts)


def main():
    if len(sys.argv) != 3:
        print("Usage: python build_bm25.py <chunks_ctx.jsonl> <bm25.pkl>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    # Load chunks
    print(f"Loading chunks from {input_path}...")
    chunks = []
    with input_path.open() as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"  loaded {len(chunks)} chunks")

    # Tokenize each chunk's document
    print("Tokenizing...")
    tokenized_docs = []
    for i, chunk in enumerate(chunks):
        doc = build_document(chunk)
        tokens = tokenize(doc)
        tokenized_docs.append(tokens)
        if (i + 1) % 500 == 0:
            print(f"  tokenized {i + 1}/{len(chunks)}")

    # Build the BM25 index
    # BM25Okapi is the standard variant with two tuning params:
    #   k1 (default 1.5): controls term frequency saturation.
    #      Higher = more reward for repeated terms.
    #   b  (default 0.75): controls document length normalization.
    #      1.0 = fully normalize (penalize long docs), 0.0 = no normalization.
    # Defaults are fine for code; no need to tune for a v1.
    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_docs)

    # Save both the index and the original chunks together.
    # At query time we need both: BM25 to rank, chunks to return the text.
    print(f"Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    print(f"Done. BM25 index built over {len(chunks)} chunks.")
    print()
    print("Quick smoke test:")
    smoke_test(bm25, chunks)


def smoke_test(bm25: BM25Okapi, chunks: list[dict]):
    """
    Run two quick test queries so you can eyeball whether the index works.
    These aren't real eval questions, just a sanity check that the top
    results look plausible before moving on.
    """
    test_queries = [
        "event grouping fingerprint",
        "api endpoint authentication",
    ]

    for query in test_queries:
        tokens = tokenize(query)
        scores = bm25.get_scores(tokens)

        # Get top 3 results
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:3]

        print(f"Query: '{query}'")
        for rank, idx in enumerate(top_indices, 1):
            chunk = chunks[idx]
            print(f"  {rank}. [{chunk['symbol_type']}] {chunk['symbol_name']}")
            print(f"     {chunk['file_path']}:{chunk['start_line']}-{chunk['end_line']}")
            print(f"     score: {scores[idx]:.3f}")
        print()


if __name__ == "__main__":
    main()