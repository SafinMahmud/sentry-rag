"""
Hybrid retriever: BM25 + dense embeddings + RRF + cross-encoder reranking.

This is the core of the RAG system. Given a question, it returns the top-k
most relevant code chunks by running four stages in sequence:

  Stage 1: BM25 keyword search      -> top 20 candidates
  Stage 2: Dense semantic search    -> top 20 candidates
  Stage 3: Reciprocal Rank Fusion   -> merged top 20
  Stage 4: Cross-encoder reranking  -> final top k (default 5)

Why four stages instead of one?
  - BM25 alone misses conceptual matches ("how does grouping work")
  - Dense alone misses exact symbol matches ("GroupingConfig class")
  - RRF combines both without needing to tune weights
  - Reranker does a deeper relevance check on the merged shortlist,
    fixing the "lost in the middle" problem where good chunks get
    buried by mediocre ones that happened to score well on both indexes

Usage (as a module):
    from retriever import Retriever
    r = Retriever()
    chunks = r.retrieve("how does Sentry group similar errors?")
    for chunk in chunks:
        print(chunk["symbol_name"], chunk["score"])

Usage (smoke test from CLI):
    python retriever.py
"""

import json
import pickle
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import CrossEncoder, SentenceTransformer

from build_bm25 import tokenize  # reuse the same tokenizer


# Paths (relative to project root, adjust if running from elsewhere)
BM25_PATH = Path("data/bm25.pkl")
CHROMA_DIR = Path("data/chroma")
CHROMA_CONFIG_PATH = CHROMA_DIR / "config.json"

# How many candidates each index returns before merging
CANDIDATE_POOL = 20

# Final number of chunks returned after reranking
TOP_K = 5

# Cross-encoder reranker model. Runs locally, no API key needed.
# bge-reranker-base is ~280MB, strong quality, runs in ~2s on CPU for 20 pairs.
RERANKER_MODEL = "BAAI/bge-reranker-base"

# RRF smoothing constant. 60 is the standard value from the original paper.
# Higher k = less penalty for being ranked lower. Don't tune this for v1.
RRF_K = 60


class Retriever:
    """
    Loads all indexes once at startup, then answers queries quickly.

    Typical usage:
        retriever = Retriever()           # load once, ~5s on first run
        results = retriever.retrieve(q)   # fast per-query, ~2-3s on CPU
    """

    def __init__(
        self,
        bm25_path: Path = BM25_PATH,
        chroma_dir: Path = CHROMA_DIR,
        candidate_pool: int = CANDIDATE_POOL,
        top_k: int = TOP_K,
        reranker_model: str = RERANKER_MODEL,
    ):
        self.candidate_pool = candidate_pool
        self.top_k = top_k

        print("Loading retriever components...")

        # Load BM25 index and chunks list
        print("  [1/4] loading BM25 index...")
        with bm25_path.open("rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]
        print(f"         {len(self.chunks)} chunks indexed")

        # Load ChromaDB
        print("  [2/4] loading ChromaDB...")
        chroma_config = json.loads(CHROMA_CONFIG_PATH.read_text())
        self.chroma_client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_collection(
            chroma_config["collection_name"]
        )
        print(f"         {self.collection.count()} vectors loaded")

        # Load embedding model (same one used to build the index)
        print("  [3/4] loading embedding model...")
        self.embedding_model = SentenceTransformer(chroma_config["embedding_model"])
        self.query_prefix = chroma_config["query_prefix"]
        print(f"         model: {chroma_config['embedding_model']}")

        # Load cross-encoder reranker
        # Downloads ~280MB on first run, cached after that
        print("  [4/4] loading reranker...")
        self.reranker = CrossEncoder(reranker_model)
        print(f"         model: {reranker_model}")

        print("Retriever ready.\n")

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """
        Run the full 4-stage retrieval pipeline for a query.

        Returns a list of chunk dicts, each with an added 'score' field
        (the reranker score, higher = more relevant). Sorted best-first.
        """
        top_k = top_k or self.top_k

        # Stage 1: BM25
        bm25_results = self._bm25_search(query)

        # Stage 2: Dense
        dense_results = self._dense_search(query)

        # Stage 3: RRF merge
        merged = self._reciprocal_rank_fusion(bm25_results, dense_results)

        # Stage 4: Rerank
        reranked = self._rerank(query, merged, top_k)

        return reranked

    def _bm25_search(self, query: str) -> list[dict]:
        """
        BM25 keyword search. Returns top CANDIDATE_POOL chunks.

        We tokenize the query the same way we tokenized documents at
        index time (same function from build_bm25.py). Then ask BM25
        for the top-scoring documents.
        """
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # argsort gives ascending order, so reverse with [::-1]
        top_indices = np.argsort(scores)[::-1][: self.candidate_pool]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:  # skip zero-score results
                chunk = dict(self.chunks[idx])
                chunk["bm25_score"] = float(scores[idx])
                chunk["bm25_rank"] = rank
                results.append(chunk)

        return results

    def _dense_search(self, query: str) -> list[dict]:
        """
        Semantic search using embeddings. Returns top CANDIDATE_POOL chunks.

        The query prefix is added here (BGE models need it for queries
        but not for documents, quirk of how they were trained).
        """
        prefixed_query = self.query_prefix + query
        query_embedding = self.embedding_model.encode(
            [prefixed_query],
            normalize_embeddings=True,
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=self.candidate_pool,
            include=["metadatas", "documents", "distances"],
        )

        chunks = []
        for rank, (meta, doc, dist) in enumerate(
            zip(
                results["metadatas"][0],
                results["documents"][0],
                results["distances"][0],
            )
        ):
            # Convert cosine distance to similarity (0=identical, 1=similar, -1=opposite)
            similarity = 1 - dist
            chunk = {
                # Reconstruct a chunk dict from ChromaDB metadata + document
                "id": results["ids"][0][rank],
                "text": doc,
                "file_path": meta["file_path"],
                "symbol_name": meta["symbol_name"],
                "symbol_type": meta["symbol_type"],
                "parent_class": meta.get("parent_class") or None,
                "start_line": meta["start_line"],
                "end_line": meta["end_line"],
                "file_summary": meta.get("file_summary") or "",
                "dense_score": similarity,
                "dense_rank": rank,
            }
            chunks.append(chunk)

        return chunks

    def _reciprocal_rank_fusion(
        self,
        bm25_results: list[dict],
        dense_results: list[dict],
    ) -> list[dict]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion (RRF).

        RRF score for a document = sum of 1/(k + rank) across all lists
        it appears in, where k is a smoothing constant (default 60).

        Why RRF instead of just combining scores?
        - BM25 scores and dense similarity scores are on different scales
          (15.4 vs 0.73 from our smoke tests). You can't add them directly.
        - RRF only uses rank position, not raw scores, so it's scale-invariant.
        - It naturally rewards chunks that appear in BOTH lists (they get
          a contribution from each), which is what we want.

        Example:
          Chunk A: BM25 rank 1, dense rank 3
            RRF = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
          Chunk B: BM25 rank 2, dense rank 1
            RRF = 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325
          Chunk C: BM25 rank 1, not in dense top-20
            RRF = 1/(60+1) = 0.0164
          Chunk D: dense rank 1, not in BM25 top-20
            RRF = 1/(60+1) = 0.0164

        B wins because it appears high in both lists.
        C and D tie because appearing #1 in one list = not appearing in other.
        """
        # Build a lookup: chunk_id -> chunk dict
        # Dense results are the base since they have full metadata
        chunk_by_id: dict[str, dict] = {}
        for chunk in dense_results:
            chunk_by_id[chunk["id"]] = chunk
        for chunk in bm25_results:
            if chunk["id"] not in chunk_by_id:
                chunk_by_id[chunk["id"]] = chunk

        # Compute RRF scores
        rrf_scores: dict[str, float] = {}

        for rank, chunk in enumerate(bm25_results):
            cid = chunk["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (RRF_K + rank)

        for rank, chunk in enumerate(dense_results):
            cid = chunk["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (RRF_K + rank)

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores, key=lambda cid: -rrf_scores[cid])

        # Return top CANDIDATE_POOL chunks with RRF score attached
        merged = []
        for cid in sorted_ids[: self.candidate_pool]:
            chunk = dict(chunk_by_id[cid])
            chunk["rrf_score"] = rrf_scores[cid]
            merged.append(chunk)

        return merged

    def _rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int,
    ) -> list[dict]:
        """
        Re-score the merged candidates using a cross-encoder reranker.

        A cross-encoder sees the query AND a candidate together in one
        forward pass, giving it much richer signal than the embedding
        model which encodes them separately. It's slower (can't pre-compute
        candidate embeddings) but far more accurate for the final ranking.

        We only run it on CANDIDATE_POOL (20) pairs, not all 1907 chunks,
        so it stays fast (~2s on CPU for 20 pairs).
        """
        if not candidates:
            return []

        # Cross-encoder expects (query, text) pairs
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)

        # Attach reranker score and sort
        for chunk, score in zip(candidates, scores):
            chunk["score"] = float(score)

        reranked = sorted(candidates, key=lambda c: -c["score"])
        return reranked[:top_k]


def smoke_test():
    """
    Run 3 test queries through the full pipeline and print results.
    Good for verifying everything works end to end.
    """
    retriever = Retriever()

    queries = [
        "how does Sentry group similar errors into issues",
        "where is rate limiting applied to event ingestion",
        "how does authentication work for API endpoints",
    ]

    for query in queries:
        print(f"{'=' * 60}")
        print(f"Query: {query}")
        print(f"{'=' * 60}")
        results = retriever.retrieve(query)
        for i, chunk in enumerate(results, 1):
            print(f"\n  {i}. [{chunk['symbol_type']}] {chunk['symbol_name']}")
            print(f"     file: {chunk['file_path']}")
            print(f"     lines: {chunk['start_line']}-{chunk['end_line']}")
            print(f"     reranker score: {chunk['score']:.4f}")
            if chunk.get("file_summary"):
                print(f"     summary: {chunk['file_summary'][:80]}")
        print()


if __name__ == "__main__":
    smoke_test()