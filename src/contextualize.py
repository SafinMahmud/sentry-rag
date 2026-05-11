"""
Contextualize chunks by prepending a per-file summary.

This implements a pragmatic version of Anthropic's Contextual Retrieval technique
(https://www.anthropic.com/news/contextual-retrieval).

The original technique generates one LLM-written context blurb per chunk. For a
20k-chunk codebase that's 20k LLM calls. We do something cheaper that captures
~90% of the benefit: ONE summary per FILE, then prepend that summary to every
chunk in that file. For Sentry that's ~3000 calls instead of 20k.

Why this helps retrieval:
- A chunk like `def get(self, request): return self.queryset.filter(...)` is
  hard to retrieve. Out of context it could be anything.
- Prepended with "This file implements the user authentication API endpoints,
  handling login, logout, and session management.", it becomes findable for
  questions like "where does Sentry handle login?"

Usage:
    python contextualize.py data/chunks.jsonl data/chunks_ctx.jsonl

Environment:
    GROQ_API_KEY must be set (free tier at console.groq.com)
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from groq import Groq, RateLimitError
from dotenv import load_dotenv
load_dotenv()

# Free tier on Groq is generous but not infinite. Llama 3.3 70B is the strongest
# free model, perfect for this kind of structured summarization task.
# MODEL = "llama-3.3-70b-versatile"
MODEL = "llama-3.1-8b-instant"
 

# How much of each file's content we send to the LLM. We don't need the whole
# file, just enough to write a good summary. The first ~3000 chars (imports +
# first few definitions + their docstrings) is plenty.
MAX_CHARS_PER_FILE = 1500

# Hard cap on summary length. Keep it short so it doesn't dominate the chunk text.
MAX_SUMMARY_TOKENS = 60

# Sleep between calls to stay under Groq's rate limit (30 req/min on free tier).
# 2.5s = 24 req/min, safely under the limit.
SLEEP_BETWEEN_CALLS = 4


SUMMARY_PROMPT = """You are summarizing one file from a large Python codebase to help a search system find it later.

File path: {file_path}

File content (truncated):
{content}

Write ONE sentence (max 30 words) describing what this file does and its role in the codebase. Focus on:
- The main responsibility (what problem it solves)
- Key classes or functions if they're the focus
- The subsystem it belongs to (auth, ingestion, billing, etc.)

Do NOT start with "This file" or "This module". Just state the purpose directly.

Example good summaries:
- "Implements event deduplication via fingerprint hashing, grouping similar errors into a single Issue."
- "Rate limiter for the event ingestion pipeline using a sliding window algorithm backed by Redis."
- "Django views for the user authentication API: login, logout, password reset, and 2FA verification."

Your summary:"""


def load_chunks(path: Path) -> list[dict]:
    """Load chunks from the JSONL file produced by chunker.py."""
    chunks = []
    with path.open() as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def group_by_file(chunks: list[dict]) -> dict[str, list[dict]]:
    """Group chunks by their file_path so we only summarize each file once."""
    by_file = defaultdict(list)
    for chunk in chunks:
        by_file[chunk["file_path"]].append(chunk)
    return by_file


def build_file_preview(chunks: list[dict]) -> str:
    """
    Build a representative preview of a file from its chunks.

    We concatenate chunks in order until we hit MAX_CHARS_PER_FILE. This gives
    the LLM the file's top-level structure (module docstring, imports if any
    were captured, first few functions/classes with their docstrings).
    """
    # Sort chunks by their start_line so the preview reads top-to-bottom
    sorted_chunks = sorted(chunks, key=lambda c: c["start_line"])

    parts = []
    total = 0
    for chunk in sorted_chunks:
        text = chunk["text"]
        if total + len(text) > MAX_CHARS_PER_FILE:
            # Add a partial chunk to fill remaining budget
            remaining = MAX_CHARS_PER_FILE - total
            if remaining > 200:  # only worth including if we have real space
                parts.append(text[:remaining] + "\n... [truncated]")
            break
        parts.append(text)
        total += len(text)

    return "\n\n".join(parts)


def summarize_file(client: Groq, file_path: str, preview: str) -> str:
    """Call the LLM to generate a one-line summary of a file."""
    prompt = SUMMARY_PROMPT.format(file_path=file_path, content=preview)

    # Retry loop for rate limits. Groq returns RateLimitError with a retry-after
    # hint when you exceed the per-minute quota; we just back off and try again.
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_SUMMARY_TOKENS,
                temperature=0.0,  # deterministic, we want consistent summaries
            )
            summary = response.choices[0].message.content.strip()
            # Clean up: strip quotes if the model wrapped its answer
            summary = summary.strip('"').strip("'").strip()
            # Keep it to one line
            summary = summary.split("\n")[0]
            return summary
        except RateLimitError as e:
            wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s, 80s
            print(f"  rate limited, waiting {wait}s...", file=sys.stderr)
            time.sleep(wait)
        except Exception as e:
            print(f"  error summarizing {file_path}: {e}", file=sys.stderr)
            return ""  # fall back to no context; chunk still indexed as-is

    return ""  # gave up after 5 retries


def contextualize_chunks(chunks: list[dict], summary: str) -> list[dict]:
    """
    Add a `contextualized_text` field to each chunk.

    We KEEP the original `text` field unchanged (BM25 will index that for exact
    symbol matches), and add `contextualized_text` for embedding (so semantic
    search benefits from the summary). This is the key design choice:
    different indexes get different views of the same chunk.
    """
    out = []
    for chunk in chunks:
        new_chunk = dict(chunk)  # shallow copy
        if summary:
            # Format: "<file context>\n\n<symbol context>\n\n<actual code>"
            symbol_line = f"Symbol: {chunk['symbol_name']} ({chunk['symbol_type']})"
            if chunk.get("parent_class"):
                symbol_line += f" in class {chunk['parent_class']}"
            new_chunk["contextualized_text"] = (
                f"File context: {summary}\n"
                f"{symbol_line}\n\n"
                f"{chunk['text']}"
            )
            new_chunk["file_summary"] = summary
        else:
            # No summary available, just use the original text
            new_chunk["contextualized_text"] = chunk["text"]
            new_chunk["file_summary"] = ""
        out.append(new_chunk)
    return out


def main():
    if len(sys.argv) != 3:
        print("Usage: python contextualize.py <chunks.jsonl> <chunks_ctx.jsonl>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY environment variable not set.")
        print("Get a free key at https://console.groq.com and put it in your .env")
        sys.exit(1)

    client = Groq(api_key=api_key)

    print(f"Loading chunks from {input_path}...")
    chunks = load_chunks(input_path)
    print(f"  loaded {len(chunks)} chunks")

    by_file = group_by_file(chunks)
    print(f"  grouped into {len(by_file)} files")
    print(f"  estimated time: {len(by_file) * SLEEP_BETWEEN_CALLS / 60:.0f} minutes")
    print()

    # Cache summaries to disk as we go, so a crash doesn't lose hours of work
    cache_path = output_path.with_suffix(".cache.json")
    summaries: dict[str, str] = {}
    if cache_path.exists():
        summaries = json.loads(cache_path.read_text())
        print(f"  resumed: loaded {len(summaries)} cached summaries from {cache_path}")

    files_to_process = [f for f in by_file if f not in summaries]
    print(f"  {len(files_to_process)} files still need summarizing")
    print()

    # Generate summaries one file at a time
    for i, file_path in enumerate(files_to_process, 1):
        file_chunks = by_file[file_path]
        preview = build_file_preview(file_chunks)

        summary = summarize_file(client, file_path, preview)
        summaries[file_path] = summary

        # Persist cache every 10 files
        if i % 10 == 0:
            cache_path.write_text(json.dumps(summaries))
            print(f"  [{i}/{len(files_to_process)}] {file_path}")
            print(f"      -> {summary[:100]}")

        time.sleep(SLEEP_BETWEEN_CALLS)

    # Final cache write
    cache_path.write_text(json.dumps(summaries))
    print(f"\nAll summaries generated. Writing contextualized chunks...")

    # Now write the contextualized chunks
    with output_path.open("w") as out:
        for file_path, file_chunks in by_file.items():
            summary = summaries.get(file_path, "")
            ctx_chunks = contextualize_chunks(file_chunks, summary)
            for chunk in ctx_chunks:
                out.write(json.dumps(chunk) + "\n")

    print(f"Done. Wrote {len(chunks)} contextualized chunks to {output_path}")
    print(f"Summary cache kept at {cache_path} (delete to regenerate)")


if __name__ == "__main__":
    main()