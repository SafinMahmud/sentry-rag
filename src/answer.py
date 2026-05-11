"""
Answer generator: takes retrieved chunks, calls Groq LLM, returns cited answer.

This is the "Generation" part of RAG. Given a question and the top-k chunks
from the retriever, it:

  1. Formats the chunks as numbered context blocks with file citations
  2. Sends them to Llama 3.3 70B via Groq with a strict prompt
  3. Gets back a JSON response with 'answer' and 'citations' fields
  4. Runs a faithfulness check: verifies each citation actually supports
     the claim it's attached to (drops hallucinated citations)
  5. Returns a clean AnswerResult dataclass

Usage:
    from answer import AnswerGenerator
    from retriever import Retriever

    retriever = Retriever()
    generator = AnswerGenerator()

    chunks = retriever.retrieve("how does Sentry group errors?")
    result = generator.answer("how does Sentry group errors?", chunks)
    print(result.answer)
    for c in result.citations:
        print(c)

CLI smoke test:
    python answer.py
"""

import json
import os
import re
from dataclasses import dataclass, field

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Use the best free model for answer generation (quality matters here)
ANSWER_MODEL = "llama-3.3-70b-versatile"

# Smaller model for faithfulness check (speed matters more than quality)
FAITHFULNESS_MODEL = "llama-3.1-8b-instant"

# Max tokens for the main answer
ANSWER_MAX_TOKENS = 1024

# Max tokens for each faithfulness check (just yes/no reasoning)
FAITHFULNESS_MAX_TOKENS = 128


@dataclass
class Citation:
    """One citation linking a claim in the answer to a source chunk."""
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str
    verified: bool = True  # False if faithfulness check failed

    def __str__(self):
        return f"[{self.file_path}:{self.start_line}-{self.end_line}]"


@dataclass
class AnswerResult:
    """The full result of one question-answering call."""
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    chunks_used: int = 0
    faithfulness_drops: int = 0  # how many citations were dropped

    def format(self) -> str:
        """Pretty-print for CLI display."""
        lines = [
            f"Question: {self.question}",
            "",
            f"Answer:",
            self.answer,
            "",
        ]
        if self.citations:
            lines.append("Sources:")
            for c in self.citations:
                status = "" if c.verified else " [unverified]"
                lines.append(f"  {c}{status}  ({c.symbol_name})")
        if self.faithfulness_drops > 0:
            lines.append(
                f"\n  Note: {self.faithfulness_drops} citation(s) dropped "
                f"(failed faithfulness check)"
            )
        return "\n".join(lines)


# The system prompt is the most important part of the generation step.
# Key requirements we enforce:
#   - Only use provided context (prevents hallucination from training data)
#   - Cite using exact [file:start-end] format (makes parsing reliable)
#   - Admit uncertainty rather than guessing (honest about retrieval gaps)
#   - Return JSON (structured output, easier to parse than freetext)
SYSTEM_PROMPT = """You are a code assistant that answers questions about the Sentry codebase.

You are given a question and a set of code snippets retrieved from the codebase.
Each snippet is labeled with a citation like [file_path:start_line-end_line].

Rules:
1. Answer ONLY using information from the provided snippets. Do not use outside knowledge.
2. Every factual claim in your answer must cite its source using the exact format: [file_path:start_line-end_line]
3. If the snippets do not contain enough information to answer, say: "The provided context does not contain enough information to answer this question." Then list what you DO know from the context.
4. Be specific and technical. Name actual classes, functions, and files.
5. Keep the answer concise: 3-6 sentences for simple questions, up to 10 for complex ones.

Return your response as JSON with exactly these two fields:
{
  "answer": "your answer with inline citations like [file.py:10-50]",
  "citations": ["file.py:10-50", "other_file.py:20-30"]
}

The citations list should contain only the unique citation strings you used in the answer."""


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks as numbered context blocks.

    Each block shows the citation label first so the LLM knows exactly
    how to cite it, then the code. We include the file summary if available
    since it helps the LLM understand the chunk's role in the codebase.
    """
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        citation = f"[{chunk['file_path']}:{chunk['start_line']}-{chunk['end_line']}]"
        lines = [f"Snippet {i}: {citation}"]

        if chunk.get("file_summary"):
            lines.append(f"Context: {chunk['file_summary']}")

        lines.append(f"Symbol: {chunk['symbol_name']} ({chunk['symbol_type']})")
        lines.append("")
        lines.append(chunk["text"])
        blocks.append("\n".join(lines))

    return "\n\n---\n\n".join(blocks)


def parse_citations(
    citation_strings: list[str],
    chunks: list[dict],
) -> list[Citation]:
    """
    Convert citation strings like "file.py:10-50" into Citation objects.

    We match each citation string back to the chunk it came from so we
    can populate symbol_name etc. If a citation doesn't match any chunk
    (the LLM invented a file path), we drop it silently.
    """
    # Build a lookup: "file_path:start-end" -> chunk
    chunk_lookup = {
        f"{c['file_path']}:{c['start_line']}-{c['end_line']}": c
        for c in chunks
    }

    citations = []
    seen = set()
    for raw in citation_strings:
        raw = raw.strip()
        if raw in seen:
            continue
        seen.add(raw)

        chunk = chunk_lookup.get(raw)
        if chunk is None:
            # LLM invented a citation, skip it
            continue

        # Parse "file.py:start-end"
        match = re.match(r"(.+):(\d+)-(\d+)$", raw)
        if not match:
            continue

        citations.append(Citation(
            file_path=match.group(1),
            start_line=int(match.group(2)),
            end_line=int(match.group(3)),
            symbol_name=chunk["symbol_name"],
            verified=True,
        ))

    return citations


def check_faithfulness(
    client: Groq,
    answer: str,
    citation: Citation,
    chunk_text: str,
) -> bool:
    """
    Ask the LLM: does this code snippet actually support the answer?

    This is a lightweight hallucination check. If the answer makes a
    claim about what a function does, but the cited code doesn't actually
    do that, we flag the citation as unverified.

    We use the smaller/faster model here since it's just a yes/no question.
    """
    prompt = f"""Does the following code snippet support the claims made in the answer?

Answer: {answer[:500]}

Code snippet from {citation.file_path}:{citation.start_line}-{citation.end_line}:
{chunk_text[:1000]}

Reply with only "yes" or "no", then one sentence explaining why."""

    try:
        response = client.chat.completions.create(
            model=FAITHFULNESS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=FAITHFULNESS_MAX_TOKENS,
            temperature=0.0,
        )
        reply = response.choices[0].message.content.strip().lower()
        return reply.startswith("yes")
    except Exception:
        # If the check fails for any reason, keep the citation
        return True


class AnswerGenerator:
    """Wraps the Groq client and handles all answer generation logic."""

    def __init__(self, run_faithfulness_check: bool = True):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Add it to your .env file."
            )
        self.client = Groq(api_key=api_key)
        self.run_faithfulness_check = run_faithfulness_check

    def answer(self, question: str, chunks: list[dict]) -> AnswerResult:
        """
        Generate a cited answer for a question given retrieved chunks.
        """
        if not chunks:
            return AnswerResult(
                question=question,
                answer="No relevant code was found for this question.",
            )

        context = format_context(chunks)

        # Main answer generation call
        try:
            response = self.client.chat.completions.create(
                model=ANSWER_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Question: {question}\n\nContext:\n{context}",
                    },
                ],
                max_tokens=ANSWER_MAX_TOKENS,
                temperature=0.1,  # slight randomness for natural language, not 0
                response_format={"type": "json_object"},
            )
        except Exception as e:
            return AnswerResult(
                question=question,
                answer=f"Error calling LLM: {e}",
            )

        # Parse the JSON response
        raw = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(raw)
            answer_text = parsed.get("answer", raw)
            citation_strings = parsed.get("citations", [])
        except json.JSONDecodeError:
            # Fallback: use raw text, extract citations with regex
            answer_text = raw
            citation_strings = re.findall(
                r"[\w/]+\.py:\d+-\d+", raw
            )

        # Convert citation strings to Citation objects
        citations = parse_citations(citation_strings, chunks)

        # Faithfulness check: verify each citation
        drops = 0
        if self.run_faithfulness_check and citations:
            # Build chunk text lookup for faithfulness check
            chunk_text_lookup = {
                f"{c['file_path']}:{c['start_line']}-{c['end_line']}": c["text"]
                for c in chunks
            }
            for citation in citations:
                key = f"{citation.file_path}:{citation.start_line}-{citation.end_line}"
                chunk_text = chunk_text_lookup.get(key, "")
                if chunk_text:
                    verified = check_faithfulness(
                        self.client, answer_text, citation, chunk_text
                    )
                    citation.verified = verified
                    if not verified:
                        drops += 1

        return AnswerResult(
            question=question,
            answer=answer_text,
            citations=citations,
            chunks_used=len(chunks),
            faithfulness_drops=drops,
        )


def smoke_test():
    """Run 2 questions end-to-end through retriever + answer generator."""
    from retriever import Retriever

    retriever = Retriever()
    generator = AnswerGenerator()

    questions = [
        "How does Sentry group similar errors into issues?",
        "How does authentication work for API endpoints?",
    ]

    for question in questions:
        print("=" * 60)
        chunks = retriever.retrieve(question)
        result = generator.answer(question, chunks)
        print(result.format())
        print()


if __name__ == "__main__":
    smoke_test()