"""
AST-based chunker for Python codebases.

Splits Python files into chunks at function and class boundaries, preserving
metadata (file path, line numbers, symbol name, parent class, docstring).

Usage:
    python chunker.py <repo_path> <output.jsonl>

Example:
    python chunker.py ./sentry/src/sentry chunks.jsonl
"""

import ast
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator


# Folders we don't want to index. Tests are noisy, migrations are auto-generated,
# __pycache__ is bytecode, .venv is dependencies.
SKIP_DIRS = {"__pycache__", "tests", "test", "migrations", ".venv", "venv", ".git", "node_modules"}

# If a chunk's source code is longer than this many characters, we'll try to
# split it further (e.g., a giant class gets split into its methods).
# ~6000 chars is roughly 1500 tokens, a comfortable size for retrieval.
MAX_CHUNK_CHARS = 6000

# Chunks shorter than this are usually noise (one-line wrappers, getters).
# We keep them but flag them; you can filter later if needed.
MIN_CHUNK_CHARS = 50


@dataclass
class Chunk:
    """One unit of code we'll index and retrieve."""
    id: str                  # unique identifier, e.g. "sentry/api/views.py:UserView.get:142-178"
    text: str                # the actual source code
    file_path: str           # relative path from repo root
    symbol_name: str         # function/class name, e.g. "UserView.get"
    symbol_type: str         # "function" | "class" | "method" | "module"
    parent_class: str | None # for methods, the class they belong to
    start_line: int          # 1-indexed, inclusive
    end_line: int            # 1-indexed, inclusive
    docstring: str | None    # the symbol's docstring if it has one


def should_skip_dir(dir_name: str) -> bool:
    """Skip hidden dirs and the SKIP_DIRS list."""
    return dir_name.startswith(".") or dir_name in SKIP_DIRS


def find_python_files(repo_root: Path) -> Iterator[Path]:
    """Walk the repo and yield every .py file we want to index."""
    for path in repo_root.rglob("*.py"):
        # Check every parent dir against the skip list
        if any(should_skip_dir(part) for part in path.relative_to(repo_root).parts):
            continue
        yield path


def get_source_segment(source_lines: list[str], node: ast.AST) -> str:
    """
    Extract the source code for an AST node using its line numbers.

    We use line numbers (not ast.get_source_segment) because we want to capture
    decorators and surrounding whitespace context, and line-based slicing is
    more predictable across Python versions.
    """
    # AST line numbers are 1-indexed; list indices are 0-indexed
    start = node.lineno - 1

    # If the node has decorators, back up to include them
    if hasattr(node, "decorator_list") and node.decorator_list:
        start = node.decorator_list[0].lineno - 1

    end = node.end_lineno  # already exclusive when used as a slice end
    return "".join(source_lines[start:end])


def get_real_start_line(node: ast.AST) -> int:
    """Like node.lineno but accounts for decorators."""
    if hasattr(node, "decorator_list") and node.decorator_list:
        return node.decorator_list[0].lineno
    return node.lineno


def make_chunk_id(file_path: str, symbol_name: str, start: int, end: int) -> str:
    """A stable, human-readable ID. Useful for citations later."""
    return f"{file_path}:{symbol_name}:{start}-{end}"


def chunk_function_or_method(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    source_lines: list[str],
    file_path: str,
    parent_class: str | None = None,
) -> Chunk:
    """Turn a function/method AST node into a Chunk."""
    symbol_name = f"{parent_class}.{node.name}" if parent_class else node.name
    start_line = get_real_start_line(node)
    end_line = node.end_lineno

    return Chunk(
        id=make_chunk_id(file_path, symbol_name, start_line, end_line),
        text=get_source_segment(source_lines, node),
        file_path=file_path,
        symbol_name=symbol_name,
        symbol_type="method" if parent_class else "function",
        parent_class=parent_class,
        start_line=start_line,
        end_line=end_line,
        docstring=ast.get_docstring(node),
    )


def chunk_class(
    node: ast.ClassDef,
    source_lines: list[str],
    file_path: str,
) -> list[Chunk]:
    """
    Turn a class AST node into chunks.

    Strategy: if the class is small enough, keep it as one chunk.
    If it's big, split it into individual methods so each is its own chunk.
    This keeps related code together when possible but avoids chunks too big
    to retrieve usefully.
    """
    full_text = get_source_segment(source_lines, node)
    start_line = get_real_start_line(node)
    end_line = node.end_lineno

    # Small enough: keep the whole class as one chunk
    if len(full_text) <= MAX_CHUNK_CHARS:
        return [Chunk(
            id=make_chunk_id(file_path, node.name, start_line, end_line),
            text=full_text,
            file_path=file_path,
            symbol_name=node.name,
            symbol_type="class",
            parent_class=None,
            start_line=start_line,
            end_line=end_line,
            docstring=ast.get_docstring(node),
        )]

    # Too big: emit one "header" chunk for the class signature + docstring,
    # then one chunk per method
    chunks = []

    # Build a header chunk: just the class definition line, decorators, and docstring.
    # This gives retrieval something to match on for "what does ClassName do" questions.
    header_end = node.body[0].lineno - 1 if node.body else node.lineno
    if ast.get_docstring(node) and isinstance(node.body[0], ast.Expr):
        # Include the docstring in the header
        header_end = node.body[0].end_lineno
    header_text = "".join(source_lines[start_line - 1:header_end])

    chunks.append(Chunk(
        id=make_chunk_id(file_path, f"{node.name}__class_header", start_line, header_end),
        text=header_text,
        file_path=file_path,
        symbol_name=f"{node.name} (class header)",
        symbol_type="class",
        parent_class=None,
        start_line=start_line,
        end_line=header_end,
        docstring=ast.get_docstring(node),
    ))

    # One chunk per method
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(chunk_function_or_method(
                child, source_lines, file_path, parent_class=node.name
            ))

    return chunks


def chunk_file(file_path: Path, repo_root: Path) -> list[Chunk]:
    """Parse one Python file into chunks."""
    try:
        source = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as e:
        print(f"  skipping {file_path}: {e}", file=sys.stderr)
        return []

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        # Some files in big repos have intentional syntax errors (test fixtures)
        # or use Python versions newer than ours. Skip them.
        print(f"  syntax error in {file_path}: {e}", file=sys.stderr)
        return []

    source_lines = source.splitlines(keepends=True)
    rel_path = str(file_path.relative_to(repo_root))
    chunks: list[Chunk] = []

    # Module-level docstring becomes its own chunk if it exists.
    # This is what someone asking "what is this module for" would match against.
    module_docstring = ast.get_docstring(tree)
    if module_docstring:
        chunks.append(Chunk(
            id=make_chunk_id(rel_path, "__module__", 1, 1),
            text=f'"""{module_docstring}"""',
            file_path=rel_path,
            symbol_name="__module__",
            symbol_type="module",
            parent_class=None,
            start_line=1,
            end_line=len(module_docstring.splitlines()) + 2,
            docstring=module_docstring,
        ))

    # Walk only the TOP LEVEL of the module. We deliberately don't recurse with
    # ast.walk because we don't want nested functions inside functions as separate
    # chunks (they're usually closures that only make sense in context).
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(chunk_function_or_method(node, source_lines, rel_path))
        elif isinstance(node, ast.ClassDef):
            chunks.extend(chunk_class(node, source_lines, rel_path))
        # We skip module-level Assign, Import, etc. They're rarely useful for
        # retrieval on their own. If a question is about constants, the module
        # docstring or surrounding functions usually carry enough context.

    # Filter chunks that are too tiny to be useful
    chunks = [c for c in chunks if len(c.text) >= MIN_CHUNK_CHARS]

    return chunks


def main():
    if len(sys.argv) != 3:
        print("Usage: python chunker.py <repo_path> <output.jsonl>")
        sys.exit(1)

    repo_root = Path(sys.argv[1]).resolve()
    output_path = Path(sys.argv[2])

    if not repo_root.is_dir():
        print(f"Not a directory: {repo_root}")
        sys.exit(1)

    print(f"Scanning {repo_root}...")
    files = list(find_python_files(repo_root))
    print(f"Found {len(files)} Python files")

    total_chunks = 0
    with output_path.open("w", encoding="utf-8") as out:
        for i, file_path in enumerate(files, 1):
            chunks = chunk_file(file_path, repo_root)
            for chunk in chunks:
                out.write(json.dumps(asdict(chunk)) + "\n")
            total_chunks += len(chunks)
            if i % 100 == 0:
                print(f"  processed {i}/{len(files)} files, {total_chunks} chunks so far")

    print(f"Done. Wrote {total_chunks} chunks to {output_path}")


if __name__ == "__main__":
    main()