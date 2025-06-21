import os
import re
from pathlib import Path
from typing import List
from tree_sitter_languages import get_parser

LANGUAGE_MAP = {
    '.py': 'python',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.c': 'c',
    '.cpp': 'cpp',
    '.java': 'java'
}


def _split_tokens(text: str, chunk_size: int) -> List[str]:
    tokens = re.findall(r'\S+', text)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunks.append(' '.join(tokens[i:i + chunk_size]))
    return chunks


def chunk_file(path: str, chunk_size: int = 400) -> List[str]:
    """Разбиение файла на семантические блоки с помощью tree-sitter."""
    ext = Path(path).suffix
    lang = LANGUAGE_MAP.get(ext)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            code = f.read()
    except Exception:
        return []

    if not lang:
        return _split_tokens(code, chunk_size)

    parser = get_parser(lang)
    tree = parser.parse(bytes(code, 'utf-8'))
    root = tree.root_node
    chunks = []

    for child in root.children:
        if child.type in {'function_definition', 'class_definition', 'method_definition'}:
            snippet = code[child.start_byte:child.end_byte]
            chunks.append(snippet)

    if not chunks:
        return _split_tokens(code, chunk_size)

    return chunks
