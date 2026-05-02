"""Chunk helpers for extraction passes."""

from typing import Dict, Iterator


MAX_CHUNK_CHARS = 1500
SUBCHUNK_OVERLAP_CHARS = 200


def iter_extraction_chunks(chunk: Dict) -> Iterator[Dict]:
    """
    Yield chunk-sized text windows for extraction.

    Dataset chunks below MAX_CHUNK_CHARS keep their original IDs for cache stability.
    Larger chunks are split into overlapping subchunks instead of being truncated.
    """
    chunk_text = chunk.get("content", "")
    if len(chunk_text) <= MAX_CHUNK_CHARS:
        yield {
            "id": chunk.get("id", "chunk_0"),
            "content": chunk_text,
        }
        return

    base_id = chunk.get("id", "chunk_0")
    step = max(1, MAX_CHUNK_CHARS - SUBCHUNK_OVERLAP_CHARS)
    start = 0
    part = 0

    while start < len(chunk_text):
        end = min(len(chunk_text), start + MAX_CHUNK_CHARS)
        text = chunk_text[start:end]
        if text.strip():
            yield {
                "id": f"{base_id}_part_{part}",
                "content": text,
            }
        if end >= len(chunk_text):
            break
        start += step
        part += 1
