"""Split the official MultiHop-RAG corpus into traceable BM25 chunks."""

import hashlib
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = PROJECT_ROOT / "data" / "raw" / "corpus.json"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
REPORT_PATH = PROJECT_ROOT / "outputs" / "chunk_validation.json"

EXPECTED_DOCUMENTS = 609
CHUNK_SIZE = 300
OVERLAP = 50
WORD_PATTERN = re.compile(r"\b\w+\b")  # Same word definition as A3.1
REQUIRED_FIELDS = ("title", "url", "source", "published_at", "category")


def sha256_file(path):
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def build_chunks_for_document(doc, doc_index):
    """Return body-only chunks with their original source offsets."""
    for field in REQUIRED_FIELDS:
        if not isinstance(doc.get(field), str) or not doc[field].strip():
            raise ValueError(f"Document {doc_index}: missing/invalid {field}")
    body = doc.get("body")
    if not isinstance(body, str) or not body.strip():
        raise ValueError(f"Document {doc_index}: missing/invalid body")

    words = list(WORD_PATTERN.finditer(body))
    if not words:
        raise ValueError(f"Document {doc_index}: no words found")

    doc_id = f"d{doc_index:04d}"  # Stable as long as corpus order stays fixed
    result = []
    start_word = 0
    while start_word < len(words):
        end_word = min(start_word + CHUNK_SIZE, len(words))
        start_char = words[start_word].start()
        end_char = words[end_word].start() if end_word < len(words) else len(body)
        chunk_index = len(result)
        result.append({
            "chunk_id": f"{doc_id}_c{chunk_index:04d}",
            "doc_id": doc_id,
            "chunk_index": chunk_index,
            "start_word": start_word,          # Zero-based, inclusive
            "end_word": end_word,              # Zero-based, exclusive
            "start_char": start_char,          # Zero-based, inclusive
            "end_char": end_char,              # Zero-based, exclusive
            "word_count": end_word - start_word,
            "text": body[start_char:end_char], # Verbatim substring of source body
            "title": doc["title"],
            "url": doc["url"],
            "source": doc["source"],
            "published_at": doc["published_at"],
            "category": doc["category"],
            "author": doc.get("author"),
        })
        if end_word == len(words):
            break  # Avoid redundant, very short trailing chunks
        start_word = end_word - OVERLAP
    return result, len(words)


def validate_document(doc, chunks, total_words):
    """Check source traceability, exact text, and complete word coverage."""
    if not chunks or chunks[0]["start_word"] != 0 or chunks[-1]["end_word"] != total_words:
        raise ValueError("Incomplete document coverage")
    previous = None
    for index, chunk in enumerate(chunks):
        if chunk["chunk_index"] != index or not 0 < chunk["word_count"] <= CHUNK_SIZE:
            raise ValueError(f"Invalid chunk index/length: {chunk['chunk_id']}")
        if chunk["text"] != doc["body"][chunk["start_char"]:chunk["end_char"]]:
            raise ValueError(f"Chunk text not from source: {chunk['chunk_id']}")
        if len(WORD_PATTERN.findall(chunk["text"])) != chunk["word_count"]:
            raise ValueError(f"Word count mismatch: {chunk['chunk_id']}")
        if previous and previous["end_word"] - chunk["start_word"] != OVERLAP:
            raise ValueError(f"Incorrect chunk overlap: {chunk['chunk_id']}")
        previous = chunk


def main():
    if not 0 < OVERLAP < CHUNK_SIZE:
        raise ValueError("Require 0 < OVERLAP < CHUNK_SIZE")
    with CORPUS_PATH.open("r", encoding="utf-8") as file:
        corpus = json.load(file)
    if not isinstance(corpus, list) or len(corpus) != EXPECTED_DOCUMENTS:
        raise ValueError(f"Expected {EXPECTED_DOCUMENTS} official documents")

    print("=" * 68)
    print("A3.2 - Corpus Chunking and Metadata Processing")
    print("=" * 68)
    print(f"Corpus documents : {len(corpus)}")
    print(f"Chunk size       : {CHUNK_SIZE} words")
    print(f"Overlap          : {OVERLAP} words")

    all_chunks = []
    source_word_total = 0
    for index, doc in enumerate(corpus):
        chunks, n_words = build_chunks_for_document(doc, index)
        validate_document(doc, chunks, n_words)
        all_chunks.extend(chunks)
        source_word_total += n_words

    ids = [chunk["chunk_id"] for chunk in all_chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs")
    covered = {chunk["doc_id"] for chunk in all_chunks}
    if len(covered) != len(corpus):
        raise ValueError("Some source documents have no chunks")

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as file:  # Overwrite, never append
        for chunk in all_chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Independent parse of the written JSONL output
    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        saved_chunks = [json.loads(line) for line in file]
    if saved_chunks != all_chunks:
        raise ValueError("Saved JSONL does not match in-memory chunks")

    lengths = [chunk["word_count"] for chunk in all_chunks]
    metadata_missing = {
        field: sum(not str(chunk.get(field) or "").strip() for chunk in all_chunks)
        for field in (*REQUIRED_FIELDS, "author")
    }
    if any(metadata_missing[field] for field in REQUIRED_FIELDS):
        raise ValueError("Missing required chunk metadata")

    report = {
        "input": {"documents": len(corpus), "body_words": source_word_total,
                  "corpus_sha256": sha256_file(CORPUS_PATH)},
        "parameters": {"chunk_size_words": CHUNK_SIZE, "overlap_words": OVERLAP,
                       "step_words": CHUNK_SIZE - OVERLAP, "word_regex": WORD_PATTERN.pattern},
        "output": {"chunks": len(all_chunks), "covered_documents": len(covered),
                   "chunks_sha256": sha256_file(CHUNKS_PATH)},
        "length_statistics": {"min": min(lengths), "max": max(lengths),
                              "mean": round(sum(lengths) / len(lengths), 2),
                              "chunks_below_100_words": sum(n < 100 for n in lengths)},
        "missing_chunk_metadata": metadata_missing,
        "checks": {"unique_chunk_ids": True, "complete_word_coverage": True,
                   "exact_source_substrings": True, "correct_overlap": True,
                   "saved_jsonl_verified": True},
    }
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\n[1] Chunk statistics")
    print(f"Total chunks      : {len(all_chunks)}")
    print(f"Documents covered : {len(covered)}/{len(corpus)}")
    print(f"Chunk words       : min={min(lengths)}, max={max(lengths)}, mean={report['length_statistics']['mean']}")
    print(f"Chunks <100 words : {report['length_statistics']['chunks_below_100_words']}")
    print("\n[2] Metadata check")
    for field, missing in metadata_missing.items():
        print(f"  {field}: {missing} missing chunks")
    print("\n[3] Validation: PASS")
    print("Chunk output:", CHUNKS_PATH)
    print("Report      :", REPORT_PATH)


if __name__ == "__main__":
    main()
