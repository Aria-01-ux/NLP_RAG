"""Build a reproducible, in-memory BM25 baseline from A3.2 chunks.
Only corpus-derived chunks are indexed. QA answers and evidence labels are never read.
"""

import argparse
import hashlib
import json
import re
from importlib.metadata import version
from pathlib import Path

from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
CHUNK_REPORT_PATH = PROJECT_ROOT / "outputs" / "chunk_validation.json"
INDEX_REPORT_PATH = PROJECT_ROOT / "outputs" / "bm25_index_report.json"
DEMO_TRACE_PATH = PROJECT_ROOT / "outputs" / "bm25_demo_trace.json"

EXPECTED_DOCUMENTS = 609
WORD_PATTERN = re.compile(r"\b\w+\b")
K1 = 1.5
B = 0.75
EPSILON = 0.25
REQUIRED_FIELDS = ("chunk_id", "doc_id", "text", "title", "url", "source", "published_at")


def sha256_file(path):
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def tokenize(text):
    """Same word boundaries as A3.2, with case folding for retrieval."""
    return WORD_PATTERN.findall(text.lower())


def load_chunks(path=CHUNKS_PATH, validation_path=CHUNK_REPORT_PATH):
    """Load and validate the exact chunk collection used for the BM25 index."""
    if not path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("Run src/chunk_corpus.py first to create chunks and validation report")

    with validation_path.open("r", encoding="utf-8") as file:
        report = json.load(file)
    digest = sha256_file(path)
    if digest != report["output"]["chunks_sha256"]:
        raise ValueError("chunks.jsonl SHA256 does not match A3.2 validation report")

    chunks = []
    chunk_ids = set()
    doc_ids = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(f"Empty JSONL line at {line_number}")
            chunk = json.loads(line)
            if not isinstance(chunk, dict):
                raise ValueError(f"Chunk at line {line_number} is not a JSON object")
            for field in REQUIRED_FIELDS:
                if not isinstance(chunk.get(field), str) or not chunk[field].strip():
                    raise ValueError(f"Line {line_number}: missing/invalid {field}")
            if chunk["chunk_id"] in chunk_ids:
                raise ValueError(f"Duplicate chunk_id: {chunk['chunk_id']}")
            chunk_ids.add(chunk["chunk_id"])
            doc_ids.add(chunk["doc_id"])
            chunks.append(chunk)

    if len(chunks) != report["output"]["chunks"]:
        raise ValueError("Chunk count differs from A3.2 report")
    if len(doc_ids) != EXPECTED_DOCUMENTS:
        raise ValueError(f"Expected {EXPECTED_DOCUMENTS} source documents; got {len(doc_ids)}")
    return chunks, digest, report


class BM25Retriever:
    """One-time in-memory index construction; reuse the instance for many queries."""

    def __init__(self, chunks):
        if not chunks:
            raise ValueError("Cannot build a BM25 index from an empty chunk list")
        self.chunks = chunks
        self.tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
        if any(not terms for terms in self.tokenized_corpus):
            raise ValueError("At least one chunk has no retrievable tokens")
        self.index = BM25Okapi(self.tokenized_corpus, k1=K1, b=B, epsilon=EPSILON)

    def retrieve(self, question, top_k=5):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        query_tokens = tokenize(question)
        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)
        # Deterministic tie breaking by original chunk order.
        ranked = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
        results = []
        for i in ranked:
            score = float(scores[i])
            if score <= 0:
                break  # Do not return arbitrary zero-score chunks as evidence.
            results.append({"rank": len(results) + 1, "score": score, **self.chunks[i]})
            if len(results) == top_k:
                break
        return results


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="A4.1 BM25 index and smoke test")
    parser.add_argument("--query", default="Amazon Cyber Monday deals", help="English test question")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    args = parser.parse_args()

    print("=" * 68)
    print("A4.1 - BM25 Chunk Index and Smoke Test")
    print("=" * 68)
    print("\n[1] Loading validated A3.2 chunks")
    chunks, digest, chunk_report = load_chunks()
    print(f"Chunks          : {len(chunks)}")
    print(f"Source documents: {len({chunk['doc_id'] for chunk in chunks})}")
    print("Chunk SHA256    :", digest)

    print("\n[2] Building in-memory BM25Okapi index")
    retriever = BM25Retriever(chunks)
    total_tokens = sum(map(len, retriever.tokenized_corpus))
    metadata = {
        "source": {"chunks_path": str(CHUNKS_PATH.relative_to(PROJECT_ROOT)),
                   "chunks_sha256": digest,
                   "chunk_count": len(chunks),
                   "source_document_count": len({chunk["doc_id"] for chunk in chunks}),
                   "chunking_parameters": chunk_report["parameters"]},
        "index": {"algorithm": "BM25Okapi", "package": "rank-bm25",
                  "package_version": version("rank-bm25"),
                  "k1": K1, "b": B, "epsilon": EPSILON,
                  "indexed_fields": ["text"],
                  "tokenizer": {"regex": WORD_PATTERN.pattern, "lowercase": True,
                                "stopwords_removed": False, "stemming": False},
                  "total_index_tokens": total_tokens,
                  "average_tokens_per_chunk": round(total_tokens / len(chunks), 2),
                  "vocabulary_size": len(retriever.index.idf),
                  "storage": "in_memory_rebuilt_from_chunks_jsonl"},
    }
    save_json(INDEX_REPORT_PATH, metadata)
    print("Vocabulary terms:", metadata["index"]["vocabulary_size"])
    print("Average tokens  :", metadata["index"]["average_tokens_per_chunk"])

    print("\n[3] Retrieval smoke test (NOT benchmark accuracy evaluation)")
    results = retriever.retrieve(args.query, args.top_k)
    trace = {"query": args.query, "query_tokens": tokenize(args.query),
             "top_k_requested": args.top_k, "results": results,
             "chunks_sha256": digest,
             "note": "One retrieval-only demonstration; no gold answers or LLM generation used"}
    save_json(DEMO_TRACE_PATH, trace)
    print("Query           :", args.query)
    print("Results returned:", len(results))
    for result in results:
        preview = " ".join(result["text"].split())[:150]
        print(f"  #{result['rank']} score={result['score']:.4f}  "
              f"{result['chunk_id']}  {result['title'][:60]}")
        print("     ", preview, "...")

    print("\n[4] Outputs")
    print("Index metadata  :", INDEX_REPORT_PATH)
    print("Demo trace      :", DEMO_TRACE_PATH)
    print("A4.1 completed: index built and retrieval executed")


if __name__ == "__main__":
    main()
