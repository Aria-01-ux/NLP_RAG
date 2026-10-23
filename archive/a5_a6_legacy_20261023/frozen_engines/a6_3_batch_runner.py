"""A6.3: Auditable and resumable BM25 -> Ollama batch prediction runner.

Usage (from project root):
  python src/a6_3_batch_runner.py --mode preflight
  python src/a6_3_batch_runner.py --mode pilot --count 20
  python src/a6_3_batch_runner.py --mode full --confirm-full-run

Only corpus chunks and question/query text are passed to retrieval/generation.
Reference answers or reference evidence are NEVER read by this script.
Do not interpret a successful run as an official benchmark score.
"""

import argparse
import hashlib
import json
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import rag_pipeline
from bm25_retriever import BM25Retriever, PROJECT_ROOT, load_chunks, K1, B, EPSILON
from rag_pipeline import (QUESTIONS_PATH, SYSTEM_PROMPT, DEFAULT_MODEL, DEFAULT_BASE_URL,
                          load_queries, run_one, sha256_file)

ROOT = PROJECT_ROOT
OUT = ROOT / "outputs"
TYPES = ("inference_query", "comparison_query", "temporal_query", "null_query")
EXPECTED = 2556


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def choose_pilot(questions, count, seed):
    """Stratified deterministic pilot; whole set only via explicitly confirmed full mode."""
    if not 4 <= count <= EXPECTED or count % 4:
        raise ValueError("--count must be divisible by 4 and >= 4")
    groups = defaultdict(list)
    for qid, q in questions.items():
        groups[q["question_type"]].append(qid)
    rng = random.Random(seed)
    result = []
    for t in TYPES:
        if len(groups[t]) < count // 4:
            raise ValueError(f"Not enough questions of type {t}")
        result.extend(rng.sample(groups[t], count // 4))
    return sorted(result, key=lambda s: int(s[1:]))


def load_checkpoint(path, ids):
    """Reject duplicate IDs and corrupted/partial records instead of silently skipping them."""
    data = {}
    if not path.exists():
        return data
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                raise ValueError(f"Blank line {line_number} in {path}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Corrupted JSONL line {line_number} in {path}: {exc}. Back up before repair.") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Invalid record at line {line_number}")
            qid = record.get("question_id")
            if qid not in ids or qid in data:
                raise ValueError(f"Unexpected or duplicate question ID {qid!r} in {path}")
            if record.get("status") not in ("success", "incomplete", "error"):
                raise ValueError(f"Unknown status for {qid}")
            data[qid] = record
    return data


def check_predictions(predictions, ids):
    expected = list(ids)
    received = [row.get("question_id") for row in predictions]
    if received != expected or len(set(received)) != len(expected):
        raise ValueError("Prediction order/IDs do not match the expected question list")
    for row in predictions:
        if set(row) != {"question_id", "model_answer"} or not isinstance(row["model_answer"], str):
            raise ValueError("Invalid prediction JSON record")


def previous_run_stats():
    """Estimate model time from actual existing local A5.3 traces, if available."""
    path = OUT / "a5_3_sample_traces.jsonl"
    if not path.exists():
        return None
    seconds, tokens_in, tokens_out = [], [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            gen = row.get("generation") or {}
            ns = gen.get("total_duration_ns")
            if isinstance(ns, (int, float)) and ns > 0:
                seconds.append(ns / 1e9)
            if isinstance(gen.get("prompt_eval_count"), int):
                tokens_in.append(gen["prompt_eval_count"])
            if isinstance(gen.get("eval_count"), int):
                tokens_out.append(gen["eval_count"])
    if not seconds:
        return None
    return {
        "source": str(path), "measured_questions": len(seconds),
        "mean_model_time_seconds": round(statistics.mean(seconds), 2),
        "median_model_time_seconds": round(statistics.median(seconds), 2),
        "rough_2556_questions_model_hours_at_same_speed": round(statistics.mean(seconds) * EXPECTED / 3600, 2),
        "mean_prompt_tokens": round(statistics.mean(tokens_in), 2) if tokens_in else None,
        "mean_output_tokens": round(statistics.mean(tokens_out), 2) if tokens_out else None,
        "warning": "Rough sequential projection, excludes retrieval/overhead; hardware, prompt length and failures may vary."
    }


def build_settings(args, qids, chunk_sha):
    return {
        "mode": args.mode, "question_ids": qids, "seed": args.seed if args.mode == "pilot" else None,
        "chunks_sha256": chunk_sha, "questions_sha256": sha256_file(QUESTIONS_PATH),
        "pipeline_source_sha256": sha256_file(Path(rag_pipeline.__file__)),
        "retriever_source_sha256": sha256_file(Path(__file__).parent / "bm25_retriever.py"),
        "runner_source_sha256": sha256_file(Path(__file__)),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "retriever": {"type": "BM25Okapi", "k1": K1, "b": B, "epsilon": EPSILON,
                      "top_k": args.top_k, "unique_docs": args.unique_docs, "index_field": "text"},
        "generator": {"model": args.model, "base_url": args.base_url,
                      "num_ctx": args.num_ctx, "num_predict": args.num_predict,
                      "temperature": args.temperature, "max_context_chars": args.max_context_chars,
                      "timeout_seconds": args.timeout, "retries": args.retries},
    }


def print_preflight(questions, chunks, chunk_sha):
    print("=" * 74)
    print("A6.3 - PRE-FLIGHT (NO LLM CALLS, NO GOLD LABELS)")
    print("=" * 74)
    print(f"Question records : {len(questions)}")
    print(f"Official ID range: {next(iter(questions))} to {next(reversed(questions))}")
    print(f"Question types   : {dict(Counter(v['question_type'] for v in questions.values()))}")
    print(f"Chunks           : {len(chunks)} | Source docs: {len({c['doc_id'] for c in chunks})}")
    print(f"Chunk SHA256     : {chunk_sha}")
    stat = previous_run_stats()
    if stat:
        print("\nExisting A5.3 model-time statistics (estimate only):")
        for key, value in stat.items():
            if key != "warning":
                print(f"  {key}: {value}")
        print("  ", stat["warning"])
    else:
        print("\nNo A5.3 generation timing records found. No runtime estimate available.")
    print("\nPreflight passed. This command did not run inference or create predictions.")


def main():
    parser = argparse.ArgumentParser(description="A6.3: reproducible, resumable RAG batch runner")
    parser.add_argument("--mode", choices=("preflight", "pilot", "full"), default="preflight")
    parser.add_argument("--count", type=int, default=20, help="Pilot size; divisible by 4")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--confirm-full-run", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--unique-docs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-context-chars", type=int, default=8500)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()

    if args.mode == "full" and not args.confirm_full_run:
        parser.error("Full 2556-question run requires --confirm-full-run")
    if args.top_k < 1 or args.num_predict < 1 or args.num_ctx < 1024 or args.timeout < 1 or args.retries < 0:
        parser.error("Invalid retrieval/generation arguments")
    if args.max_context_chars < 1000 or not 0 <= args.temperature <= 2:
        parser.error("Invalid context limit or temperature")
    if not hasattr(rag_pipeline, "first_chunk_per_document"):
        raise RuntimeError("Install the A5.2 rag_pipeline.py with --unique-docs support")

    questions = load_queries()  # never exposes answer/evidence_list to inference
    chunks, chunk_sha, _ = load_chunks()
    if len(questions) != EXPECTED or len(chunks) != 4573 or len({c["doc_id"] for c in chunks}) != 609:
        raise ValueError("Unexpected official question/chunk/document counts")
    if list(questions) != [f"q{i:04d}" for i in range(EXPECTED)]:
        raise ValueError("Question order/IDs changed")

    if args.mode == "preflight":
        print_preflight(questions, chunks, chunk_sha)
        return

    if args.mode == "pilot":
        ids = choose_pilot(questions, args.count, args.seed)
        run_dir = OUT / f"a6_3_pilot_{args.count}_seed{args.seed}"
    else:
        ids = list(questions)
        run_dir = OUT / "a6_3_full"

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    trace_path = run_dir / "traces.jsonl"
    pred_path = run_dir / "predictions.json"
    report_path = run_dir / "report.json"
    settings = build_settings(args, ids, chunk_sha)

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            old = json.load(f)
        if old.get("settings") != settings:
            raise ValueError("Manifest settings/source hash mismatch. Do not mix runs; rename the old output directory.")
    else:
        if trace_path.exists():
            raise ValueError("Trace exists but manifest is missing; refusing to overwrite")
        write_json_atomic(manifest_path, {"created_at_utc": timestamp(), "settings": settings,
                                          "warning": "Predictions generated without gold labels; official score not computed"})
    saved = load_checkpoint(trace_path, set(ids))

    print("=" * 74)
    print(f"A6.3 {args.mode.upper()} - REAL RAG BATCH (NO GOLD DATA)")
    print("=" * 74)
    print(f"Questions: {len(ids)}, saved: {len(saved)}, remaining: {len(ids)-len(saved)}")
    print(f"Model: {args.model} | Top-K={args.top_k} | unique_docs={args.unique_docs}")
    print(f"Output directory: {run_dir}")

    retriever = BM25Retriever(chunks) if len(saved) < len(ids) else None
    pipeline_args = SimpleNamespace(
        top_k=args.top_k, unique_docs=args.unique_docs, max_context_chars=args.max_context_chars,
        model=args.model, base_url=args.base_url, num_ctx=args.num_ctx,
        num_predict=args.num_predict, temperature=args.temperature,
        timeout=args.timeout, retries=args.retries)
    with trace_path.open("a", encoding="utf-8") as f:
        for ix, qid in enumerate(ids, 1):
            if qid in saved:
                continue
            started = time.monotonic()
            record = run_one(qid, questions[qid], retriever, pipeline_args)
            record["end_to_end_seconds"] = round(time.monotonic() - started, 3)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            saved[qid] = record
            gen = record.get("generation") or {}
            print(f"[{ix:04d}/{len(ids):04d}] {qid} [{record['question_type']}]: {record['status']} "
                  f"in={gen.get('prompt_eval_count')} out={gen.get('eval_count')} "
                  f"time={record['end_to_end_seconds']:.1f}s", flush=True)
            if record.get("error"):
                print("  ERROR:", record["error"], flush=True)

    predictions = [
        {"question_id": qid,
         "model_answer": saved[qid]["model_answer"] if saved[qid]["status"] == "success" else ""}
        for qid in ids
    ]
    check_predictions(predictions, ids)
    if args.mode == "full" and len(predictions) != EXPECTED:
        raise ValueError("Incomplete full prediction output")
    write_json_atomic(pred_path, predictions)

    statuses = Counter(saved[qid]["status"] for qid in ids)
    total_time = sum(saved[qid].get("end_to_end_seconds", 0) for qid in ids)
    report = {
        "generated_at_utc": timestamp(), "mode": args.mode,
        "question_count": len(ids), "status_counts": dict(statuses),
        "elapsed_generation_seconds_sum": round(total_time, 2),
        "missing_or_failed_predictions": sum(saved[qid]["status"] != "success" for qid in ids),
        "data_and_parameters": settings,
        "warning": "No official qa_evaluate.py run. Error/incomplete answers become empty strings in exported predictions. Inspect traces."
    }
    write_json_atomic(report_path, report)
    print("\nComplete. Saved:")
    for p in (manifest_path, trace_path, pred_path, report_path):
        print(" ", p)
    print("Status counts:", dict(statuses))
    if any(saved[qid]["status"] != "success" for qid in ids):
        print("WARNING: Failed/incomplete answers were exported as empty strings. Review before submission.")
    print("NOT an official benchmark score.")


if __name__ == "__main__":
    main()
