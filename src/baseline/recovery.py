"""A6.3 full-run recovery: retry ONLY error/incomplete questions, then merge.

Usage (from NLP_RAG root):
  python src/a6_3_recover_full.py --dry-run
  python src/a6_3_recover_full.py --num-predict-incomplete 384

This program reads original, actually-sent Ollama requests from traces.jsonl.
It never reads gold answers or evidence_list and never modifies original files.
Recovery is append-only, and reruns skip successful recovery records while
reattempting previous failed recovery records.
"""
import argparse
import copy
import hashlib
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = PROJECT_ROOT / "outputs" / "baseline_runs" / "full"
EXPECTED = 2556


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path, allow_recovery_history=False):
    latest = {}
    seen = set()
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank line in {path}:{line_no}")
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("question_id"), str):
                raise ValueError(f"Bad record in {path}:{line_no}")
            qid = record["question_id"]
            if qid in seen and not allow_recovery_history:
                raise ValueError(f"Duplicate question_id {qid} in {path}:{line_no}")
            seen.add(qid)
            latest[qid] = record
    return latest


def write_json_atomic(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(tmp, path)


def iso_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def check_ollama(url):
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"Invalid preserved Ollama URL: {url}")
    base = f"{parts.scheme}://{parts.netloc}"
    with urlopen(base + "/api/tags", timeout=8) as response:
        if response.status != 200:
            raise OSError(f"Unexpected Ollama health status: {response.status}")
    return base


def invoke_original_request(original, num_predict_incomplete, timeout, attempts_total, pause):
    saved = original.get("llm_request")
    if not isinstance(saved, dict) or not isinstance(saved.get("body"), dict) or not saved.get("url"):
        return {"status": "error", "error": "Missing original llm_request; cannot safely reconstruct prompt",
                "attempts": []}

    body = copy.deepcopy(saved["body"])
    previous_limit = (body.get("options") or {}).get("num_predict")
    if original["status"] == "incomplete":
        if not isinstance(previous_limit, int) or num_predict_incomplete <= previous_limit:
            return {"status": "error", "error": f"New num_predict must exceed original {previous_limit}",
                    "attempts": []}
        body["options"]["num_predict"] = num_predict_incomplete
    # For request failures keep exactly the original model, context, prompt and num_predict.

    request = {"url": saved["url"], "body": body}
    attempts = []
    for n in range(1, attempts_total + 1):
        started = time.monotonic()
        try:
            req = Request(request["url"], method="POST",
                          headers={"Content-Type": "application/json"},
                          data=json.dumps(body, ensure_ascii=False).encode("utf-8"))
            with urlopen(req, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict) or payload.get("done") is not True:
                raise ValueError("Ollama did not return a completed JSON result")
            if not isinstance(payload.get("response"), str) or not payload["response"].strip():
                raise ValueError("Ollama returned an empty response")
            reason = payload.get("done_reason")
            status = "success" if reason == "stop" else "incomplete"
            attempts.append({"number": n, "status": "received", "seconds": round(time.monotonic()-started, 3)})
            return {"status": status, "error": None, "attempts": attempts,
                    "llm_request": request, "model_answer": payload["response"].strip(),
                    "generation": {k: payload.get(k) for k in (
                        "model", "done_reason", "prompt_eval_count", "eval_count",
                        "total_duration", "load_duration", "prompt_eval_duration", "eval_duration")}}
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
            detail = f"{type(exc).__name__}: {exc}"
            attempts.append({"number": n, "status": "error", "error": detail,
                             "seconds": round(time.monotonic()-started, 3)})
            if isinstance(exc, HTTPError) and exc.code < 500:
                break  # Non-retryable request validation/authorization error
            if n < attempts_total:
                time.sleep(pause * (2 ** (n - 1)))

    return {"status": "error", "error": attempts[-1]["error"],
            "attempts": attempts, "llm_request": request, "model_answer": ""}


def validate_full(originals, predictions, manifest):
    if manifest.get("settings", {}).get("mode") != "full":
        raise ValueError("This script accepts A6.3 FULL run only")
    expected_ids = [f"q{i:04d}" for i in range(EXPECTED)]
    if len(predictions) != EXPECTED or [x.get("question_id") for x in predictions] != expected_ids:
        raise ValueError("Original predictions must contain exactly q0000..q2555 in order")
    if set(originals) != set(expected_ids):
        raise ValueError("Original traces are missing or duplicate question IDs")
    for item in originals.values():
        if item.get("status") not in ("success", "error", "incomplete"):
            raise ValueError(f"Unknown original status: {item.get('status')}")


def main():
    parser = argparse.ArgumentParser(description="Recover only failed/incomplete A6.3 full-run answers")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--num-predict-incomplete", type=int, default=384)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--attempts", type=int, default=4, help="HTTP attempts per question")
    parser.add_argument("--backoff", type=float, default=2.0, help="Initial backoff seconds")
    parser.add_argument("--dry-run", action="store_true", help="Inspect errors without calling Ollama or writing files")
    args = parser.parse_args()
    if args.num_predict_incomplete < 1 or args.timeout < 1 or args.attempts < 1 or args.backoff < 0:
        parser.error("Invalid numeric option")

    run_dir = args.run_dir.resolve()
    trace_path, pred_path, manifest_path = (run_dir / n for n in ("traces.jsonl", "predictions.json", "manifest.json"))
    for p in (trace_path, pred_path, manifest_path):
        if not p.is_file():
            raise FileNotFoundError(p)
    originals = read_jsonl(trace_path)
    predictions = read_json(pred_path)
    manifest = read_json(manifest_path)
    validate_full(originals, predictions, manifest)
    ids = [r["question_id"] for r in predictions]
    candidates = [qid for qid in ids if originals[qid]["status"] != "success"]
    status_counts = Counter(x["status"] for x in originals.values())
    missing_req = [qid for qid in candidates if not (originals[qid].get("llm_request") or {}).get("body")]
    error_types = Counter((originals[qid].get("error") or "Unknown")[:100] for qid in candidates
                          if originals[qid]["status"] == "error")
    print(f"Original: {EXPECTED} questions | statuses={dict(status_counts)}")
    print(f"Recovery candidates={len(candidates)} | preserved requests missing={len(missing_req)}")
    print("Original error types (up to five):")
    for message, count in error_types.most_common(5):
        print(f"  {count} × {message}")

    folder = run_dir / f"recovery_errors_original_incomplete_{args.num_predict_incomplete}"
    manifest_out = folder / "manifest.json"
    retry_path = folder / "recovery_traces.jsonl"
    merged_path = folder / "predictions_review.json"
    report_path = folder / "report.json"
    meta = {"source_traces_sha256": sha256_file(trace_path),
            "source_manifest_sha256": sha256_file(manifest_path),
            "source_predictions_sha256": sha256_file(pred_path),
            "candidate_ids": candidates,
            "original_status_counts": dict(status_counts),
            "incomplete_num_predict": args.num_predict_incomplete,
            "request_timeout_seconds": args.timeout,
            "http_attempts_per_question": args.attempts, "backoff_seconds": args.backoff,
            "note": "Only original error/incomplete records are retried using their saved prompts; no gold labels accessed."}
    if manifest_out.exists() and read_json(manifest_out) != meta:
        raise ValueError("Existing recovery manifest mismatch. Do not mix configs or source runs")
    if not manifest_out.exists() and retry_path.exists():
        raise ValueError("Recovery traces exist but manifest is missing")
    history = read_jsonl(retry_path, allow_recovery_history=True)
    if any(qid not in candidates for qid in history):
        raise ValueError("Recovery trace contains an unexpected question ID")
    pending = [qid for qid in candidates if history.get(qid, {}).get("status") != "success"]
    print(f"Already recovered={len(candidates)-len(pending)} | remaining={len(pending)}")
    if args.dry_run:
        print("DRY RUN: no files written, no model inference")
        if missing_req:
            print("Missing preserved request IDs:", ", ".join(missing_req[:15]))
        return

    # Check the API endpoint before starting many requests. Handles service-offline cases early.
    for qid in pending:
        saved = originals[qid].get("llm_request") or {}
        if saved.get("url"):
            base = check_ollama(saved["url"])
            print("Ollama API reachable:", base)
            break
    folder.mkdir(parents=True, exist_ok=True)
    if not manifest_out.exists():
        write_json_atomic(manifest_out, meta)
    consecutive_connection_errors = 0
    with retry_path.open("a", encoding="utf-8") as out:
        for index, qid in enumerate(pending, 1):
            previous = history.get(qid)
            attempt_run = (previous.get("recovery_run_number", 0) + 1) if previous else 1
            started = time.monotonic()
            result = invoke_original_request(originals[qid], args.num_predict_incomplete,
                                             args.timeout, args.attempts, args.backoff)
            result.update({"question_id": qid, "original_status": originals[qid]["status"],
                           "recovery_run_number": attempt_run, "created_at_utc": iso_now(),
                           "end_to_end_seconds": round(time.monotonic()-started, 3)})
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            os.fsync(out.fileno())
            history[qid] = result
            gen = result.get("generation") or {}
            print(f"[{index:03d}/{len(pending):03d}] {qid}: {result['status']} "
                  f"output={gen.get('eval_count')} reason={gen.get('done_reason')} "
                  f"time={result['end_to_end_seconds']:.1f}s", flush=True)
            if result.get("error"):
                print("  ERROR:", result["error"], flush=True)
            # Stop if the server has evidently gone away; rerun this command after service recovery.
            error_text = result.get("error") or ""
            connection_related = any(t in error_text.lower() for t in (
                "connection refused", "connection reset", "timed out", "timeout", "remote end closed"))
            consecutive_connection_errors = consecutive_connection_errors + 1 if connection_related else 0
            if consecutive_connection_errors >= 5:
                print("Stopping after 5 consecutive connection-related failures; check Ollama, then rerun.")
                break

    merged = []
    for qid in ids:
        src = originals[qid]
        if src["status"] == "success":
            answer = src.get("model_answer", "")
        elif history.get(qid, {}).get("status") == "success":
            answer = history[qid]["model_answer"]
        else:
            answer = ""
        merged.append({"question_id": qid, "model_answer": answer})
    write_json_atomic(merged_path, merged)
    recovered = sum(history.get(qid, {}).get("status") == "success" for qid in candidates)
    empty_ids = [r["question_id"] for r in merged if not r["model_answer"].strip()]
    report = {"created_at_utc": iso_now(), "question_count": EXPECTED,
              "original_status_counts": dict(status_counts), "recovery_candidates": len(candidates),
              "recovered_success": recovered, "remaining_unrecovered": len(candidates)-recovered,
              "recovery_latest_status_counts": dict(Counter(history[qid]["status"] for qid in history)),
              "empty_answer_ids": empty_ids, "empty_answers": len(empty_ids),
              "ready_for_official_evaluation": len(empty_ids) == 0,
              "original_files_untouched": True,
              "note": "Preview only; inspect outputs and run official preparation/scorer after recovery."}
    write_json_atomic(report_path, report)
    print("\nRecovery report:", report_path)
    print("Merged preview:", merged_path)
    print(f"Recovered: {recovered}/{len(candidates)} | still pending: {len(candidates)-recovered}")
    print(f"Empty merged predictions: {len(empty_ids)}")
    print("This script never modifies original traces or predictions.")


if __name__ == "__main__":
    main()
