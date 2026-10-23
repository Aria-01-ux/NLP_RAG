"""Unified A6 BASELINE entry point: preflight, pilot, full, recover, surface, audit, evaluate.

Packages the original A5/A6 logic with renamed modules; does not change
retrieval ranking, prompt, model options, retry requests or the official evaluator.

Run from NLP_RAG root, e.g.:
    python src/baseline/baseline.py preflight
    python src/baseline/baseline.py pilot --count 4 --seed 20261024
    python src/baseline/baseline.py surface --predictions outputs/.../predictions.json --output-dir outputs/.../surface_export

Production logic is split intentionally: newly packaged BM25/retrieval/generation,
original batch/recovery scripts, and this lightweight orchestration & export.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "baseline"
HISTORIC_SOURCES = ROOT / "archive" / "a5_a6_legacy_20261023" / "frozen_engines"
CONFIG = ROOT / "config" / "a6_baseline_frozen.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def atomic_json(path: Path, data: object) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temp.replace(path)


def get_config() -> dict:
    if not CONFIG.is_file():
        raise FileNotFoundError(f"Frozen config missing: {CONFIG}")
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    for name, expected in data["source_sha256"].items():
        if expected is not None:
            actual = sha256(HISTORIC_SOURCES / name)
            if actual != expected:
                raise RuntimeError(
                    f"Historical engine changed: {name}\n"
                    f"expected={expected}\nactual={actual}\n"
                    "Restore original version before any A6 generation; do not overwrite historic manifests."
                )
    # Verify constants from the locked source without requiring packages or model calls.
    tree = ast.parse((SRC / "bm25_retriever.py").read_text(encoding="utf-8"))
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("K1", "B", "EPSILON"):
                    constants[target.id] = node.value.value
    for name, key in [("K1", "k1"), ("B", "b"), ("EPSILON", "epsilon")]:
        if constants.get(name) != data["retrieval"][key]:
            raise RuntimeError(f"Frozen BM25 parameter changed: {name}")
    return data


def subprocess_command(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def build_generation_args(cfg: dict) -> list[str]:
    r, g = cfg["retrieval"], cfg["generation"]
    return [
        "--top-k", str(r["top_k"]),
        "--unique-docs" if r["unique_docs"] else "--no-unique-docs",
        "--model", g["model"], "--base-url", g["base_url"],
        "--num-ctx", str(g["num_ctx"]), "--num-predict", str(g["num_predict"]),
        "--temperature", str(g["temperature"]),
        "--max-context-chars", str(r["max_context_chars"]),
        "--timeout", str(g["timeout_seconds"]),
        "--retries", str(g["retries"]),
    ]


def surface_export(predictions: Path, output_dir: Path) -> None:
    # Import ONLY the original gold-free Surface operation, not A6.4's research CLI.
    from surface import surface_repair
    source, target = predictions.resolve(), output_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {target}")
    rows = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty prediction list")
    seen, surface_rows, audit = set(), [], []
    reasons = Counter()
    for pos, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"question_id", "model_answer"}:
            raise ValueError(f"Invalid record fields at index {pos}")
        qid, answer = row["question_id"], row["model_answer"]
        if not isinstance(qid, str) or len(qid) != 5 or not qid.startswith("q") or not qid[1:].isdigit() or qid in seen:
            raise ValueError(f"Invalid/duplicate question ID: {qid!r}")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Empty prediction {qid}; recover failed generations first")
        seen.add(qid)
        formatted, rule = surface_repair(answer)
        surface_rows.append({"question_id": qid, "model_answer": formatted})
        reasons[rule] += 1
        if answer != formatted:
            audit.append({"question_id": qid, "raw_answer": answer,
                          "surface_answer": formatted, "rule": rule})
    target.mkdir(parents=True)
    raw_path = target / "predictions_raw.json"
    surface_path = target / "predictions_surface.json"
    shutil.copyfile(source, raw_path)
    atomic_json(surface_path, surface_rows)
    atomic_json(target / "surface_audit.json", audit)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "A6.3 -> gold-free A6.4 Surface normalization (not semantic scoring)",
        "question_count": len(rows), "changed_count": len(audit),
        "rule_counts": dict(sorted(reasons.items())),
        "input_predictions_sha256": sha256(source),
        "raw_copy_sha256": sha256(raw_path),
        "surface_predictions_sha256": sha256(surface_path),
        "a6_4_source_sha256": sha256(HISTORIC_SOURCES / "a6_4_output_baseline.py"),
        "active_surface_source_sha256": sha256(SRC / "surface.py"),
        "entrypoint_source_sha256": sha256(Path(__file__)),
        "no_gold_labels_or_evidence_list_used": True,
    }
    atomic_json(target / "surface_report.json", report)
    assert sha256(source) == sha256(raw_path)
    print("A6 SURFACE EXPORT: PASS (no Ollama; no gold)")
    print(f"Questions: {len(rows)} | changed: {len(audit)} | reasons: {dict(reasons)}")
    for name in ("predictions_raw.json", "predictions_surface.json", "surface_audit.json", "surface_report.json"):
        print("Saved:", target / name)


def evaluate(predictions: Path, dest: Path) -> None:
    # Never import the gold dataset in an inference path. Only call official tools here.
    predictions, dest = predictions.resolve(), dest.resolve()
    if not predictions.is_file():
        raise FileNotFoundError(predictions)
    if dest.exists():
        raise FileExistsError(f"Refusing overwrite of evaluation directory: {dest}")
    rows = json.loads(predictions.read_text(encoding="utf-8-sig"))
    ids = [f"q{i:04d}" for i in range(2556)]
    if not isinstance(rows, list) or [r.get("question_id") for r in rows] != ids:
        raise ValueError("Official evaluation requires exactly q0000..q2555 in order")
    dest.mkdir(parents=True)
    converted = dest / "qa_submission.json"
    queryfile = ROOT / "data" / "raw" / "MultiHopRAG.json"
    subprocess_command([sys.executable, str(ROOT / "eval_tutorial" / "prepare_eval.py"),
                        "--predictions", str(predictions), "--queries", str(queryfile),
                        "--output", str(converted)])
    command = [sys.executable, str(ROOT / "eval_tutorial" / "qa_evaluate.py"),
               "--file", str(converted), "--queries", str(queryfile)]
    print("Running:", " ".join(command), flush=True)
    with (dest / "official_eval.txt").open("w", encoding="utf-8") as log:
        # stderr is included to preserve the official evaluator's progress output.
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, check=True)
    print((dest / "official_eval.txt").read_text(encoding="utf-8"))
    print("Official evaluation exported:", converted, dest / "official_eval.txt")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unified A6 Baseline command entry (historical sources archived)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight", help="Check A6 data/index without Ollama")
    p = sub.add_parser("pilot", help="Stratified pilot: real BM25 + Ollama")
    p.add_argument("--count", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261024)
    p = sub.add_parser("full", help="Run/resume full A6.3 2556-question batch")
    p.add_argument("--confirm-full-run", action="store_true")
    p = sub.add_parser("recover", help="Use original saved HTTP requests to retry full-run errors")
    p.add_argument("--confirm-recovery", action="store_true", help="Actually call Ollama; without this flag, dry-run")
    p.add_argument("--num-predict-incomplete", type=int, default=384)
    p = sub.add_parser("surface", help="Export raw/surface/audit, with no scoring or gold")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p = sub.add_parser("audit-history", help="Audit original 2556-question A6.3->A6.4 results (no model calls)")
    p.add_argument("--export", action="store_true", help="Export historical frozen predictions; default check only")
    p.add_argument("--output-dir", type=Path)
    p = sub.add_parser("evaluate", help="Official scoring only; never runs as part of generation")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    sub.add_parser("config", help="Print locked generation/retrieval config; no calls")
    args = parser.parse_args(argv)
    try:
        cfg = get_config()
        if args.command == "config":
            print(json.dumps(cfg, indent=2, ensure_ascii=False))
        elif args.command in ("preflight", "pilot", "full"):
            call = [sys.executable, str(SRC / "batch_runner.py")]
            if args.command == "preflight":
                call += ["--mode", "preflight"]
            elif args.command == "pilot":
                call += ["--mode", "pilot", "--count", str(args.count), "--seed", str(args.seed)]
                call += build_generation_args(cfg)
            else:
                if not args.confirm_full_run:
                    parser.error("Full run requires --confirm-full-run; refuses accidental 2556 Ollama calls")
                call += ["--mode", "full", "--confirm-full-run"] + build_generation_args(cfg)
            subprocess_command(call)
        elif args.command == "recover":
            call = [sys.executable, str(SRC / "recovery.py"),
                    "--num-predict-incomplete", str(args.num_predict_incomplete)]
            if not args.confirm_recovery:
                call.append("--dry-run")
            subprocess_command(call)
        elif args.command == "surface":
            surface_export(args.predictions, args.output_dir)
        elif args.command == "audit-history":
            call = [sys.executable, str(SRC / "historical_audit.py"),
                    "--mode", "export" if args.export else "check"]
            if args.output_dir is not None:
                call += ["--output-dir", str(args.output_dir)]
            elif args.export:
                parser.error("--export requires --output-dir (must be new or empty)")
            subprocess_command(call)
        elif args.command == "evaluate":
            evaluate(args.predictions, args.output_dir)
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError, KeyError, AssertionError) as exc:
        print(f"A6 BASELINE FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
