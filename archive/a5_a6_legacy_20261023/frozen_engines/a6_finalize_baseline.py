"""A6 frozen baseline: audit and export A6.3 + recovery + A6.4 Surface.

No LLM calls; no reretrieval; no answer, evidence_list or question_type used
in answering or surface formatting. The course evaluator is run separately.

From project root:
  python src/a6_finalize_baseline.py --mode check
  python src/a6_finalize_baseline.py --mode export
"""
import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from a6_4_output_baseline import surface_repair

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/a6_3_full"
REC = RUN / "recovery_errors_original_incomplete_384"
A64 = ROOT / "outputs/a6_4_format_baseline"
OUT = ROOT / "outputs/a6_final_baseline"
N = 2556


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_trace(path, allow_history=False):
    check(path.is_file(), f'Missing trace: {path}')
    rows = {}
    count = 0
    with path.open(encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            check(bool(line.strip()), f'Blank JSONL line {path}:{i}')
            row = json.loads(line)
            check(isinstance(row, dict) and isinstance(row.get('question_id'), str),
                  f'Invalid trace row {path}:{i}')
            qid = row['question_id']
            check(allow_history or qid not in rows, f'Duplicate trace ID {qid} in {path}')
            rows[qid] = row
            count += 1
    return rows, count


def atomic_json(path, data):
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')
    os.replace(temp, path)


def evidence_block(row, index):
    return (f"[Evidence {index}]\nchunk_id: {row['chunk_id']}\n"
            f"doc_id: {row['doc_id']}\ntitle: {row['title']}\n"
            f"source: {row['source']}\npublished_at: {row['published_at']}\n"
            f"url: {row['url']}\ntext:\n{row['text']}")


def expected_context(retrieval, max_chars):
    ranked = retrieval['results']
    if retrieval['selection_strategy'] == 'unique_documents':
        selected = []
        seen = set()
        skipped = []
        for row in ranked:
            if row['doc_id'] in seen:
                skipped.append(row['chunk_id'])
            else:
                seen.add(row['doc_id'])
                selected.append(row)
    else:
        selected = ranked
        skipped = []
    blocks, used = [], []
    for row in selected:
        block = evidence_block(row, len(used) + 1)
        if len('\n\n'.join(blocks + [block])) > max_chars:
            break
        blocks.append(block)
        used.append(row['chunk_id'])
    check(not selected or bool(used), 'Top-1 evidence exceeds context limit')
    excluded = [row['chunk_id'] for row in selected[len(used):]]
    return ('\n\n'.join(blocks) if blocks else 'No relevant evidence was retrieved.',
            used, skipped, excluded)


def audit_and_build():
    paths = {
        'run_manifest': RUN/'manifest.json', 'run_report': RUN/'report.json',
        'original_traces': RUN/'traces.jsonl', 'original_predictions': RUN/'predictions.json',
        'recovery_manifest': REC/'manifest.json', 'recovery_report': REC/'report.json',
        'recovery_traces': REC/'recovery_traces.jsonl',
        'recovered_predictions': REC/'predictions_review.json',
        'a64_surface': A64/'predictions_surface.json',
        'a64_report': A64/'format_report.json',
    }
    for name, path in paths.items():
        check(path.is_file(), f'Missing {name}: {path}')

    manifest = read_json(paths['run_manifest'])
    settings = manifest['settings']
    check(settings['mode'] == 'full', 'A6.3 manifest is not a full run')
    ids = [f'q{i:04d}' for i in range(N)]
    check(settings['question_ids'] == ids, 'Manifest has unexpected question IDs')
    sources = {
        'chunks_sha256': ROOT/'data/processed/chunks.jsonl',
        'questions_sha256': ROOT/'data/raw/MultiHopRAG.json',
        'pipeline_source_sha256': ROOT/'src/rag_pipeline.py',
        'retriever_source_sha256': ROOT/'src/bm25_retriever.py',
        'runner_source_sha256': ROOT/'src/a6_3_batch_runner.py',
    }
    for key, path in sources.items():
        check(path.is_file() and digest(path) == settings[key],
              f'Source changed since A6.3 original run: {path}')
    check(settings['retriever']['type'] == 'BM25Okapi', 'Unexpected retriever')
    check(settings['retriever']['top_k'] == 10 and settings['retriever']['unique_docs'] is True,
          'Retrieval configuration differs from frozen baseline')

    recmanifest = read_json(paths['recovery_manifest'])
    for field, p in [('source_traces_sha256', paths['original_traces']),
                     ('source_manifest_sha256', paths['run_manifest']),
                     ('source_predictions_sha256', paths['original_predictions'])]:
        check(recmanifest[field] == digest(p), f'Recovery source hash mismatch: {field}')
    original, n_orig = read_trace(paths['original_traces'])
    recovery, n_hist = read_trace(paths['recovery_traces'], allow_history=True)
    check(n_orig == N and len(original) == N and list(original) == ids,
          'Original traces must cover exactly q0000..q2555 in order')
    check(len(recovery) == n_hist, 'Recovery history contains duplicated IDs; review manually')
    raw_pred = read_json(paths['original_predictions'])
    rec_pred = read_json(paths['recovered_predictions'])
    old_surface = read_json(paths['a64_surface'])
    for name, rows in [('original', raw_pred), ('recovery', rec_pred), ('surface', old_surface)]:
        check(isinstance(rows, list) and len(rows) == N and
              all(isinstance(r, dict) and set(r) == {'question_id','model_answer'}
                  and r['question_id'] == ids[i] and isinstance(r['model_answer'], str)
                  for i, r in enumerate(rows)), f'Invalid {name} predictions')
    rec_ids = [qid for qid in ids if original[qid]['status'] != 'success']
    check(recmanifest['candidate_ids'] == rec_ids, 'Recovery candidate IDs mismatch')
    check(set(recovery) == set(rec_ids), 'Recovery traces have missing/extra questions')
    check(read_json(paths['recovery_report'])['remaining_unrecovered'] == 0,
          'Recovery report has incomplete cases')
    check(read_json(paths['a64_report'])['input_predictions_sha256'] ==
          digest(paths['recovered_predictions']), 'A6.4 formatted a different source')

    chunk_source = {}
    with sources['chunks_sha256'].open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            check(row['chunk_id'] not in chunk_source, 'Duplicated chunk in corpus')
            chunk_source[row['chunk_id']] = row

    statuses, surface_reasons, used_counts, final_pred, raw_answers, status_rows = (
        Counter(), Counter(), Counter(), [], [], [])
    for i, qid in enumerate(ids):
        origin = original[qid]
        state = origin['status']
        gen = origin.get('generation') or {}
        if state == 'success':
            check(gen.get('done_reason') == 'stop', f'{qid}: success but non-stop generation')
        elif state == 'incomplete':
            check(gen.get('done_reason') == 'length', f'{qid}: incomplete but unexpected reason')
        else:
            check(state == 'error', f'{qid}: unknown state {state}')
        original_answer = origin['model_answer'] if state == 'success' else ''
        check(raw_pred[i]['model_answer'] == original_answer,
              f'{qid}: original prediction mismatch')

        retrieval = origin.get('retrieval')
        req = origin.get('llm_request') or {}
        body = req.get('body') or {}
        check(isinstance(retrieval, dict) and isinstance(body, dict) and body,
              f'{qid}: missing original retrieval/request')
        check(body.get('model') == settings['generator']['model'], f'{qid}: wrong model')
        options = body.get('options') or {}
        for key in ('num_ctx', 'num_predict', 'temperature'):
            check(options.get(key) == settings['generator'][key],
                  f'{qid}: unexpected original generation setting {key}')
        check(body.get('system') == origin.get('system_prompt') and
              body.get('prompt') == origin.get('user_prompt'),
              f'{qid}: sent prompt differs from trace')
        check(retrieval['returned_count'] == len(retrieval['results']) <=
              settings['retriever']['top_k'], f'{qid}: invalid Top-K results')
        for j, row in enumerate(retrieval['results'], 1):
            check(row['rank'] == j, f'{qid}: retrieval rank mismatch')
            chunk = chunk_source.get(row['chunk_id'])
            check(chunk is not None, f'{qid}: unknown Chunk {row["chunk_id"]}')
            check(all(row.get(k) == chunk.get(k) for k in
                      ('chunk_id','doc_id','text','title','source','published_at','url')),
                  f'{qid}: retrieved Chunk changed')
        context, used, skipped, excluded = expected_context(
            retrieval, settings['generator']['max_context_chars'])
        check(retrieval['used_chunk_ids'] == used and
              retrieval['skipped_chunk_ids_same_document'] == skipped and
              retrieval['excluded_chunk_ids_due_to_char_budget'] == excluded,
              f'{qid}: evidence selection differs')
        check(origin.get('context') == context, f'{qid}: evidence text differs')
        prompt = (f"Answer the QUESTION based only on the EVIDENCE below.\n\n"
                  f"QUESTION:\n{origin['query']}\n\nEVIDENCE:\n{context}\n\nSHORT ANSWER:")
        check(body['prompt'] == prompt, f'{qid}: actual prompt differs from stored query/evidence')

        if state == 'success':
            final = origin
            source = 'original'
        else:
            final = recovery[qid]
            source = 'recovery'
            check(final['status'] == 'success' and
                  (final.get('generation') or {}).get('done_reason') == 'stop',
                  f'{qid}: recovery did not complete normally')
            recovered_body = final['llm_request']['body']
            expected_body = json.loads(json.dumps(body))
            if state == 'incomplete':
                expected_body['options']['num_predict'] = recmanifest['incomplete_num_predict']
            check(recovered_body == expected_body,
                  f'{qid}: recovery changed prompt/model or unexpected options')
        answer = final['model_answer']
        check(isinstance(answer, str) and answer.strip(), f'{qid}: final answer empty')
        check(rec_pred[i]['model_answer'] == answer, f'{qid}: recovered prediction mismatch')
        formatted, reason = surface_repair(answer)
        check(old_surface[i]['model_answer'] == formatted,
              f'{qid}: A6.4 Surface cannot be reproduced')
        final_pred.append({'question_id': qid, 'model_answer': formatted})
        raw_answers.append({'question_id': qid, 'model_answer': answer})
        status_rows.append({
            'question_id': qid, 'question_type_for_reporting_only': origin['question_type'],
            'original_status': state, 'final_status': 'success',
            'final_source': source, 'final_done_reason': final['generation']['done_reason'],
            'evidence_chunk_ids': used, 'evidence_count': len(used),
            'surface_repair_rule': reason,
        })
        statuses[state] += 1
        surface_reasons[reason] += 1
        used_counts[len(used)] += 1

    summary = {
        'baseline': 'A6.3 BM25+Ollama -> A6.3 recovery -> A6.4 Surface',
        'original_question_count': N, 'original_status_counts': dict(statuses),
        'recovered_count': len(rec_ids), 'final_nonempty_answers': len(final_pred),
        'surface_repair_counts': dict(surface_reasons),
        'evidence_chunks_per_question': {str(k):v for k,v in sorted(used_counts.items())},
        'source_files_sha256': {key:digest(path) for key,path in {**sources,**paths}.items()},
        'generation_settings_from_A6_3': settings['generator'],
        'retrieval_settings_from_A6_3': settings['retriever'],
        'verification': 'All source hashes, evidence blocks, prompts, statuses, recovery joins, and A6.4 Surface outputs match.',
        'limitations': ['No semantic verification is implied by evidence integrity.',
                        'Surface normalization does not improve retrieval or model reasoning.',
                        'No answer labels or gold evidence used in answer construction.',
                        'Official metric must be computed separately with the pinned evaluator.']
    }
    return final_pred, raw_answers, status_rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['check','export'], default='check')
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    surface, raw, statuses, report = audit_and_build()
    print('A6 FROZEN BASELINE INTEGRITY: PASS')
    print('Questions:', len(surface), '| original statuses:', report['original_status_counts'])
    print('Recovery:', report['recovered_count'], '| surface repair:',report['surface_repair_counts'])
    print('Gold-free construction; no Ollama calls; no BM25 change; no scoring performed.')
    if args.mode == 'check':
        print('CHECK ONLY: no files written.')
        return
    dest = args.output_dir.resolve()
    check(dest not in (RUN.resolve(), REC.resolve(), A64.resolve()) and
          not dest.is_relative_to(RUN.resolve()), 'Output must not overwrite baseline inputs')
    check(not dest.exists() or not any(dest.iterdir()),
          f'Output directory already contains files; refusing overwrite: {dest}')
    dest.mkdir(parents=True, exist_ok=True)
    atomic_json(dest/'predictions_raw.json',raw)
    atomic_json(dest/'predictions_surface.json',surface)
    with (dest/'final_status.jsonl').open('w',encoding='utf-8') as f:
        for row in statuses:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    report['exported_at_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    report['finalizer_source_sha256'] = digest(Path(__file__))
    report['exported_files_sha256'] = {
        name:digest(dest/name) for name in
        ('predictions_raw.json','predictions_surface.json','final_status.jsonl')}
    atomic_json(dest/'freeze_report.json',report)
    for name in ('predictions_raw.json','predictions_surface.json',
                 'final_status.jsonl','freeze_report.json'):
        print('Exported:', dest/name)


if __name__ == '__main__':
    main()
