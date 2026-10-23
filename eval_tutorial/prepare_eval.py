"""Validate course predictions and convert IDs to the official evaluator format.

This utility does not calculate or change the official evaluation metric.
"""
import argparse
import hashlib
import json
from pathlib import Path

DATA_SHA256 = '03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, default=Path('predictions.json'))
    parser.add_argument('--queries', type=Path, default=Path('MultiHopRAG.json'))
    parser.add_argument('--output', type=Path, default=Path('qa_submission.json'))
    args = parser.parse_args()
    if args.output.resolve() in {args.predictions.resolve(), args.queries.resolve()}:
        raise ValueError('Output must not overwrite an input file.')
    raw = args.queries.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError('Question file differs from the pinned official file. Run download_eval_files.py.')
    questions = json.loads(raw)
    if len(questions) != 2556 or len({q['query'] for q in questions}) != 2556:
        raise ValueError('Expected 2556 unique official questions.')
    rows = json.loads(args.predictions.read_text(encoding='utf-8-sig'))
    if not isinstance(rows, list):
        raise ValueError('predictions.json must be a JSON array.')
    expected = {f'q{i:04d}' for i in range(len(questions))}
    answers = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'question_id', 'model_answer'}:
            raise ValueError('Each row must contain only question_id and model_answer.')
        qid = row['question_id']
        if not isinstance(qid, str) or qid not in expected:
            raise ValueError(f'Unknown question ID: {qid!r}')
        if qid in answers:
            raise ValueError(f'Duplicate question ID: {qid}')
        if not isinstance(row['model_answer'], str):
            raise ValueError(f'model_answer must be a string: {qid}')
        answers[qid] = row['model_answer']
    missing = sorted(expected - answers.keys())
    if missing:
        raise ValueError(f'Missing {len(missing)} IDs, e.g. {missing[:5]}. Keep failed questions with an empty string answer.')
    converted = [dict(query=q['query'], question_type=q['question_type'],
                      model_answer=answers[f'q{i:04d}']) for i, q in enumerate(questions)]
    args.output.write_text(json.dumps(converted, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    empty = sum(not a.strip() for a in answers.values())
    print(f'Validated {len(converted)} questions; empty answers: {empty}.')
    print(f'Created {args.output}. Empty answers remain in the denominator.')

if __name__ == '__main__':
    main()
