"""A6.4: Reproducible, reference-answer-free Baseline answer-format experiment.

Inputs: previously generated predictions and ONLY question text from official QA data.
Outputs: two NEW prediction files (surface repair vs. concise answer) and audit JSON.
The original predictions, retrieval traces, and model calls are never changed.
No reference answer or evidence_list is consulted for any formatting decision.
Use the course's pinned evaluator independently; never substitute this audit for it.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / 'outputs/a6_3_full/recovery_errors_original_incomplete_384/predictions_review.json'
DEFAULT_QA = ROOT / 'data/raw/MultiHopRAG.json'
DEFAULT_OUTPUT = ROOT / 'outputs/a6_4_format_baseline'
EXPECTED_QA_SHA256 = '03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff'

LEADING_BOOLEAN = re.compile(r'^\s*(yes|no)([,.:;!?]+)(?=\s|$)', re.I)
YES_NO_QUERY = re.compile(r'^\s*(?:do|does|did|is|are|was|were|has|have|had|can|could|will|would|should|may)\b', re.I)
FINAL_ANSWER = re.compile(r'^\s*(?:final\s+answer|short\s+answer|answer)\s*:\s*(.*?)\s*$', re.I)
CANONICAL_BOOL = re.compile(r'^\s*(yes|no)\s*[,.!?;:]*\s*$', re.I)
CANONICAL_UNKNOWN = re.compile(r'^\s*insufficient\s+information\s*[.!?]*\s*$', re.I)
SINGLE_LETTER = re.compile(r'^[A-Za-z]$')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def surface_repair(answer):
    """Minimal surface change: make a leading Yes/No an isolated whitespace token.

    Preserve the rest of the response, including any later self-correction.
    Do not infer a new Yes/No label or use question labels.
    """
    s = answer.strip()
    m = LEADING_BOOLEAN.match(s)
    if not m:
        return s, 'unchanged'
    first = m.group(1).capitalize()
    tail = s[m.end():].lstrip()
    return first + ('\n' + tail if tail else ''), 'leading_boolean_punctuation'


def concise_answer(answer, question):
    """Conservative, question-dependent final-answer extraction, no gold labels.

    Prefer explicitly marked final answers over an earlier preliminary assertion.
    When the question is visibly binary, use a first-word yes/no only if no final
    marker exists. Otherwise do not guess from arbitrary prose.
    """
    s = answer.strip()
    if CANONICAL_UNKNOWN.fullmatch(s):
        return 'Insufficient Information', 'canonical_abstention'

    # An explicit final answer after the model's own self-correction takes precedence.
    marked = [m.group(1).strip() for line in s.splitlines()
              if (m := FINAL_ANSWER.match(line)) and m.group(1).strip()]
    if marked:
        last = marked[-1]
        b = CANONICAL_BOOL.fullmatch(last)
        if b:
            return b.group(1).capitalize(), 'explicit_final_boolean'
        if CANONICAL_UNKNOWN.fullmatch(last):
            return 'Insufficient Information', 'explicit_final_abstention'
        if len(last.split()) <= 8:
            return last.rstrip('.!? '), 'explicit_short_final_answer'

    if YES_NO_QUERY.match(question):
        m = LEADING_BOOLEAN.match(s)
        if m:
            return m.group(1).capitalize(), 'leading_boolean_for_yesno_question'
        b = CANONICAL_BOOL.fullmatch(s)
        if b:
            return b.group(1).capitalize(), 'canonical_boolean'

    if s and len(s.split()) <= 5 and s[-1] in '.!?':
        return s.rstrip('.!? '), 'short_terminal_punctuation'
    return s, 'unchanged'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--queries', type=Path, default=DEFAULT_QA)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.predictions.is_file() or not args.queries.is_file():
        raise FileNotFoundError('Missing input predictions or official question file')
    if digest(args.queries) != EXPECTED_QA_SHA256:
        raise ValueError('Official question file SHA256 mismatch')
    original = json.loads(args.predictions.read_text(encoding='utf-8-sig'))
    raw_qa = json.loads(args.queries.read_text(encoding='utf-8'))
    if not isinstance(original, list) or len(original) != 2556 or len(raw_qa) != 2556:
        raise ValueError('Expected exactly 2556 predictions and questions')

    # Strip references immediately: ONLY query text is available to the formatter.
    queries = [item['query'] for item in raw_qa]
    del raw_qa
    surface_rows, concise_rows, audit_rows = [], [], []
    changes = Counter()
    for i, (row, query) in enumerate(zip(original, queries)):
        qid = f'q{i:04d}'
        if not isinstance(row, dict) or set(row) != {'question_id', 'model_answer'}:
            raise ValueError(f'Unexpected prediction fields at {qid}')
        if row['question_id'] != qid or not isinstance(row['model_answer'], str):
            raise ValueError(f'Question ID/type mismatch at {qid}')
        raw = row['model_answer']
        surface, surface_reason = surface_repair(raw)
        concise, concise_reason = concise_answer(raw, query)
        surface_rows.append({'question_id': qid, 'model_answer': surface})
        concise_rows.append({'question_id': qid, 'model_answer': concise})
        changes['surface_' + surface_reason] += 1
        changes['concise_' + concise_reason] += 1
        if surface != raw or concise != raw or SINGLE_LETTER.fullmatch(raw.strip()):
            audit_rows.append({
                'question_id': qid,
                'raw_answer': raw,
                'surface_answer': surface,
                'concise_answer': concise,
                'surface_rule': surface_reason,
                'concise_rule': concise_reason,
                'suspicious_single_letter': bool(SINGLE_LETTER.fullmatch(raw.strip())),
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_dir.resolve() == args.predictions.parent.resolve():
        raise ValueError('Use a separate output directory to preserve original predictions')
    out = {
        'surface': args.output_dir / 'predictions_surface.json',
        'concise': args.output_dir / 'predictions_concise.json',
        'audit': args.output_dir / 'format_audit.json',
        'report': args.output_dir / 'format_report.json',
    }
    for key, data in [('surface', surface_rows), ('concise', concise_rows), ('audit', audit_rows)]:
        out[key].write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    report = {
        'purpose': 'Formatting experiment only; NOT official evaluation',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'input_predictions_sha256': digest(args.predictions),
        'official_question_sha256': digest(args.queries),
        'questions': len(original),
        'surface_changed': sum(x['model_answer'] != y['model_answer'] for x, y in zip(original, surface_rows)),
        'concise_changed': sum(x['model_answer'] != y['model_answer'] for x, y in zip(original, concise_rows)),
        'one_letter_original': sum(bool(SINGLE_LETTER.fullmatch(x['model_answer'].strip())) for x in original),
        'rule_counts': dict(sorted(changes.items())),
        'assurance': 'No gold answer or evidence_list is used to select or modify any answer.',
        'warning': 'Surface/concise outputs are separate diagnostic candidates, not automatic submission replacements.',
    }
    out['report'].write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('A6.4 format-only Baseline comparison: NO LLM calls, NO GOLD-DRIVEN REWRITES')
    print(f"Questions: {len(original)} | one-letter raw anomalies: {report['one_letter_original']}")
    print(f"Surface changed: {report['surface_changed']} | Concise changed: {report['concise_changed']}")
    for key in ('surface', 'concise', 'audit', 'report'):
        print(f'{key}: {out[key]}')
    print('Original predictions untouched. Run the pinned course scorer for either variant.')


if __name__ == '__main__':
    main()
