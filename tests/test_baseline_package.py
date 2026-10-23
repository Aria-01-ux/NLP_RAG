"""Offline regression for the separated baseline package; never calls Ollama."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'/'baseline'))
import baseline
import batch_runner
import rag_pipeline
import bm25_retriever
from baseline import get_config, surface_export, sha256

class BaselinePackageTests(unittest.TestCase):
    def test_frozen_config(self):
        cfg=get_config()
        self.assertEqual(cfg['retrieval']['top_k'],10)
        self.assertEqual(cfg['generation']['num_predict'],256)
        self.assertEqual(bm25_retriever.PROJECT_ROOT,ROOT)
        self.assertEqual(batch_runner.ROOT,ROOT)
        self.assertEqual(batch_runner.OUT,ROOT/'outputs/baseline_runs')

    def test_gold_free_surface(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'pred.json'
            path.write_text(json.dumps([
              {'question_id':'q0000','model_answer':'Yes, agreed.'},
              {'question_id':'q0001','model_answer':'Insufficient Information'},
            ]))
            out=Path(folder)/'surface'
            surface_export(path,out)
            self.assertEqual(sha256(path),sha256(out/'predictions_raw.json'))
            formatted=json.loads((out/'predictions_surface.json').read_text())
            self.assertEqual(formatted[0]['model_answer'],'Yes\nagreed.')
            self.assertEqual(formatted[1]['model_answer'],'Insufficient Information')
            with self.assertRaises(FileExistsError):surface_export(path,out)

    def test_pilot_uses_new_batch_path_and_frozen_args(self):
        with patch.object(baseline,'subprocess_command') as fake:
            self.assertEqual(baseline.main(['pilot','--count','4','--seed','20261025']),0)
        c=fake.call_args.args[0]
        self.assertEqual(Path(c[1]),ROOT/'src/baseline/batch_runner.py')
        self.assertEqual(c[c.index('--num-predict')+1],'256')
        self.assertIn('--unique-docs',c)

    def test_original_prompt_unchanged(self):
        self.assertIn('Answer the QUESTION based only on the EVIDENCE below.',rag_pipeline.make_user_prompt('X','Y'))
        self.assertEqual(batch_runner.TYPES,('inference_query','comparison_query','temporal_query','null_query'))

    def test_full_surface_equivalence(self):
        source=ROOT/'outputs/a6_3_full/recovery_errors_original_incomplete_384/predictions_review.json'
        expected=ROOT/'outputs/a6_4_format_baseline/predictions_surface.json'
        if not source.is_file() or not expected.is_file():
            self.skipTest('Historical predictions not installed')
        with tempfile.TemporaryDirectory() as d:
            surface_export(source,Path(d)/'surface')
            self.assertEqual(sha256(Path(d)/'surface/predictions_surface.json'),sha256(expected))

if __name__=='__main__':unittest.main(verbosity=2)
