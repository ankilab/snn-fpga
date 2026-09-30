"""Failure-path tests for unattended campaign acceptance and resume."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

spec = importlib.util.spec_from_file_location('campaign', Path(__file__).resolve().parents[2] / 'tools/qualify_references.py')
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class CampaignTests(unittest.TestCase):
    def test_failed_process_is_recorded_and_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            run = campaign.Run(Path(directory), {'source': 'test'}, 1)
            with patch.object(campaign.subprocess, 'Popen', return_value=Mock(wait=Mock(return_value=2))):
                with self.assertRaises(RuntimeError):
                    run.step('build', ['fake-tool'])
            self.assertEqual(json.loads(run.path.read_text())['stages']['build']['status'], 'failed')
            with patch.object(campaign.subprocess, 'Popen', return_value=Mock(wait=Mock(return_value=0))) as process:
                run.step('build', ['fake-tool'])
                run.step('build', ['fake-tool'])
                self.assertEqual(process.call_count, 1)

    def test_validator_failure_is_not_accepted_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            run = campaign.Run(Path(directory), {}, 1)
            with patch.object(campaign.subprocess, 'Popen', return_value=Mock(wait=Mock(return_value=0))):
                with self.assertRaises(ValueError):
                    run.step('gate', ['fake-tool'], validate=Mock(side_effect=ValueError('mismatch')))
            self.assertEqual(json.loads(run.path.read_text())['stages']['gate']['status'], 'failed')

    def test_resume_rejects_changed_or_missing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'artifact'
            p.write_bytes(b'original')
            record = {'status': 'passed', 'key': 'source', 'outputs': {str(p): campaign.digest(p)}}
            self.assertTrue(campaign.reusable(record, 'source'))
            self.assertFalse(campaign.reusable(record, 'other-source'))
            p.write_bytes(b'changed')
            self.assertFalse(campaign.reusable(record, 'source'))
            p.unlink()
            self.assertFalse(campaign.reusable(record, 'source'))

    def test_gate_checks_rows_even_if_summary_claims_success(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'gate.jsonl'
            rows = [{'index': 0, 'label': 1, 'prediction': 1, 'reference_prediction': 2,
                     'matches_reference': True}, {'samples': 1, 'accuracy': 1,
                     'reference_accuracy': 1, 'reference_mismatches': 0, 'no_prediction': 0}]
            p.write_text('\n'.join(map(json.dumps, rows)))
            with self.assertRaises(ValueError):
                campaign.qualification(p, 1, True, 1)
            rows[0]['reference_prediction'] = 1
            p.write_text('\n'.join(map(json.dumps, rows)))
            campaign.qualification(p, 1, True, 1)
            with self.assertRaises(ValueError):
                campaign.qualification(p, 2, True, 1)

    def test_full_accuracy_and_no_prediction_are_recomputed(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'full.jsonl'
            rows = [{'index': 0, 'label': 1, 'prediction': None},
                    {'samples': 1, 'accuracy': 0, 'no_prediction': 1}]
            p.write_text('\n'.join(map(json.dumps, rows)))
            campaign.qualification(p, 1, False, 0)
            with self.assertRaises(ValueError):
                campaign.qualification(p, 1, False, 0.5)
            rows[-1]['no_prediction'] = 0
            p.write_text('\n'.join(map(json.dumps, rows)))
            with self.assertRaises(ValueError):
                campaign.qualification(p, 1, False, 0)


if __name__ == '__main__':
    unittest.main()
