"""Catch record schema collisions before loading a model or using a GPU."""
import ast
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / 'experiments/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/reward_aligned_reader_gsm8k_pilot_seed1234/experiment.py'


def return_fields(name):
    tree = ast.parse(SOURCE.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    returns = [n.value for n in ast.walk(function) if isinstance(n, ast.Return)]
    value = next(v for v in returns if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == 'dict')
    return {k.arg for k in value.keywords}


class RecordContract(unittest.TestCase):
    def test_reused_cache_identity_is_available_before_new_manifest(self):
        function = next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'source_row')
        def missing_manifest():
            raise AssertionError('Manifest does not exist during prepare')
        environment = {'CFG': {'source_cache_reuse_manifest_sha256': 'original-cache-identity'}, 'identity': missing_manifest}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), environment)
        row = environment['source_row']({'id': 'test', 'source_identity': 'PILOT_MANIFEST'})
        self.assertEqual(row['source_identity'], 'original-cache-identity')

    def test_completed_trajectory_preserves_token_ids_and_diagnostics(self):
        trajectory = {k: None for k in return_fields('sample')}
        trajectory.update(tokens=[10, 20, 30], completed=True)
        diagnostics = {k: None for k in return_fields('replay')}
        self.assertFalse(trajectory.keys() & diagnostics.keys())
        diagnostics['supervised_tokens'] = len(trajectory['tokens'])
        merged = dict(id='question', replicate=0, **trajectory, **diagnostics)
        self.assertEqual(merged['tokens'], [10, 20, 30])
        self.assertEqual(merged['supervised_tokens'], 3)

    def test_censored_trajectory_keeps_raw_token_ids(self):
        trajectory = {k: None for k in return_fields('sample')}
        trajectory.update(tokens=[10, 20], completed=False, correct=None)
        merged = dict(id='question', replicate=0, **trajectory, **{})
        self.assertEqual(merged['tokens'], [10, 20])
        self.assertIsNone(merged['correct'])


if __name__ == '__main__':
    unittest.main()
