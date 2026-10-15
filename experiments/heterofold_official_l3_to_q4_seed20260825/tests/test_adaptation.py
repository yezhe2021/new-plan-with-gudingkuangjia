import os
import unittest
from unittest.mock import patch

import torch

from heterofold.modeling import runtime_dtype
from heterofold.registry import MODELS, parse_pair, resolve_model


class AdaptationTest(unittest.TestCase):
    def test_precision_is_explicit_and_default_preserved(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(runtime_dtype(), torch.bfloat16)
        with patch.dict(os.environ, {'HETEROFOLD_DTYPE': 'float16'}):
            self.assertEqual(runtime_dtype(), torch.float16)
        with patch.dict(os.environ, {'HETEROFOLD_DTYPE': 'invalid'}):
            with self.assertRaises(ValueError):
                runtime_dtype()

    def test_real_alias_requires_local_model(self):
        self.assertEqual(parse_pair('llama3_to_qwen4'), ('llama3', 'qwen4'))
        self.assertIn('3.2-3B', MODELS['llama3'].repo)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                resolve_model('llama3')
