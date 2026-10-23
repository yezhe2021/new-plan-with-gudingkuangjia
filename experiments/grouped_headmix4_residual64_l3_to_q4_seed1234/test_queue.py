import hashlib
import json
import tempfile
import unittest
import fcntl
from pathlib import Path
from unittest.mock import patch
from run_pipeline import dependency_ready, readiness


class QueueTests(unittest.TestCase):
    def test_dependency_requires_success_and_released_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertFalse(dependency_ready(root)[0])
            (root / 'manifest.json').write_text('{}')
            (root / 'completed.json').write_text(json.dumps(dict(status='completed', manifest_sha256=hashlib.sha256(b'{}').hexdigest())))
            with (root / 'pipeline.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertEqual(dependency_ready(root)[1], 'previous_pipeline_still_exiting')
            self.assertTrue(dependency_ready(root)[0])
            (root / 'failure.json').write_text('{}')
            self.assertEqual(dependency_ready(root)[1], 'previous_experiment_failed')

    def test_failed_and_unprepared_jobs_cannot_start(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(readiness(root, {})['reason'], 'not_prepared')
            (root / 'failure.json').write_text('{}')
            self.assertEqual(readiness(root, {})['reason'], 'failed_requires_review')

    def test_completion_identity_must_match(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'manifest.json').write_text('{}')
            (root / 'completed.json').write_text('{"status":"completed","manifest_sha256":"wrong"}')
            self.assertEqual(dependency_ready(root)[1], 'previous_completion_identity_mismatch')


if __name__ == '__main__':
    unittest.main()
