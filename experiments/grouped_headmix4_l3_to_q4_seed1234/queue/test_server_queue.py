import unittest
from server_queue import classify


class ServerQueueTests(unittest.TestCase):
    def test_waits_for_dependency_or_resources(self):
        for reason in ['previous_experiment_not_completed', 'previous_pipeline_still_exiting',
                       'insufficient_gpu_memory', 'insufficient_disk']:
            self.assertEqual(classify(dict(launched=False, reason=reason)), 'waiting')

    def test_exits_after_start_or_completion(self):
        self.assertEqual(classify(dict(launched=True, pid=123)), 'started')
        self.assertEqual(classify(dict(reason='already_running', pid=123)), 'started')
        self.assertEqual(classify(dict(reason='already_completed')), 'completed')

    def test_no_retry_on_failure_or_unknown_status(self):
        for reason in ['previous_experiment_failed', 'failed_requires_review',
                       'previous_launch_exited_requires_review', 'not_prepared', 'unknown']:
            self.assertEqual(classify(dict(reason=reason)), 'blocked')


if __name__ == '__main__':
    unittest.main()
