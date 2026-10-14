from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class UnifiedAlgorithmCliTest(unittest.TestCase):
    def test_explicit_unlimited_cap_overrides_config_for_both_cli_spellings(self):
        script = Path(__file__).resolve().parents[1] / 'scripts' / 'run_unified_algorithm.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            problem = root / 'problem.json'
            config = root / 'config.json'
            problem.write_text(json.dumps({'alternatives': [[1., 1., 1.], [0., 0., 0.]],
                                          'target_weights': [.2, .3, .5]}))
            config.write_text(json.dumps({'max_queries': 1, 'workers': 1}))
            for i, override in enumerate([['--max-queries', 'none'], ['--max-queries=none']]):
                with self.subTest(override=override):
                    output = root / f'result-{i}.json'
                    process = subprocess.run([sys.executable, str(script), '--input-json', str(problem),
                                              '--config-json', str(config), '--simulate', '--output-json', str(output),
                                              *override], capture_output=True, text=True)
                    self.assertEqual(process.returncode, 0, process.stderr)
                    result = json.loads(output.read_text())
                    self.assertIsNone(result['configuration']['max_queries'])
                    self.assertEqual(result['result']['selected_candidate'], 0)
                    self.assertEqual(result['result']['question_count'], 0)
                    self.assertTrue(result['benchmark_quality']['selection_is_correct'])


if __name__ == '__main__':
    unittest.main()
