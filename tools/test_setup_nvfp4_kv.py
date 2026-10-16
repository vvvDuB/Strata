"""NVFP4 launcher tests. Hardware, builds and downloads are mocked; no GPU claim."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import setup
from test_setup_golden import PROFILES, install


class Nvfp4Setup(unittest.TestCase):
    def run_install(self, context, streaming='auto', build=True):
        ram, found = PROFILES['128GB-1x24GB']
        argv = ['--family', 'qwen', '--model', 'Q2_0', '--no-start', '--vision', 'none',
                '--context', str(context), '--kv', 'nvfp4', '--kv-streaming', streaming]
        if build:
            argv += ['--build']
        return install(ram, found, argv, extra=[mock.patch.object(setup, 'build_engine',
                       side_effect=lambda *a, **k: setup.ROOT / 'engine')])

    def test_main_and_drafter_memory(self):
        self.assertEqual(setup.kv_budget_bytes_per_token('nvfp4'), 8160)
        self.assertEqual(setup.kv_budget_bytes_per_token('int8'), 13728)
        self.assertEqual(setup.kv_budget_bytes_per_token('q4_0'), 7488)
        self.assertEqual(setup.kv_budget_bytes_per_token('k8v4'), 13728)  # legacy estimate retained

    def test_resident_only_at_long_context(self):
        for mode in ('auto', 'on', 'off'):
            with self.subTest(mode=mode):
                rc, out, cfg, _ = self.run_install(131072, mode)
                self.assertEqual(rc, 0, out)
                a = cfg['args']
                self.assertEqual(a[a.index('--kv')+1], 'nvfp4')
                self.assertEqual(a[a.index('--kv-resident')+1], '0')
                self.assertIn('experimental NVFP4 K+V', out)

    def test_small_context_honors_explicit_codec(self):
        rc, out, cfg, _ = self.run_install(8192)
        self.assertEqual(rc, 0, out)
        self.assertEqual(cfg['args'][cfg['args'].index('--kv')+1], 'nvfp4')

    def test_source_build_is_required(self):
        rc, out, cfg, _ = self.run_install(131072, build=False)
        self.assertEqual(rc, 1)
        self.assertIn('requires --build', out)
        self.assertIsNone(cfg)

    def test_codec_survives_config_choice_reload(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'strata-q2_0.json'
            path.write_text(json.dumps({'args':['--kv','nvfp4','--max-context','131072']}))
            self.assertEqual(setup.choices_from_config(path)['kv'], 'nvfp4')


if __name__ == '__main__':
    unittest.main()
