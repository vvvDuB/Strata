"""CLI opt-in/compatibility checks; no model is loaded by these probes."""
import os
import subprocess
import unittest
from pathlib import Path
BIN = Path(os.environ.get('STRATA_TEST_BIN', Path(__file__).resolve().parents[1]/'build/strata'))

class Nvfp4Cli(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run([str(BIN), *args], capture_output=True, text=True, timeout=20)

    def test_help_advertises_opt_in_nvfp4(self):
        p = self.invoke('--help')
        self.assertEqual(p.returncode, 0)
        self.assertTrue('--kv nvfp4' in p.stdout+p.stderr, 'NVFP4 CLI missing')

    def test_nvfp4_rejects_streamed_kv_before_model_load(self):
        p = self.invoke('--kv','nvfp4','--kv-resident','20480','--tokens','1')
        self.assertNotEqual(p.returncode, 0)
        self.assertTrue('--kv nvfp4 requires --kv-resident 0' in p.stdout+p.stderr,
                        'resident-only NVFP4 guard missing')

    def test_invalid_storage_is_still_rejected(self):
        p = self.invoke('--kv','unknown-format','--tokens','1')
        self.assertNotEqual(p.returncode, 0)
        self.assertTrue('--kv must be fp16, int8, q4_0, k8v4 or nvfp4' in p.stdout+p.stderr,
                        'supported storage enum/guard incomplete')

if __name__ == '__main__': unittest.main()
