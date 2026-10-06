"""Source contract gates; device arithmetic is tested by the CMake GPU suites."""
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

class Nvfp4KernelContract(unittest.TestCase):
    def prompt_source(self):
        return (ROOT/'src/kernels/cuda/qsa_prompt_attn.cu').read_text()

    def test_exact_vector_unpack_is_available_and_used(self):
        self.assertTrue((ROOT/'include/strata/kernels/kv_nvfp4_unpack4.cuh').is_file(),
                        'the exact four-code unpack helper is not integrated')
        text = self.prompt_source()
        self.assertEqual(text.count('nvfp4::unpack_twice_e2m1_4('), 4)

    def test_value_scale_maxima_are_shared_before_scores(self):
        text = self.prompt_source()
        self.assertTrue('struct NVScaleStore<5>' in text, 'shared NVFP4 scales missing')
        self.assertIn('__shfl_xor_sync(0xffffffffu, vmax, o, 8)', text)
        self.assertLess(text.index('S.nvup[sg] ='), text.index('// scores: warp'))
        self.assertIn('vup = S.nvup[vg]', text)
        self.assertIn('vdown_g[gi] = S.nvdown[vg]', text)

    def test_query_fragments_are_cached_before_selected_cell_loop(self):
        text = self.prompt_source()
        self.assertTrue('qreg_hi[KV_MODE == 5 ? 16 : 1][4]' in text, 'query fragment cache missing')
        start = text.index('qreg_hi[g][0] =')
        loop = text.index('for (int c0 = 0; c0 < n; c0 += CH)')
        self.assertLess(start, loop)
        self.assertIn('__syncthreads(); // query halves ready', text[:start])
        self.assertIn('ah[j] = qreg_hi[g][j]', text[loop:])
        self.assertIn('al[j] = qreg_lo[g][j]', text[loop:])

    def test_exhaustive_unpack_device_gate_is_registered(self):
        text = (ROOT/'CMakeLists.txt').read_text()
        self.assertTrue('add_executable(kv_nvfp4_unpack_test' in text, 'exhaustive CUDA unpack gate missing')

    def test_sanitizer_runner_covers_exhaustive_unpack_gate(self):
        text = (ROOT/'tools/run_nvfp4_gpu_checks.sh').read_text()
        self.assertEqual(text.count('for test in kv_nvfp4_native_conversion kv_nvfp4_gpu_test kv_nvfp4_unpack_test; do'), 2,
                         'unpack gate must run normally and under all four sanitizers')

    def test_opt_in_cli_and_resident_only_guard_are_integrated(self):
        text = (ROOT/'src/program/generate.cpp').read_text()
        self.assertTrue('--kv nvfp4' in text, 'NVFP4 CLI missing')
        self.assertIn('nvfp4', (ROOT/'src/core/mtp.cpp').read_text())

if __name__ == '__main__': unittest.main()
