"""No GPU/assets required: public wrapper and private replay configuration contracts."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import engine_args as wrapper_args
from conversation_cache_parity import engine_args, without_cache_options
from conversation_cache_disabled import common_args


class ForkCacheContracts(unittest.TestCase):
    def test_legacy_launch_arguments_are_unchanged(self):
        args = ['--pack', '/pack', '--prompt-cache-policy', 'balanced',
                '--prefix-cache-file', '/legacy/root.bin', '--conversation-cache-dir', '/legacy/chats',
                '--conversation-cache-mib', '8192', '--conversation-cache-slots', '4']
        cfg = {'args': list(args), 'gpu': 0}
        self.assertEqual(wrapper_args(cfg), args)
        self.assertEqual(cfg['args'], args)

    def test_replay_never_writes_production_stores(self):
        args = ['--pack', '/pack', '--prefix-cache-file', '/private/prefix',
                '--conversation-cache-dir', '/private/legacy', '--conversation-cache-mib', '8192',
                '--conversation-ram-cache-mib', '2048', '--conversation-cache-disk', '/private/shared',
                '--conversation-cache-disk-mib', '4096', '--prompt-cache-policy', 'balanced']
        cfg = {'args': list(args)}
        stripped = without_cache_options(args)
        self.assertEqual(stripped, ['--pack', '/pack', '--prompt-cache-policy', 'balanced'])
        for result in (engine_args(cfg, 1024, 4), common_args(cfg, 4)):
            self.assertFalse(any('/private/' in value for value in result))
            self.assertIn('balanced', result)
        self.assertEqual(cfg['args'], args)

    def test_replay_missing_cache_value_is_rejected(self):
        for name in ('--prefix-cache-file', '--conversation-cache-dir', '--conversation-ram-cache-mib'):
            with self.assertRaises(AssertionError):
                without_cache_options([name])

    def test_shared_tokenizer_identity_does_not_rewrite_legacy_options(self):
        args = ['--conversation-cache-dir', '/legacy', '--conversation-cache-mib', '8192',
                '--conversation-ram-cache-mib', '1024', '--conversation-cache-disk', '/shared']
        result = wrapper_args({'args': args, 'gpu': 0}, tokenizer=Path('/tokenizer'), template=Path('/template'))
        self.assertEqual(result[:len(args)], args)
        self.assertEqual(result[len(args):], ['--conversation-cache-tokenizer', '/tokenizer',
                                             '--conversation-cache-template', '/template'])


if __name__ == '__main__':
    unittest.main()
