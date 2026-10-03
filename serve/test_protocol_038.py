import unittest
from serve.server import StrataEngine
class Done038(unittest.TestCase):
 def parse(self, tail):
  e=StrataEngine.__new__(StrataEngine);e._parse_done('DONE 0 10026 125.0 0.0 cancel 0 0 8192 9 10 '+tail);return e.last
 def test_stock_read_position(self):
  d=self.parse('11 12 13.5 1024');self.assertEqual(d.get('prompt_read'),1024)
 def test_owner_keyed_read_position(self):
  d=self.parse('10000 64 checkpoint cache_select_ms=2.5 prompt_read=1024 ram_blobs=11 file_blobs=12 file_mb=13.5');self.assertEqual(d.get('prompt_read'),1024);self.assertEqual(d['cache_source'],'checkpoint')
