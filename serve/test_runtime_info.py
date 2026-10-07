"""The monitor must receive native cache resizes during a request without taking its token/control replies."""
import io
import queue
import threading
import unittest
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service, StrataEngine


class RuntimeInfo(unittest.TestCase):
    def setUp(self):
        self.engine = StrataEngine("unused", ["--max-context", "131072"], lazy=True)
        self.engine.info.update(version="0.1.40", expert_slots=3354, expert_cache_mib=5611)
        self.engine.proc = mock.Mock()
        self.engine.proc.poll.return_value = None
        self.engine.proc.stdin = io.StringIO()
        self.engine.ended = self.engine.unloaded = False
        self.engine.lines = queue.Queue()
        self.engine.slot_q = [queue.Queue()]
        self.native = queue.Queue()
        started = threading.Event()
        def output():
            started.set()
            yield from iter(self.native.get, None)
        self.engine.proc.stdout = output()
        self.pump = threading.Thread(target=self.engine._pump)
        self.engine.pump = self.pump
        self.pump.start()
        self.assertTrue(started.wait(2))
        self.svc = Service(self.engine, ByteTokenizer(), ChatTemplate(Path(__file__).parent / "chat_template.jinja"))

    def tearDown(self):
        self.native.put(None)
        self.pump.join(timeout=2)
        self.assertFalse(self.pump.is_alive())

    def update(self, slots, mib, cells, reply="T 7\n"):
        self.native.put(f"INFO expert_slots={slots} expert_cache_mib={mib} kv_cells={cells} kv_grow=1\n")
        self.native.put(reply)
        # The reply is after INFO on the same pipe, proving telemetry has landed by the time it is received.
        replies = self.engine.slot_q[0] if reply.startswith("BT ") else self.engine.lines
        self.assertEqual(replies.get(timeout=2), reply)
        self.assertTrue(self.engine.lines.empty())
        self.assertTrue(self.engine.slot_q[0].empty())

    def test_monitor_tracks_grow_trim_and_vision_while_busy(self):
        self.svc.status.update(busy=True, prompt_tokens=40045)
        for slots, mib, cells in [(3256, 5440, 40960), (2478, 4160, 40960),
                                  (3256, 5440, 40960), (3354, 5611, 8192)]:
            self.update(slots, mib, cells)
            facts = self.svc.metrics()["engine"]
            self.assertEqual((facts["expert_slots"], facts["expert_cache_mib"], facts["kv_cells"]),
                             (slots, mib, cells))
            self.assertEqual(facts["version"], "0.1.40")
            self.assertEqual(facts["max_context"], 131072)

    def test_runtime_info_does_not_enter_batch_token_queue(self):
        self.update(3256, 5440, 40960, "BT 0 7\n")
        self.assertEqual(self.engine.info["expert_slots"], 3256)

    def test_old_process_cannot_overwrite_restarted_engine(self):
        self.engine.proc = mock.Mock()
        self.update(100, 200, 131072)
        self.assertEqual(self.engine.info["expert_slots"], 3354)

    def test_vram_reply_updates_monitor_bytes_and_preserves_token_reply(self):
        self.native.put("INFO expert_slots=2478 expert_cache_mib=4160 kv_cells=40960\n")
        self.native.put("VRAM reserve_mib=1408 expert_slots=2478 expert_cache_mib=4160\n")
        result = self.engine.vision_vram(1408)
        self.assertEqual(result["expert_slots"], 2478)
        self.assertEqual(self.svc.metrics()["engine"]["expert_cache_mib"], 4160)
        self.assertEqual(self.engine.proc.stdin.getvalue(), "VRAM BEGIN 1408\n")


if __name__ == "__main__":
    unittest.main()
