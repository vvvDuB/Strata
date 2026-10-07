"""On-demand process lifetime, failure cleanup, memory leases and batch exclusion (no GPU required)."""
import contextlib
import os
from pathlib import Path
import struct
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from serve.server import Vision, VisionGate, Service, MockEngine, ByteTokenizer, ChatTemplate, ROOT, parse_server_options


WORKER = '''#!/usr/bin/env python3
import os, sys, struct, time
from pathlib import Path
Path("child.pid").write_text(str(os.getpid()))
mode = sys.argv[sys.argv.index("--model") + 1]
if mode == "crash": sys.exit(3)
print("READY 3", flush=True)
line = sys.stdin.readline()
if mode == "hang": time.sleep(60)
if mode == "error":
 print("ERR bad picture", flush=True); sys.exit(0)
out = Path(line.split()[2])
out.write_bytes(struct.pack("<5i6f", 0x31455653, 2, 2, 1, 3, *range(6)))
if mode == "invalid": out.write_bytes(b"broken")
print("OK 2 2 1 1", flush=True)
'''


@unittest.skipUnless(os.name == "posix", "executable Python fixture and child PID checks need POSIX")
class OnDemandProcess(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.parent = Path(self.temp.name)
        self.exe = self.parent / "encoder"
        self.exe.write_text(WORKER)
        self.exe.chmod(0o700)
        self.vision = None

    def tearDown(self):
        if self.vision:
            self.vision.shutdown()
        self.temp.cleanup()

    def make(self, mode="ok", **cfg):
        self.vision = Vision(dict(exe=str(self.exe), mmproj="unused", model=mode,
                                  mode="on-demand", cache_dir=str(self.parent), **cfg))
        return self.vision

    def assert_reaped(self, v):
        self.assertIsNone(v.proc)
        pid = int((v.dir / "child.pid").read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_no_child_at_start_restart_or_unload(self):
        with mock.patch("serve.server.popen") as spawn:
            v = self.make()
            v.restart()
            v.unload()
            self.assertFalse(v.alive())
            spawn.assert_not_called()

    def test_success_exits_before_restoration_and_cache_hit_never_borrows(self):
        v = self.make()
        events = []
        @contextlib.contextmanager
        def lease():
            events.append("borrow")
            yield
            self.assert_reaped(v)
            events.append("restore")
        v.memory_lease = lease
        first = v.encode(b"\x89PNG\r\n\x1a\nfixture")
        self.assertEqual(first[1], 2)
        self.assertEqual(struct.unpack("<5i", first[0].read_bytes()[:20]), (0x31455653, 2, 2, 1, 3))
        self.assertEqual(v.encode(b"\x89PNG\r\n\x1a\nfixture"), first)
        self.assertEqual(events, ["borrow", "restore"])
        self.assertFalse(list(v.dir.glob("*.img")))

    def test_worker_failures_leave_no_child_or_partial_embeddings(self):
        for mode in ("error", "crash", "invalid", "hang"):
            with self.subTest(mode=mode):
                v = self.make(mode, timeout_s=.15 if mode == "hang" else 5)
                with self.assertRaises(ValueError):
                    v.encode(b"\x89PNG\r\n\x1a\nfixture")
                self.assert_reaped(v)
                self.assertFalse(v.cache)
                self.assertFalse(list(v.dir.glob("*.sve")))
                self.assertFalse(list(v.dir.glob("*.img")))
                v.shutdown()

    def test_failed_memory_lease_never_starts_child(self):
        v = self.make()
        @contextlib.contextmanager
        def lease():
            raise ValueError("no GPU space")
            yield
        v.memory_lease = lease
        with mock.patch("serve.server.popen") as spawn:
            with self.assertRaisesRegex(ValueError, "no GPU space"):
                v.encode(b"\x89PNG\r\n\x1a\nfixture")
            spawn.assert_not_called()
        self.assertFalse(list(v.dir.glob("*.img")))


class MemoryLease(unittest.TestCase):
    def make(self, free=128):
        tok = ByteTokenizer()
        engine = MockEngine(tok, ["ok"])
        engine.vision_vram = mock.Mock(side_effect=[{"vram_free_mib": 2048}, {"expert_slots": 1000}])
        engine.unload = mock.Mock()
        vision = mock.Mock(on_demand=True, gpu=True, vram_mib=2048)
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=vision)
        svc.free_vram_mib = lambda: free
        return svc, engine

    def test_restores_exact_lease_even_when_encoding_fails(self):
        svc, engine = self.make()
        with self.assertRaisesRegex(ValueError, "encode failed"):
            with svc._vision_memory_lease():
                raise ValueError("encode failed")
        self.assertEqual(engine.vision_vram.call_args_list, [mock.call(2048), mock.call(None)])
        self.assertIsNone(svc.vram_reserve)
        engine.unload.assert_not_called()

    def test_no_resize_when_already_enough_free(self):
        svc, engine = self.make(free=4096)
        with svc._vision_memory_lease():
            pass
        engine.vision_vram.assert_not_called()

    def test_insufficient_shrink_is_restored_before_returning_error(self):
        svc, engine = self.make()
        engine.vision_vram.side_effect = [{"vram_free_mib": 1024}, {}]
        with self.assertRaisesRegex(ValueError, "cannot release"):
            with svc._vision_memory_lease():
                self.fail("must not start encoder")
        self.assertEqual(engine.vision_vram.call_args_list, [mock.call(2048), mock.call(None)])

    def test_restoration_failure_unloads_engine_instead_of_resuming_partial_cache(self):
        svc, engine = self.make()
        engine.vision_vram.side_effect = [{"vram_free_mib": 2048}, ValueError("restore failed")]
        with self.assertRaisesRegex(ValueError, "restore failed"):
            with svc._vision_memory_lease():
                pass
        engine.unload.assert_called_once()

    def test_incomplete_slot_restore_is_detected(self):
        svc, engine = self.make()
        engine.vision_vram.side_effect = [{"vram_free_mib": 2048, "expert_slots_before": 1000},
                                         {"expert_slots": 900}]
        with self.assertRaisesRegex(ValueError, "every expert cache slot"):
            with svc._vision_memory_lease():
                pass
        engine.unload.assert_called_once()

    def test_idle_encoder_does_not_restart_text_engine(self):
        svc, engine = self.make()
        svc.vision.alive.return_value = False
        engine.restart = mock.Mock()
        svc.ensure_loaded()
        engine.restart.assert_not_called()
        svc.vision.restart.assert_not_called()


class BatchExclusion(unittest.TestCase):
    def test_service_batch_generation_holds_shared_lease_until_it_finishes(self):
        tok = ByteTokenizer()
        active = threading.Barrier(3)
        release = threading.Event()
        encoded = threading.Event()
        failures = []
        class BlockingEngine(MockEngine):
            batch = 2
            def generate(self, *args, **kwargs):
                active.wait(timeout=5)
                release.wait(timeout=5)
                yield from super().generate(*args, **kwargs)
        engine = BlockingEngine(tok, ["ok"])
        vision = SimpleNamespace(on_demand=True, gpu=False, encode=lambda src: encoded.set())
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=vision)
        def text():
            try:
                list(svc.run(tok.encode("hello"), False, [], 8, {}, threading.Event()))
            except BaseException as e: failures.append(e)
        threads = [threading.Thread(target=text) for _ in range(2)]
        for t in threads: t.start()
        active.wait(timeout=5)
        image = threading.Thread(target=lambda: svc._encode_images([b"image"]))
        image.start()
        deadline = time.monotonic() + 5
        while not svc.vision_gate.writers and time.monotonic() < deadline: time.sleep(.01)
        self.assertEqual(svc.vision_gate.writers, 1)
        self.assertFalse(encoded.is_set())
        release.set()
        for t in [*threads, image]:
            t.join(timeout=5); self.assertFalse(t.is_alive())
        self.assertFalse(failures, failures)
        self.assertTrue(encoded.is_set())

    def test_encoder_waits_for_all_readers_and_blocks_new_text_admission(self):
        gate = VisionGate()
        cancel = threading.Event()
        active = threading.Barrier(3)
        release = threading.Event()
        encoded = threading.Event()
        finished = threading.Event()
        later = threading.Event()
        def text():
            with gate.shared(cancel) as acquired:
                self.assertTrue(acquired)
                active.wait(timeout=5)
                release.wait(timeout=5)
        threads = [threading.Thread(target=text) for _ in range(2)]
        for t in threads: t.start()
        active.wait(timeout=5)
        def image():
            with gate.exclusive(timeout=5):
                encoded.set()
                finished.wait(timeout=5)
        writer = threading.Thread(target=image); writer.start()
        deadline = time.monotonic() + 5
        while not gate.writers and time.monotonic() < deadline: time.sleep(.01)
        self.assertEqual(gate.writers, 1)
        def late_text():
            with gate.shared(cancel) as acquired:
                if acquired: later.set()
        late = threading.Thread(target=late_text); late.start()
        self.assertFalse(encoded.is_set())
        self.assertFalse(later.is_set())
        release.set()
        self.assertTrue(encoded.wait(timeout=5))
        self.assertFalse(later.is_set())
        finished.set()
        for t in [*threads, writer, late]:
            t.join(timeout=5); self.assertFalse(t.is_alive())
        self.assertTrue(later.is_set())

    def test_cancelled_reader_does_not_hold_encoder(self):
        gate = VisionGate()
        cancel = threading.Event(); cancel.set()
        with gate.shared(cancel) as acquired: self.assertFalse(acquired)
        with gate.exclusive(timeout=.1): self.assertEqual(gate.readers, 0)

    def test_cancelled_image_does_not_clear_another_exclusive_owner(self):
        gate = VisionGate()
        cancel = threading.Event(); cancel.set()
        with gate.exclusive():
            with gate.exclusive(cancel=cancel) as acquired:
                self.assertFalse(acquired)
            self.assertTrue(gate.encoding)

    def test_image_decode_excludes_batch_text_until_positions_are_no_longer_used(self):
        tok = ByteTokenizer()
        entered = threading.Event()
        release = threading.Event()
        text_entered = threading.Event()
        errors = []
        class Engine(MockEngine):
            batch = 2
            def generate(self, *args, **kwargs):
                if kwargs.get("embeddings"):
                    entered.set(); release.wait(timeout=5)
                else:
                    text_entered.set()
                yield from super().generate(*args, **kwargs)
        svc = Service(Engine(tok, ["ok"]), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"),
                      vision=SimpleNamespace(on_demand=True, gpu=False))
        with tempfile.TemporaryDirectory() as tmp:
            emb = Path(tmp) / "req.sve"; emb.touch()
            def run(image=False):
                try:
                    if image: svc.embeddings.path = str(emb)
                    list(svc.run(tok.encode("hello"), False, [], 8, {}, threading.Event()))
                except BaseException as e: errors.append(e)
            image = threading.Thread(target=lambda: run(True)); image.start()
            self.assertTrue(entered.wait(timeout=5))
            text = threading.Thread(target=run); text.start()
            self.assertFalse(text_entered.wait(timeout=.1))
            release.set()
            for t in (image, text):
                t.join(timeout=5); self.assertFalse(t.is_alive())
            self.assertTrue(text_entered.is_set())
            self.assertFalse(errors, errors)


class VisionCLI(unittest.TestCase):
    def test_direct_on_demand_preserves_native_args(self):
        native = ["--serve", "--vision", "--vram-elastic", "--vram-reserve-mib", "640"]
        _, _, cfg = parse_server_options(["--engine", "strata", "--exe", "engine", "--tokenizer", "tok",
            "--vision-exe", "encoder", "--vision-mmproj", "mmproj.gguf", "--vision-model", "model.gguf",
            "--vision-mode", "on-demand", "--vision-gpu", "--", *native])
        self.assertEqual(cfg["args"], native)
        self.assertEqual(cfg["vision"]["mode"], "on-demand")
        self.assertTrue(cfg["vision"]["gpu"])


if __name__ == "__main__":
    unittest.main()
