"""Real HTTP disconnects must cancel a silent prefill and remove abandoned queued work."""
import json
import socket
import threading
import time
import unittest
from pathlib import Path
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

class SilentEngine(MockEngine):
    def __init__(self, tok):
        super().__init__(tok, '4')
        self.started = threading.Event()
        self.cancelled = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.calls += 1
        self.started.set()
        # No heartbeat / output while the "prefill" runs.
        until = time.monotonic() + 5
        while time.monotonic() < until and not self.release.is_set():
            if cancel.wait(.02):
                self.cancelled.set()
                return
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)

class Disconnect(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = SilentEngine(tok)
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent/'chat_template.jinja'))
        self.httpd = serve(self.svc, port=0)
        self.sockets = []

    def tearDown(self):
        self.engine.release.set()
        for s in self.sockets: s.close()
        self.httpd.shutdown()
        self.httpd.server_close()

    def connect(self, path, stream):
        body = json.dumps(dict(model='test',messages=[dict(role='user',content='hi')],max_tokens=4,stream=stream)).encode()
        sock = socket.create_connection(self.httpd.server_address,timeout=2)
        sock.sendall(f'POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n'.encode()+body)
        self.sockets.append(sock)
        return sock

    def abort(self, sock):
        sock.shutdown(socket.SHUT_RDWR)
        sock.close()

    def test_anthropic_silent_prefill(self):
        self.assert_disconnect('/v1/messages',True)

    def test_openai_silent_prefill(self):
        self.assert_disconnect('/v1/chat/completions',True)

    def test_nonstream_silent_prefill(self):
        self.assert_disconnect('/v1/messages',False)

    def assert_disconnect(self,path,stream):
        sock=self.connect(path,stream)
        self.assertTrue(self.engine.started.wait(2))
        self.abort(sock)
        self.assertTrue(self.engine.cancelled.wait(1),'disconnected silent request was not cancelled')

    def test_abandoned_queue_does_not_cancel_other_request_or_run(self):
        first=self.connect('/v1/messages',True)
        self.assertTrue(self.engine.started.wait(2))
        second=self.connect('/v1/messages',True)
        until=time.monotonic()+2
        while not self.svc.status['queued'] and time.monotonic()<until:time.sleep(.01)
        self.assertEqual(self.svc.status['queued'],1)
        self.abort(second)
        time.sleep(.4)
        self.assertFalse(self.engine.cancelled.is_set())
        self.assertTrue(self.svc.status['busy'])
        self.assertEqual(self.svc.status['queued'],0,'cancelled request still waiting for inference lock')
        self.engine.release.set()
        time.sleep(.3)
        self.assertEqual(self.engine.calls,1,'abandoned queued request executed')

    def test_status_finalized_before_next_request_can_start(self):
        svc = self.svc
        case = self
        class CheckedStatus(dict):
            def __setitem__(self, key, value):
                if key == 'busy' and value is False:
                    case.assertTrue(svc.fifo.locked(), 'old request may clear the next request status')
                super().__setitem__(key, value)
        svc.status = CheckedStatus(svc.status)
        self.engine.release.set()
        list(svc.run([1]*30, False, None, 4, {}, threading.Event()))

if __name__=='__main__':unittest.main()
