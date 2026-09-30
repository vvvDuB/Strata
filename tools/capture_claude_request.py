#!/usr/bin/env python3
"""Temporary Anthropic diagnostic receiver; no inference, no header logging."""
import argparse
import json
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

NOTICE = 'Richiesta acquisita per la diagnostica; inferenza non eseguita.'
MAX_BODY = 16 * 1024 * 1024

def make_server(directory: Path, host: str, port: int):
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # never log headers, query parameters, or prompt contents

        def json_response(self, status, value):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/health':
                self.json_response(200, {'status':'capture-only','inference':False})
            else:
                self.json_response(404, {'error':{'type':'not_found_error','message':'diagnostic receiver'}})

        def do_POST(self):
            if self.path.split('?')[0].rstrip('/') != '/v1/messages':
                self.json_response(404, {'error':{'type':'not_found_error','message':'only /v1/messages'}})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= MAX_BODY:
                    raise ValueError('invalid body size')
                req = json.loads(self.rfile.read(size))
                if not isinstance(req, dict):
                    raise ValueError('expected JSON object')
            except (ValueError, OSError):
                self.json_response(400, {'error':{'type':'invalid_request_error','message':'invalid JSON body'}})
                return
            identifier = uuid.uuid4().hex
            path = directory / f'request-{time.time_ns()}-{identifier[:8]}.json'
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w') as f:
                json.dump(req, f, ensure_ascii=False, indent=2)
                f.write('\n')
            print(f'captured {path.name}: bytes={size} messages={len(req.get("messages", []))} '
                  f'tools={len(req.get("tools", []))}', flush=True)
            message = {'id': 'msg_' + identifier, 'type': 'message', 'role': 'assistant',
                       'model': req.get('model', 'diagnostic-capture'),
                       'content': [{'type': 'text', 'text': NOTICE}], 'stop_reason': 'end_turn',
                       'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}
            try:
                if not req.get('stream'):
                    self.json_response(200, message)
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'close')
                self.end_headers()
                events = [
                    ('message_start', {'message': {**message, 'content': [], 'stop_reason': None}}),
                    ('content_block_start', {'index': 0, 'content_block': {'type':'text','text':''}}),
                    ('content_block_delta', {'index': 0, 'delta': {'type':'text_delta','text':NOTICE}}),
                    ('content_block_stop', {'index': 0}),
                    ('message_delta', {'delta': {'stop_reason':'end_turn','stop_sequence':None},
                                       'usage': {'output_tokens':0}}),
                    ('message_stop', {}),
                ]
                for name, event in events:
                    self.wfile.write(f'event: {name}\ndata: {json.dumps({"type":name, **event})}\n\n'.encode())
                self.wfile.flush()
                self.close_connection = True
            except OSError:
                pass  # capture is complete even if Claude cancels the response
    return ThreadingHTTPServer((host, port), Handler)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=11434)
    a = p.parse_args()
    server = make_server(a.directory, a.host, a.port)
    print(f'capture-only ready: {a.host}:{a.port}; no inference; no headers saved', flush=True)
    server.serve_forever()
