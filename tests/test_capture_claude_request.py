import importlib.util,json,os,tempfile,threading,unittest,urllib.request
from pathlib import Path
spec=importlib.util.spec_from_file_location('capture',Path(__file__).parents[1]/'tools/capture_claude_request.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Capture(unittest.TestCase):
 def test_capture_private_body_not_headers_and_both_response_modes(self):
  with tempfile.TemporaryDirectory() as d:
   server=m.make_server(Path(d),'127.0.0.1',0)
   t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
   try:
    for stream in (False,True):
     payload={'model':'diagnostic','system':[{'type':'text','text':'test system'}],'messages':[{'role':'user','content':'TEST CACHE'}],'max_tokens':10,'stream':stream}
     req=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/v1/messages?beta=true',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer DO_NOT_SAVE'})
     with urllib.request.urlopen(req) as r:body=r.read().decode()
     self.assertIn('diagnostica',body)
     if stream:self.assertIn('event: message_stop',body)
     else:self.assertEqual(json.loads(body)['stop_reason'],'end_turn')
    files=list(Path(d).glob('request-*.json'));self.assertEqual(len(files),2)
    for p in files:
     self.assertEqual(p.stat().st_mode & 0o777,0o600)
     self.assertNotIn('DO_NOT_SAVE',p.read_text())
     self.assertEqual(json.loads(p.read_text())['system'][0]['text'],'test system')
   finally:server.shutdown();server.server_close();t.join()
if __name__=='__main__':unittest.main()
