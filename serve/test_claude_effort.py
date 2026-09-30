"""Claude Code effort is a per-request API setting, not a launch flag."""
import unittest
from pathlib import Path
from serve.frontend import ChatTemplate,anthropic_to_messages
from serve.server import Service,MockEngine,ByteTokenizer
class ClaudeEffort(unittest.TestCase):
 def test_adaptive_effort_and_disabled_priority(self):
  for value,effective in [('low','low'),('medium','medium'),('high','xhigh'),('xhigh','xhigh'),('max','xhigh')]:
   req={'thinking':{'type':'adaptive'},'output_config':{'effort':value},'messages':[{'role':'user','content':'hi'}]}
   self.assertEqual(anthropic_to_messages(req)[2],{'reasoning_effort':effective})
   req['thinking']={'type':'disabled'}
   self.assertEqual(anthropic_to_messages(req)[2],{'enable_thinking':False})
 def test_explicit_effort_wins_over_shared_defaults(self):
  t=ByteTokenizer();s=Service(MockEngine(t,'4'),t,ChatTemplate(Path(__file__).parent/'chat_template.jinja'))
  s.set_shared({'reasoning_effort':'low'})
  req={'thinking':{'type':'adaptive'},'output_config':{'effort':'max'}}
  self.assertEqual(anthropic_to_messages(s.with_shared(req,'anthropic'))[2],{'reasoning_effort':'xhigh'})
 def test_rendered_prefix_varies_with_effort_not_user_message(self):
  tpl=ChatTemplate(Path(__file__).parent/'chat_template.jinja')
  roots={}
  for effort in ['low','medium','high','xhigh','max']:
   for content in ['TEST ONE','TEST TWO']:
    req={'system':'Stable system','tools':[{'name':'test','input_schema':{'type':'object'}}],
         'thinking':{'type':'adaptive'},'output_config':{'effort':effort},'messages':[{'role':'user','content':content}]}
    m,t,k=anthropic_to_messages(req);text=tpl.render(m,t,**k)
    root=text.split('<|im_start|>user')[0]
    if effort in roots:self.assertEqual(root,roots[effort])
    roots[effort]=root
  self.assertEqual(roots['high'],roots['max']);self.assertEqual(roots['xhigh'],roots['max'])
  self.assertNotEqual(roots['low'],roots['medium']);self.assertNotEqual(roots['medium'],roots['high'])
if __name__=='__main__':unittest.main()
