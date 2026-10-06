#!/usr/bin/env python3
import json
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
from nvfp4_compare_logits import compare, compare_first, load_dump, positions_for

class CompareTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.x=np.array([[0,1,2,3],[2,1,4,0],[2,3,1,0]],dtype='<f4')
    def tearDown(self): self.tmp.cleanup()
    def dump(self,name,x):
        p=self.root/name;p.write_bytes(struct.pack('<ii',x.shape[1],x.shape[0])+x.tobytes());return p
    def test_identity(self):
        p=self.dump('a',self.x);r=compare(p,p,[0,1,2])
        self.assertEqual(r['top1_agreement'],1)
        self.assertEqual(r['max_kl_baseline_to_candidate_nats'],0)
        self.assertEqual(r['delta_mean_nll_nats'],0)
        json.dumps(r,allow_nan=False)
    def test_invariant_to_logit_shift(self):
        r=compare(self.dump('a',self.x),self.dump('b',self.x+16),[0,1,2])
        self.assertEqual(r['mean_kl_baseline_to_candidate_nats'],0)
        self.assertEqual(r['max_abs_logit_delta'],16)
    def test_changed(self):
        y=self.x.copy();y[0,0]=10
        r=compare(self.dump('a',self.x),self.dump('b',y),[0,1,2])
        self.assertAlmostEqual(r['top1_agreement'],2/3)
        self.assertGreater(r['mean_kl_baseline_to_candidate_nats'],0)
    def test_final_row_once(self):
        self.assertEqual(positions_for(5,2),[0,2,4])
        self.assertEqual(positions_for(6,2),[0,2,4,5])
    def test_bad_stride(self):
        with self.assertRaises(ValueError): positions_for(3,0)
    def test_wrong_rows(self):
        p=self.dump('a',self.x)
        with self.assertRaises(ValueError): compare(p,p,[0,1,2,3])
    def test_nonfinite(self):
        y=self.x.copy();y[1,0]=np.nan
        with self.assertRaises(ValueError): compare(self.dump('a',self.x),self.dump('b',y),[0,1,2])
    def test_bad_token(self):
        p=self.dump('a',self.x)
        with self.assertRaises(ValueError): compare(p,p,[0,4,2])
    def test_truncated(self):
        p=self.dump('a',self.x);p.write_bytes(p.read_bytes()[:-1])
        with self.assertRaises(ValueError): load_dump(p)
    def test_empty(self):
        p=self.root/'empty';p.write_bytes(b'')
        with self.assertRaises(ValueError): load_dump(p)
    def test_extra_bytes(self):
        p=self.dump('a',self.x);p.write_bytes(p.read_bytes()+b'bad')
        with self.assertRaises(ValueError): load_dump(p)

    def raw(self, name, x):
        p=self.root/name; p.write_bytes(np.asarray(x,dtype='<f4').tobytes()); return p
    def test_raw_identity(self):
        p=self.raw('a',self.x[0]); r=compare_first(p,p,[0,1,2])
        self.assertEqual(r['top1_agreement'],1)
        self.assertEqual(r['kl_baseline_to_candidate_nats'],0)
        self.assertEqual(r['position'],2)
        self.assertNotIn('delta_mean_nll_nats',r)
        json.dumps(r,allow_nan=False)
    def test_raw_change(self):
        r=compare_first(self.raw('a',[0,1,2,3]),self.raw('b',[9,1,2,3]),[0,1])
        self.assertEqual(r['top1_agreement'],0)
        self.assertGreater(r['kl_baseline_to_candidate_nats'],0)
    def test_raw_shape(self):
        with self.assertRaises(ValueError):
            compare_first(self.raw('a',[0,1,2]),self.raw('b',[0,1]),[0])
    def test_raw_empty(self):
        p=self.raw('a',[])
        with self.assertRaises(ValueError): compare_first(p,p,[0])
    def test_raw_nonfinite(self):
        with self.assertRaises(ValueError):
            compare_first(self.raw('a',[0,1,np.inf]),self.raw('b',[0,1,2]),[0])
    def test_raw_tokens(self):
        p=self.raw('a',[0,1,2])
        for tokens in ([],[3],[-1]):
            with self.assertRaises(ValueError): compare_first(p,p,tokens)

if __name__=='__main__': unittest.main()
