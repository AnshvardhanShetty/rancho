"""Use authenticated offered-suffix GBTs without executing Python pickle objects.

Assumptions: a supplied, checked saved fit, four original features, one numeric
user vocabulary and three continuous inputs, identity target transform. Parse
pickle opcodes as data ONLY; never invoke pickle.load(s), REDUCE or GLOBAL.
Extract the native LightGBM text and f64 vocabulary, then load native trees.
No fit or score is selected from serving results. Original training geometry is
retained, whereas serving page geometry/workload are scaled and synthetic.
"""
from pathlib import Path
import hashlib,json,pickletools
import numpy as np
from predictor_bridge import encode

FEATURES=('user','causal_request_position','session_age_seconds','previous_session_gap_seconds')
CUT=259200000000

class SuffixGBT:
 def __init__(self,spec):
  from lightgbm import Booster
  p=Path(spec['directory']);raw=(p/'COMPLETE.json').read_bytes()
  assert hashlib.sha256(raw).hexdigest()==spec['receipt_sha256']
  r=json.loads(raw);self.cut_us=r['cut_us'];self.features=FEATURES
  assert self.cut_us==spec['cut_us']==CUT and tuple(r['expanded_recipe']['features'])==FEATURES
  assert r['target']=='offered_suffix' and r['target_transform']=='none'
  assert r['latest_training_request_us']<CUT and r['label_available_strictly_before_us']==CUT
  assert r['direct_prefix_and_inner_label_audit'] and r['future_buckets_excluded_from_training']
  raw=(p/'model.pkl').read_bytes();self.model_sha256=hashlib.sha256(raw).hexdigest()
  assert self.model_sha256==r['model_sha256']
  ops=list(pickletools.genops(raw))
  trees=[arg for op,arg,pos in ops if isinstance(arg,str) and arg.startswith('tree\nversion=')]
  assert len(trees)==1
  start=next(i for i,(op,arg,pos) in enumerate(ops) if arg=='vocab')
  stop=next(i for i in range(start+1,len(ops)) if ops[i][1]=='transform')
  segment=ops[start:stop]
  arrays=[bytes(arg) for op,arg,pos in segment if op.name in ('BYTEARRAY8','BINBYTES','SHORT_BINBYTES')]
  assert len(arrays)==1 and sum(op.name=='NONE' for op,arg,pos in segment)==6
  strings=[a for op,a,pos in segment if isinstance(a,str)]
  assert 'f8' in strings and '<' in strings and 'C' in strings
  values=np.frombuffer(arrays[0],dtype='<f8').copy()
  assert len(values)>0 and np.isfinite(values).all() and (values[1:]>values[:-1]).all()
  self.vocab=[values,None,None,None]
  self.booster=Booster(model_str=trees[0]);assert self.booster.num_trees()==r['tree_count']
  self.native_model_sha256=hashlib.sha256(trees[0].encode()).hexdigest()
  self.receipt=r
 def predict(self,rows,now_us):
  assert now_us>=self.cut_us and all(set(r)==set(FEATURES) for r in rows)
  x=np.array([[r[n] for n in FEATURES] for r in rows],dtype=np.float64)
  out=np.maximum(np.asarray(self.booster.predict(encode(x,self.vocab),num_threads=1)),0)
  assert out.shape==(len(rows),) and np.isfinite(out).all()
  return out
