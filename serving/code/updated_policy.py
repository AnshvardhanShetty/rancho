"""Live port of the frozen filtered-linear, cost-band, half-behind RANCHO.

Assumptions: cold controller, 1000-offer updates, last 20000 observations, cost
bands floor(log2(write cost)), pooled fallback below100 band observations.
Bonus =1+sum(strictly earlier eligible HBM-shadow misses)/distinct suffix pages.
Hypothetical HBM uses whole-request LRU and actual usable HBM capacity; counters
start empty and advance in actual completed-prefill order. This serialization is
causal but differs from arrival-order serial offline replays under concurrency.
The external physical write guard is deliberately stricter than the archived
four-day wallet. No forced admissions, oracle reads, calibration history or
later score arrays. Reuse counters never include the current request's reads.
"""
from collections import OrderedDict
import math,time
from controllers import Wallet,Observation,hash01,DAY
from updated_model import SuffixGBT,CUT

class BandWallet(Wallet):
 def recompute(self,now):
  super().recompute(now)
  values=[v for v in self.ring if v.cost>0 and math.isfinite(v.score)]
  if len(values)<100:return
  span=max(1.,(max(v.time_us for v in self.ring)-min(v.time_us for v in self.ring))/1e6)
  demand=len(values)/span*max(1.,(self.end_us-now)/1e6)*(sum(v.cost for v in values)/len(values))
  fraction=min(1.,max(0.,(self.cap-self.W)/max(1e-9,demand)))
  groups={};self.bandcuts={}
  for v in values:groups.setdefault(v.cost.bit_length()-1,[]).append(v)
  for key,group in groups.items():
   if len(group)<100:continue
   if fraction in (0.,1.):self.bandcuts[key]=(math.inf,0.) if fraction==0 else (-math.inf,1.);continue
   weights={}
   for v in group:weights[v.score]=weights.get(v.score,0)+v.cost
   wanted=fraction*sum(weights.values());above=0
   for score in sorted(weights,reverse=True):
    equal=weights[score]
    if above+equal>=wanted:self.bandcuts[key]=(score,max(0.,min(1.,(wanted-above)/equal)));break
    above+=equal
 def updated_decide(self,event,now,prediction,distinct,cost,past,split,payment_limit=None):
  assert self.last_id<event and self.last_us<=now<self.end_us and past>=0 and distinct>=0 and cost>=0
  assert distinct or not cost
  self.last_id,self.last_us=event,now;self.phase(now)
  multiplier=1+past/distinct if distinct else 1.
  score=prediction*float(distinct)/float(cost)*multiplier if cost else math.inf
  cutoff,coin=getattr(self,'bandcuts',{}).get(cost.bit_length()-1,(self.threshold,self.tie)) if cost and self.have else (self.threshold,self.tie)
  behind=bool(cost and self.have and self.W*(self.end_us-split)<self.budget*(now-split))
  if behind:cutoff*=.5
  admit=bool(distinct and (not cost or cost<=self.budget-self.W and cost<=self.cap-self.W and (not self.have or score>cutoff or score==cutoff and hash01(event,self.seed^0x5BF03635)<coin)))
  before_guard=admit
  if payment_limit is not None:
   assert type(payment_limit) is int and payment_limit>=0
   admit=admit and cost<=payment_limit
  details=dict(score=score if math.isfinite(score) else str(score),cutoff=cutoff if math.isfinite(cutoff) else str(cutoff),tie=float(coin),have_threshold=bool(self.have),half_behind=behind,history_multiplier=multiplier,past_filtered_reuses=past,score_admit_before_physical_guard=before_guard,physical_guard_rejected=before_guard and not admit)
  self._observe(Observation(event,score,cost,now,distinct))
  if admit:self.W+=cost
  return admit,details

class FilteredHistory:
 def __init__(self,capacity):
  self.capacity=capacity;self.cache=OrderedDict();self.stamps={};self.counts={};self.processed=set()
 def prepare(self,request_id,keys,session,suffix_begin):
  assert request_id not in self.processed and 0<=suffix_begin<=len(keys)
  readable=0
  for key in keys:
   if key not in self.cache:break
   self.cache.move_to_end(key);readable+=1
  suffix=list(dict.fromkeys(keys[suffix_begin:]));past=sum(self.counts.get(k,0) for k in suffix)
  pending=[];seen=set()
  for j,key in enumerate(keys):
   if key in seen:continue
   seen.add(key)
   if j>=readable and self.stamps.get(key)==session:pending.append(key)
  return dict(request_id=request_id,keys=keys,session=session,suffix_begin=suffix_begin,shadow_cached_pages=readable,past_filtered_reuses=past,pending=pending)
 def commit(self,t):
  for key in t['keys'][t['shadow_cached_pages']:]:
   if self.capacity:
    self.cache[key]=True;self.cache.move_to_end(key)
    if len(self.cache)>self.capacity:self.cache.popitem(last=False)
  for key in t['pending']:self.counts[key]=self.counts.get(key,0)+1
  for key in t['keys']:self.stamps[key]=t['session']
  self.processed.add(t['request_id'])

class UpdatedAdmission:
 def __init__(self,config):
  self.config=config
  from pathlib import Path
  import hashlib,json
  raw=Path(config['controller_history']).read_bytes();assert hashlib.sha256(raw).hexdigest()==config['controller_history_sha256']
  receipt=json.loads(raw);assert receipt['cold'] and receipt['observations']==receipt['paid_pages']==0 and receipt['fit']==config['fit']
  self.model=SuffixGBT(config['fit']);assert self.model.model_sha256==receipt['model_sha256']
  self.wallet=BandWallet('hbf',config['HBF_pages'],config['repetition'],end_us=CUT+DAY,release_us=CUT+DAY,reserve=0.)
  self.completed=set();self.past=0
 def set_past(self,past):
  assert type(past) is int and past>=0
  self.past=past
 def hbf_prefill_complete(self,row,now,features,distinct,cost,*,computed_prompt_tokens,prompt_tokens,gpu_completed,payment_limit=None):
  assert gpu_completed is True and computed_prompt_tokens==prompt_tokens and prompt_tokens>0 and row not in self.completed
  start=time.perf_counter_ns();prediction=float(self.model.predict([features],now)[0]);predictor_ns=time.perf_counter_ns()-start
  start=time.perf_counter_ns();admit,detail=self.wallet.updated_decide(row,now,prediction,distinct,cost,self.past,CUT,payment_limit)
  controller_ns=time.perf_counter_ns()-start
  self.completed.add(row)
  return dict(admit=admit,prediction=prediction,paid_cost=cost if admit else 0,total_paid_writes=self.wallet.W,fit_cut_us=CUT,model_sha256=self.model.model_sha256,predictor_cpu_ns=predictor_ns,controller_cpu_ns=controller_ns,**detail)
