"""Construct a fixed-cache-benefit serving workload without fixing admissions.

Assumptions: declared session/document groups; immutable full 16-token
hash-chain blocks. A fixed earlier synthetic reference trace contains the later
request contexts (ordinary repeated-document reuse); it seeds reference stamps
and causal request features but contributes NO KV to measured engines. GPU C7
calibration is separate and uses actually observed earlier prefill offers.
Each fresh engine executes one real fill request per group before the timed
window; actual native lookup counts determine hits. The target is about95%
OFFLINE eligible previous-reference-same-session, first-distinct block reuse,
not an assigned hit counter. Workload selection is permitted only on feasibility
and matched hit ratio, never throughput/TTFT. This is a constructed capacity
experiment and not a reproduction of a production trace distribution.
"""
import hashlib,struct
from features import RequestHistory
from synthetic_workload import FIT

def tokens(setup,i,calibration=False,warm=False):
 g=i%setup['groups'];p=setup['prefix_tokens'];u=setup['unique_suffix_tokens'];rep=setup.get('repetition',17)
 base=4000 if calibration else 1000
 prefix=[base+g*64+j%32 for j in range(p)]
 tail=(10000+i if calibration else 30000+g if warm else 20000+rep*1300+i)
 return prefix+[tail]*u

def prompts(setup,calibration=False):
 if calibration:return [tokens(setup,i,True) for i in range(setup['calibration_requests'])]
 return [tokens(setup,g,warm=True) for g in range(setup['groups'])]+[tokens(setup,i) for i in range(setup['requests'])]

def records(setup):
 h=RequestHistory();maxn=setup['max_reference_requests'];cal=setup['calibration_requests'];gap=setup['logical_gap_seconds']*1000000;cut=FIT['cut_us'];row=0
 for i in range(maxn):
  h.arrival(row,cut-(maxn+cal-i)*gap,i%setup['groups'],setup['synthetic_user']);row+=1
 earlier=[]
 for i in range(cal):
  t=cut-(cal-i)*gap;info=h.arrival(row,t,10000+i,setup['synthetic_user']);row+=1
  earlier.append(dict(sequence_row=row-1,session=10000+i,policy_us=t,features=info.shared()))
 measured=[]
 for i in range(setup['groups']+setup['requests']):
  g=i if i<setup['groups'] else (i-setup['groups'])%setup['groups'];t=cut+i*gap
  info=h.arrival(row,t,g,setup['synthetic_user']);row+=1
  measured.append(dict(sequence_row=row-1,session=g,policy_us=t,features=info.shared(),warmup=i<setup['groups']))
 return earlier,measured

def bound_requests(setup):return {f'row{i}':dict(features=v['features'],policy_us=v['policy_us'],session=v['session']) for i,v in enumerate(records(setup)[1])}

def block_keys(prompt):
 if len(prompt)%16:raise ValueError('Only full input blocks in this controlled study')
 previous=bytes(32);out=[]
 for a in range(0,len(prompt),16):
  previous=hashlib.sha256(previous+struct.pack('<16I',*prompt[a:a+16])).digest();out.append(previous)
 return out

def eligibility(setup):
 # Reconstruct independent offline-style reference stamps before any cache fill.
 stamps={}
 for i in range(setup['max_reference_requests']):
  for b in block_keys(tokens(setup,i)):stamps[b]=i%setup['groups']
 for i in range(setup['calibration_requests']):
  for b in block_keys(tokens(setup,i,calibration=True)):stamps[b]=10000+i
 out=[]
 for i,prompt in enumerate(prompts(setup)):
  group=i if i<setup['groups'] else (i-setup['groups'])%setup['groups'];keys=block_keys(prompt);seen=set();bits=[]
  for b in keys:
   bits.append(b not in seen and stamps.get(b)==group);seen.add(b);stamps[b]=group
  out.append(bits)
 return out
