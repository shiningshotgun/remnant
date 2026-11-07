"""EXPLORATORY development sweep on dev seeds 9001-9010 only. Not confirmatory.
Seeds here are disjoint from any confirmatory seed. All output kept in dev/."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dev.optimistic_runner import OptimisticRunner
arm=sys.argv[1]; vopt=float(sys.argv[2]); seeds=[int(x) for x in sys.argv[3].split(',')]
td=None if sys.argv[4]=='none' else float(sys.argv[4])
trunc=sys.argv[5] if len(sys.argv)>5 else 'dawn'
mask=(sys.argv[6]=='mask') if len(sys.argv)>6 else False
rows=[]
for s in seeds:
    t=time.time()
    r=OptimisticRunner(s,arm,vopt,trace_decay=td,truncation=trunc,mask_noops=mask)
    pre=r.evaluate(3,s*7+1)
    log=r.run(100)
    post=r.evaluate(3,s*7+2)
    esc=[x['tide'] for x in log if x['escaped']]
    row=dict(seed=s,pre=pre['escapes'],post=post['escapes'],first=esc[0] if esc else None,
             n_esc=len(esc),last20=sum(x['escaped'] for x in log[-20:]),
             edges=log[-1]['map_edges'],pearl_last20=sum(x['pearls'] for x in log[-20:]),
             post_ret=[t2['ret'] for t2 in post['tides']],sec=round(time.time()-t,1))
    print(arm,vopt,td,trunc,mask,row,flush=True); rows.append(row)
