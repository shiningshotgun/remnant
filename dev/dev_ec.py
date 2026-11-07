"""EXPLORATORY dev check of episodic_runner on dev seeds (9001-9005). Not confirmatory."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remnant_core.episodic_runner import EpisodicRunner
arm=sys.argv[1]; seeds=[int(x) for x in sys.argv[2].split(',')]
for s in seeds:
    t=time.time(); r=EpisodicRunner(s,arm)
    pre=r.evaluate(5,s*7+1); log=r.run(100); post=r.evaluate(5,s*7+2)
    esc=[x for x in log if x['escaped']]
    pol_esc=[x['tide'] for x in esc if x['kind']=='policy']
    print(arm,dict(seed=s,pre=pre['escapes'],post=post['escapes'],first=esc[0]['tide'] if esc else None,
      first_kind=esc[0]['kind'] if esc else None,n_esc=len(esc),policy_esc=len(pol_esc),
      policy_tides_after_first=sum(1 for x in log if esc and x['tide']>esc[0]['tide'] and x['kind']=='policy'),
      edges=log[-1]['map_edges'],post_strides=[t2['strides'] for t2 in post['tides']],sec=round(time.time()-t,1)),flush=True)
