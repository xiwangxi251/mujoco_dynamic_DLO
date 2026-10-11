"""Gate invariants: weak scenes, fresh windows, replay and failure reset."""
from rollout_gate import Windows

good=dict(success=1,loaded=1,secured=1,attempt=1)
bad=dict(success=0,loaded=1,secured=1,attempt=1)
def fill(w,n,k,row):
    for _ in range(k):w.add(n,row)

w=Windows(('strong','weak'),())
fill(w,'strong',100,good);fill(w,'weak',100,bad)
assert not w.decide()['passed'] and w.decide() is None
fill(w,'strong',100,good);fill(w,'weak',99,good)
assert w.decide() is None
fill(w,'weak',1,good)
assert w.decide()['consecutive']==1 and w.decide() is None
w=Windows(('strong','weak'),(),w.state())
fill(w,'strong',100,good);fill(w,'weak',100,good)
assert w.decide()['promote']
w=Windows(('target',),('old_a','old_b'))
fill(w,'target',100,good);fill(w,'old_a',50,good);fill(w,'old_b',50,bad)
assert not w.decide()['passed']
for n,k in [('target',100),('old_a',50),('old_b',50)]:fill(w,n,k,good)
assert w.decide()['consecutive']==1
fill(w,'target',100,bad);fill(w,'old_a',50,good);fill(w,'old_b',50,good)
assert w.decide()['consecutive']==0
w=Windows(('target',),())
fill(w,'target',100,dict(success=1,loaded=0,secured=1,attempt=1))
assert not w.decide()['passed']
fill(w,'target',100,dict(success=1,loaded=1,secured=0,attempt=1))
assert not w.decide()['passed']
print('Gate invariants passed: scene isolation, denominators, nonoverlap, resume state, replay, consecutive reset, loaded and secured gates')
