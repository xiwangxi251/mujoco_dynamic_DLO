"""Record rounds 16..20 ahead of the ascending batch, using four workers.

The ascending batch checks completed provenance before starting a job, so
these pre-completed high-numbered rounds are reused. Do not run gallery
writes here: the primary driver collects every completed episode.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import json
import record_cable_x075_20episodes as batch

if __name__=='__main__':
    jobs=[(s,i,batch.SEEDS[i-1]) for i in range(20,15,-1) for s in batch.SCENES]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        pending=[pool.submit(batch.record_job,j) for j in jobs]
        for f in as_completed(pending):
            print(json.dumps(dict(tail_completed=True,last=f.result())),flush=True)
    print(json.dumps(dict(tail_done=30)),flush=True)
