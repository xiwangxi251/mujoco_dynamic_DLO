"""Finish rounds 2..20 after the first six reference recordings."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import json
import time
import record_cable_x075_20episodes as batch

if __name__ == '__main__':
    jobs = [(s,i,seed) for i,seed in enumerate(batch.SEEDS,1) if i>1 for s in batch.SCENES]
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(batch.record_job,j) for j in jobs]
        for f in as_completed(futures):
            last=f.result();summary=batch.gallery()
            print(json.dumps(dict(completed=summary['completed_episodes'],of=120,wall_s=round(time.monotonic()-start,1),last=last)),flush=True)
    batch.final_verify();summary=batch.gallery()
    print(json.dumps(dict(done=summary['completed_episodes'],clips=summary['completed_clips'])),flush=True)
