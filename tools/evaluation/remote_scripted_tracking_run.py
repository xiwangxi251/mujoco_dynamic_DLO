"""Launch/status/export the isolated frozen-source diagnostic on project 151."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'outputs/benchmarks/nero_scripted_tracking_fix_20261001'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--export',action='store_true')
    parser.add_argument('--workers',type=int,default=24)
    args=parser.parse_args()
    if args.launch:
        source=json.loads((ROOT/'bundle_manifest.json').read_text())
        for path,expected in source['sha256'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
        assert not (ROOT/'remote_launch.json').exists(),'already launched in this isolated checkout'
        import mujoco,numpy,scipy
        env=os.environ.copy()
        env.update(PYTHONPATH='src',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        pilotlog=(ROOT/'remote_pilot.log').open('wb')
        subprocess.run([sys.executable,'tools/evaluation/compare_scripted_tracking_fix.py','--stage','pilot','--workers',str(min(args.workers,18))],cwd=ROOT,env=env,stdout=pilotlog,stderr=subprocess.STDOUT,check=True)
        pilotlog.close()
        verifylog=(ROOT/'remote_verify.log').open('wb')
        subprocess.run([sys.executable,'tools/evaluation/verify_scripted_tracking_fix.py'],cwd=ROOT,env=env,stdout=verifylog,stderr=subprocess.STDOUT,check=True)
        verifylog.close()
        log=(ROOT/'remote_main.log').open('wb')
        process=subprocess.Popen([sys.executable,'tools/evaluation/compare_scripted_tracking_fix.py','--stage','main','--episodes','50','--workers',str(args.workers)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        metadata={'pid':process.pid,'root':str(ROOT),'created_utc':datetime.now(timezone.utc).isoformat(),
            'hostname':platform.node(),'platform':platform.platform(),'python':platform.python_version(),
            'mujoco':mujoco.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__,
            'workers':args.workers,'cpu_count':os.cpu_count(),'load_at_launch':os.getloadavg(),
            'all_source_hashes_verified':True,'episode_count':900}
        (ROOT/'remote_launch.json').write_text(json.dumps(metadata,indent=2))
        print(json.dumps(metadata,indent=2))
    if args.status:
        launch=ROOT/'remote_launch.json'
        if launch.exists():
            metadata=json.loads(launch.read_text())
            try:
                os.kill(metadata['pid'],0); metadata['process_exists']=True
            except ProcessLookupError:
                metadata['process_exists']=False
            metadata['complete']=(OUTPUT/'main/summary.json').exists()
            p=OUTPUT/'main/progress.json'
            if p.exists(): metadata['progress']=json.loads(p.read_text())
            print(json.dumps(metadata,indent=2))
        else:
            print('pilot/verification running before main launch')
    if args.export:
        assert (OUTPUT/'main/summary.json').exists(),'main not complete'
        for name in ('remote_launch.json','bundle_manifest.json','remote_main.log','remote_pilot.log','remote_verify.log'):
            (OUTPUT/name).write_bytes((ROOT/name).read_bytes())
        (OUTPUT/'main/source_snapshot.py').write_bytes((ROOT/'tools/evaluation/compare_scripted_tracking_fix.py').read_bytes())
        path=ROOT/'completed_results.tar.gz'
        with tarfile.open(path,'w:gz') as archive:
            archive.add(OUTPUT,arcname=OUTPUT.name)
        print(str(path))


if __name__=='__main__':
    main()
