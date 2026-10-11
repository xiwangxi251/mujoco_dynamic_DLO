"""Fresh positive/negative Y pairs under exactly the enhanced full-6D controller."""
import argparse
from pathlib import Path
from scan_full6_enhanced_workspace import EnhancedRig, OUT, native
from record_retest_136_videos import capture

def factory(directory):
    rig=EnhancedRig()
    return rig,rig.env,native

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--side',choices=('positive','negative'),required=True);args=p.parse_args()
    side=1 if args.side=='positive' else -1
    for x,y,name in ((.67,.28,'limit_pair'),(.55,.28,'endpoint_pair')):
        capture(6,x,side*y,name+'_'+args.side,rig_factory=factory,
            output_root=OUT/'asymmetry/videos',scan_directory=OUT,
            label='6+重力前馈+增强伺服+增益10')
