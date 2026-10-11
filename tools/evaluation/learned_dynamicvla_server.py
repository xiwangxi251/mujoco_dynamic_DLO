"""Run the unchanged DVLA bridge with isolated actuator correction and no videos."""
import argparse
from functools import partial
import json
from pathlib import Path
import sys
from unittest.mock import patch
from learned_tracking_common import LearnedTrackingEnv,dump
from panda_cable_grasp.cli import run_dynamicvla as runner
from panda_cable_grasp.evaluation import recording

AUDIT_ROOT=None

class NullWriter:
    def isOpened(self):return True
    def release(self):pass

class StateOnlyRecorder(recording.EpisodeRecorder):
    def __init__(self,*args,**kwargs):
        with patch.object(recording.cv2,'VideoWriter',side_effect=lambda *a,**k:NullWriter()):
            super().__init__(*args,**kwargs)
    def _write_frame(self,state_index):
        # Policy images are still rendered by the original bridge, on its original schedule.
        pass
    def finish(self,metadata):
        metadata.update(self.base_env.audit())
        metadata['method']='dynamicvla_opt5820'
        metadata['angular_feedback_x4_applied']=False
        metadata['video_recording_enabled']=False
        artifacts=super().finish(metadata)
        document=json.loads(artifacts.metadata.read_text())
        document['files']['global_video']=None;document['files']['wrist_video']=None
        document['video_recording_enabled']=False
        dump(artifacts.metadata,document)
        with (AUDIT_ROOT/'episodes_audit.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(metadata),default=str)+'\n')
        rows=[json.loads(l) for l in (AUDIT_ROOT/'episodes_audit.jsonl').read_text().splitlines()]
        dump(AUDIT_ROOT/'progress.json',{'completed':len(rows),'successes':sum(r['task_success'] for r in rows),'last_seed':rows[-1]['requested_seed']})
        class NoVideoArtifacts:
            episode_dir=artifacts.episode_dir
            def relative_to(self,root):
                values=artifacts.relative_to(root);values['global_video']='';values['wrist_video']='';return values
        return NoVideoArtifacts()

def main():
    global AUDIT_ROOT
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--variant',choices=('baseline','tracking_fix'),required=True)
    parser.add_argument('--audit-root',type=Path,required=True)
    custom,rest=parser.parse_known_args()
    AUDIT_ROOT=custom.audit_root;AUDIT_ROOT.mkdir(parents=True,exist_ok=True)
    assert not (AUDIT_ROOT/'episodes_audit.jsonl').exists(),'Use a fresh case directory'
    runner.CableGraspEnv=partial(LearnedTrackingEnv,variant=custom.variant,output=AUDIT_ROOT)
    runner.EpisodeRecorder=StateOnlyRecorder
    sys.argv=[sys.argv[0],*rest]
    parsed=runner.parse_args()
    runner.run_server(parsed)
    manifest_path=parsed.video_dir/parsed.run_name/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest['method']='dynamicvla_opt5820';manifest['intervention_variant']=custom.variant
    manifest['recording']['video_enabled']=False;manifest['recording']['state_recording_enabled']=True
    manifest['recording']['cameras']=[];manifest['artifacts']['global_videos']=[];manifest['artifacts']['wrist_videos']=[]
    manifest['angular_feedback_x4_applied']=False
    dump(manifest_path,manifest)

if __name__=='__main__':main()
