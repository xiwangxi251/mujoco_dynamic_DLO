"""Official frozen DVLA inference, with the same RNG seed per paired episode."""
import argparse
import importlib.util
import logging
import os
from pathlib import Path
import random
import sys
import numpy as np
import torch

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inference-source',type=Path,required=True)
    parser.add_argument('--weights',required=True);parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--img-port',type=int,required=True);parser.add_argument('--act-port',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='[%(levelname)s] %(asctime)s %(message)s')
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    sys.path.insert(0,'/data1/hxai/mujoco/DynamicVLA')
    spec=importlib.util.spec_from_file_location('official_inference',args.inference_source)
    client=importlib.util.module_from_spec(spec);sys.modules[spec.name]=client;spec.loader.exec_module(client)
    original=client.get_vla_model
    args.output.mkdir(parents=True,exist_ok=True)
    def loader(*a,**kw):
        model,cfg=original(*a,**kw)
        original_reset=model.reset;counter=[0]
        def reset():
            seed=args.seed+counter[0];counter[0]+=1
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            with (args.output/'episode_rng_seeds.jsonl').open('a') as f:f.write(str(seed)+'\n')
            return original_reset()
        model.reset=reset
        return model,cfg
    client.get_vla_model=loader
    client.args=argparse.Namespace(host='127.0.0.1',img_port=args.img_port,act_port=args.act_port)
    # All action/history/chunk behavior stays in the official inference module.
    client.main(args.weights,'opt5820_tracking_pair',1,'euler',True,False,
        '127.0.0.1',args.img_port,args.act_port,str(args.output),None,True,8)

if __name__=='__main__':main()
