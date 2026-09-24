"""Diagnose img-model dump: GT-vs-cloud offset, fwd/rev, strand collapse."""
import sys
import numpy as np
from scipy.spatial import cKDTree

npz = sys.argv[1]
d = np.load(npz, allow_pickle=True)
cloud, gt, pred = d["cloud"], d["gt"], d["pred"]

fwd_all, rev_all, gt2c, p2c, bias = [], [], [], [], []
for i in range(len(gt)):
    c = cloud[i]
    g, p = gt[i], pred[i]
    fwd = np.linalg.norm(p - g, axis=1).mean() * 1000
    rev = np.linalg.norm(p[::-1] - g, axis=1).mean() * 1000
    fwd_all.append(fwd); rev_all.append(rev)
    tree = cKDTree(c)
    gt2c.append(tree.query(g)[0].mean() * 1000)
    p2c.append(tree.query(p)[0].mean() * 1000)
    # signed-ish bias: mean(pred-gt) per axis
    bias.append((p - g).mean(0) * 1000)

fwd_all = np.asarray(fwd_all); rev_all = np.asarray(rev_all)
bias = np.asarray(bias)
print(f"fwd {fwd_all.mean():.1f}  rev {rev_all.mean():.1f}  "
      f"flip_frames {(rev_all < fwd_all).sum()}/{len(fwd_all)}")
print(f"GT->cloud {np.mean(gt2c):.1f} mm   pred->cloud {np.mean(p2c):.1f} mm")
print(f"mean pred-gt offset (mm): X {bias[:,0].mean():.1f} "
      f"Y {bias[:,1].mean():.1f}  Z {bias[:,2].mean():.1f}")
worst = np.argsort(fwd_all)[-5:]
print("worst fwd:", [(int(i), round(float(fwd_all[i]),1)) for i in worst])
