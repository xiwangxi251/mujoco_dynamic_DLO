"""Benchmark image-dataset and model step time."""
import argparse, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
from panda_cable_grasp.perception.dataset import PackedDLOImageDataset
from panda_cable_grasp.perception.model_img import UNetSkeleton


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed-dir", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    ds = PackedDLOImageDataset(args.packed_dir)
    print(f"ds n={len(ds)}", flush=True)

    # 1) single __getitem__
    t0 = time.time()
    for i in range(32):
        _ = ds[i]
    t_item = (time.time() - t0) / 32
    print(f"getitem: {t_item*1000:.1f} ms/item -> "
          f"est batch{args.batch} single-worker {t_item*args.batch:.2f}s", flush=True)

    # 2) dataloader throughput
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True,
                    persistent_workers=True, drop_last=True)
    t0 = time.time()
    n_batch = 10
    for i, b in enumerate(dl):
        if i >= n_batch:
            break
    t_dl = (time.time() - t0) / n_batch
    print(f"dataloader: {t_dl:.3f} s/batch (w={args.workers})", flush=True)

    # 3) model step on GPU
    dev = torch.device(args.device)
    model = UNetSkeleton().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    it = iter(dl)
    batch = next(it)
    x = batch["img"].to(dev, non_blocking=True)
    tgt = batch["tgt"].to(dev, non_blocking=True)
    print(f"img shape {tuple(x.shape)} tgt {tuple(tgt.shape)}", flush=True)

    from torch.nn import functional as F
    def loss_fn(out, tgt):
        bce = F.binary_cross_entropy_with_logits(out[:, 0], tgt[:, 0])
        return bce

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        out = model(x)
        loss = loss_fn(out, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    t_step = (time.time() - t0) / 10
    print(f"train step: {t_step*1000:.0f} ms/batch", flush=True)
    print(f"epoch est: {(t_dl+t_step)*len(ds)//args.batch/60:.1f} min "
          f"(steps={len(ds)//args.batch})", flush=True)


if __name__ == "__main__":
    main()
