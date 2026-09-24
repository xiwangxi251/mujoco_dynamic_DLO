"""Post-hoc orientation stabilisation for dumped perception episodes.

Failure mode: the estimator occasionally outputs the chain with node0/node13
endpoint identity swapped (geometry correct, direction flipped). The physical
end identity cannot change within an episode, so we track it.

Anchors available per frame:
  * continuity: tracked node0 position must move smoothly
  * hand proximity: in ambiguous windows the GT node0 is usually the end
    nearer the gripper (weak cue alone, strong inside flip windows)

Rule: never flip on tracking alone. Per frame compute, for both candidate
orientations, (a) distance of candidate node0 to predicted node0 position
(constant-velocity extrapolation), (b) candidate node0 distance to hand.
A flip commits only when the reversed orientation wins BOTH cues clearly
for K consecutive frames.

Usage:
  python fix_orientation.py <dump.npz> [--hand-from-packed SCENARIO SEED]
"""
from __future__ import annotations

import argparse
import glob
import sys

import numpy as np


def load_hand(packed_dir: str, scenario: str, seed: int) -> np.ndarray:
    rows = []
    for f in sorted(glob.glob(f"{packed_dir}/{scenario}__part*.npz")):
        z = np.load(f)
        m = z["seeds"] == seed
        if m.any():
            idx = np.where(m)[0]
            order = idx[np.argsort(z["frame_i"][idx])]
            rows.append(z["hand"][order, :3])
    hand = np.concatenate(rows, 0)
    return hand


def stabilise(pred: np.ndarray, hand: np.ndarray | None, K: int = 4):
    """Return corrected preds (T,M,3) and flip state per frame."""
    T = pred.shape[0]
    flip = False
    E = pred[0, 0].copy()          # tracked node0 position
    V = np.zeros(3)                # its velocity estimate
    votes = 0                      # consecutive frames favouring a flip
    out = np.empty_like(pred)
    flips = np.zeros(T, bool)
    for i in range(T):
        Ep = E + V
        cand = [(pred[i], False), (pred[i, ::-1], True)]
        # score each candidate: continuity dist of node0 (+ hand dist if given)
        scores = []
        for chain, _ in cand:
            s = np.linalg.norm(chain[0] - Ep)
            if hand is not None:
                s += 0.5 * np.linalg.norm(chain[0] - hand[i])
            scores.append(s)
        prefer_flip = scores[1] < 0.8 * scores[0]
        votes = votes + 1 if prefer_flip else 0
        if votes >= K:
            flip = not flip
            votes = 0
        chain = pred[i, ::-1] if flip else pred[i]
        out[i] = chain
        flips[i] = flip
        # update tracker only with committed chain, and only when the
        # committed node0 is plausible (don't let a bad frame teleport E)
        d = np.linalg.norm(chain[0] - Ep)
        if d < 0.08:               # metres; node moves ~12mm/frame
            V = 0.7 * V + 0.3 * (chain[0] - E)
            E = chain[0]
        else:
            E = Ep                 # coast on velocity
    return out, flips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--packed-dir", default="/data1/hxai/mujoco/perception_runs/dataset_v1_packed")
    ap.add_argument("--scenario")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    gt, pred = d["gt"], d["pred"]
    hand = None
    if args.scenario and args.seed:
        hand = load_hand(args.packed_dir, args.scenario, args.seed)[: len(pred)]
        print("hand loaded:", hand.shape)

    e = np.linalg.norm(pred - gt, axis=2).mean(1) * 1000
    er = np.linalg.norm(pred[:, ::-1] - gt, axis=2).mean(1) * 1000
    print(f"raw      : mean={e.mean():6.1f} med={np.median(e):6.1f} "
          f"flipped_frames={(er < e).sum()}/{len(e)}")

    for use_hand in ([False, True] if hand is not None else [False]):
        fixed, flips = stabilise(pred, hand if use_hand else None)
        ef = np.linalg.norm(fixed - gt, axis=2).mean(1) * 1000
        tag = "cont+hand" if use_hand else "cont-only"
        print(f"{tag}: mean={ef.mean():6.1f} med={np.median(ef):6.1f} "
              f"flip_state_frames={flips.sum()}")

    if args.out:
        fixed, flips = stabilise(pred, hand)
        np.savez(args.out, cloud=d["cloud"], gt=gt, pred=fixed,
                 ok=d["ok"] if "ok" in d else np.ones(len(gt), bool))
        print("saved", args.out)


if __name__ == "__main__":
    main()
