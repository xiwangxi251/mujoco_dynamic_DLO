# AGENTS.md — project conventions (mujoco_dynamic_DLO)

## Server / environment

- Repo on server: `/data1/hxai/mujoco/mujoco_dynamic_DLO` (local mirror:
  `C:\Users\27642\Desktop\mujoco\mujoco_dynamic_DLO`, ssh `-p 60022 hxai@10.108.17.151`)
- Python envs: `/data1/hxai/miniconda3/envs/dynamicvla` (main),
  `/data1/hxai/miniconda3/envs/mp2cdlo` (MP2CDLO baseline)
- Run python with `PYTHONPATH=src` from repo root.
- Packed dataset: `/data1/hxai/mujoco/perception_runs/dataset_v1_packed`
  (mmap `.npy` parts; ~780k frames after extra-episode merge).
- Test seeds (held-out): `/data1/hxai/mujoco/perception_runs/test_seeds_<scenario>.txt`
- Model runs: `/data1/hxai/mujoco/perception_runs/runs/<variant>/`
  (`checkpoint_best.pt`, `train_log.jsonl`)
- Launch long jobs with `ulimit -n 4096` (mmap parts × workers hits the
  default open-file limit otherwise).

## Evaluation — PRIMARY CRITERION (user directive)

- **Sequential episode-level self-feedback eval is the primary metric**,
  not per-frame independent eval.
- Use `tools/perception/dump_occdyn_ep.py --seed <held-out test seed>`:
  replays a full episode in order, feeds the model's OWN previous output
  back as `prev_pos`. This is the deployment-equivalent metric.
- `evaluate_estimator.py` (packed, shuffled frames) is a diagnostic only;
  for `use_prev_pos` models it is teacher-forced (prev = GT t-1) and
  overestimates deployable accuracy by a large margin.
- Report both numbers clearly labelled. Strict ordered MPNE is the
  comparable metric; the renderer `render_tracked_npz.py` shows
  flip-tolerant `min(fwd, rev)` err on screen — note the difference.
- Held-out viz seeds per scenario: static 20281878, rigid 20282002,
  shape 20283215, combined 20283026.

## Data conventions

- Packed rows: `points` (2×384 world-frame cloud), `pos14`, `vel14`,
  `fut14`, `vis14`, `masks` (2×90×120), `hand`, `hand_fut`, `lead`,
  `ep_id`, `frame_i`, `seeds`.
- Seed values collide ACROSS scenarios — always pass `scenarios=` to
  `PackedDLODataset` when filtering by seed.
- Normalisation: `(x - center)/0.5` where `center` is stored in each
  checkpoint.

## Scope notes

- Perception vs closed-loop policy are separate tasks; do not run the
  grasp policy unless explicitly asked.
- Real robot cannot propagate simulator physics — deployed methods must
  infer state from observations + robot pose + temporal feedback only.
- EPN3D implementation is an "EPN3D-style proxy", not exact reproduction.
