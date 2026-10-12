# Server 2: complete frozen-policy range test, 400 episodes

User confirmed 20 episodes per cell and pausing training to prioritize this test on 2026-10-12. All successes and failures receive native global and wrist camera videos, 1280x960 at 50fps. There are 7 scenes, 20 cells, 400 executions, and 800 videos. No rerun solely for video is allowed.

Training source remains the frozen server snapshot of 44af6e30f9ae95de99924776d0c3814174f8bdfa. The separately deployed recorder uses the original training factory and checks its source hashes, RL configuration and workspace map. The user-requested pause sends SIGSTOP only to the verified trainer/supervisor tree, retaining the current policy, value, optimizer, environment and unfinished rollout in memory. A validated disk checkpoint is copied separately. Do not resume training automatically.

- Training run: `/root/rivermind-data/runs/ppo_range_session_height_scratch_20m_20261012`.
- Test output: the run's `all_cells_video400_20261012` directory.
- In-memory pause: 5,898,240 samples; 43 verified processes paused.
- Frozen disk checkpoint: 5,800,000 samples, SHA256 `17d5078b136c3fcbb754c6c9ddffe9f2bb5fff349a119e3ea88b891db43d0ebd`.
- Initially eight single-threaded simulation/recording workers. Following the user's request to maximize throughput, recording expanded to 32 workers. The 8-worker run used 9.51 CPU cores out of a 40-core quota, with no CPU throttling. Each encoder and OSMesa renderer has a bounded thread count; 32 workers leave room within that quota for encoding.
- Deterministic PPO actions. Every cell uses the corresponding seeds 20260804 through 20260823.
- The initial-state/replay bank is the training bank, not a newly generated holdout bank. Reset seeds absent from the bank follow the original resampling fallback; actual bank source seeds are retained in each episode's `initial_info`. Repeated reset seeds across cells are corresponding executions, not new unique initial states. These results diagnose behavior and parameter sensitivity, not unseen-state generalization.
- Parameters are sampled continuously and uniformly within the selected cell at reset, then held fixed throughout the episode. Adaptive cell selection is disabled.
- Ordinary static and complex static: one cell each, 20 episodes each.
- Simple and complex-initial rigid motion: two cells each, 40 episodes each.
- Shape replay: two time-scale cells, 40 episodes.
- Shape: four drive-strength/frequency cells, 80 episodes.
- Combined: eight rigid-speed/drive-strength/frequency cells, 160 episodes.
- Bounds: rigid nominal speed [0.1944444444, 0.30] m/s; drive strength [1, 1.5]; shape frequency and replay time scale [7/9, 1]. Cells split each parameter range at its midpoint. These are command parameters, not measured velocity bins.
- Same reward, control, physics, public success criteria, workspace map, contact cache and penalty-ramp progress as the frozen training configuration.
- Each execution ends on its physical termination/success or the original 15-second timeout.

Preflight executes three real policy steps in each of all 20 cells through the same worker path. It verifies finite observations/actions, valid MP4 encoding, bitwise identical observations, rewards, qpos and qvel against a non-camera environment, and zero render-induced physical state changes. An independent intentional exit-7 test verifies that the supervisor reports failure. Smoke executions are separate from the formal 400 results.

Supervisor atomically records identity, worker-ready receipts, source heartbeat, real progress and failures. After launch, confirm survival after launcher exit and fresh worker-ready evidence. A separate collector reads existing results and downloads every episode; it never performs simulation. Present per-cell denominators, every episode's video and CSV/JSON exports. Twenty episodes per cell have substantial sampling uncertainty.

## Expansion handoff

The original 8-worker recorder generation is stopped by verified supervisor PID/birth/command and descendant identities. Seventeen complete episodes are validated and retained unchanged; 383 pending jobs are assigned disjointly to 32 workers. Incomplete recordings are archived separately and restarted, never included in results. Pending/completed IDs form exactly the original 400-job set. The checkpoint, reset seeds, parameter ranges, deterministic actions, video resolution/frame rate and training source stay fixed. A two-worker, six-real-step preflight and an intentional exit-7 supervisor test verify the revised dispatcher before cutover. The original all-20-cell preflight is retained.

The expansion receipt, previous generation, preserved IDs and parallel recorder SHA256 are stored with the test. New supervisor PID 52965 / birth 112797145; boot ID remains 9ce8123f-b48a-44d1-ad8d-60e76819fd5c. Training remains paused; no training process is included in the recorder tree cutover.
