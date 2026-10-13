# Original shape drive: joint measured amplitude/speed response

The user rejected shape-reference tracking, pose-hold compensation and applying
the Shape replay airborne force gate to deformation. The earlier uncommitted
compiler prototype is preserved outside Git in
`outputs/range_motion_amplitude_speed_20261013/superseded_force_prototype`.
The tracked force compiler is restored; this change does not alter ordinary
shape dynamics. Shape force remains active during grasp and off the table.
Existing force-family controllers and archived scene identities are unchanged.

The range nominal rigid command upper limit changes from 0.30 to 0.25 m/s.
The lower limit remains 0.25 * 7/9. A new protocol version rejects previous
range checkpoint evidence. Ordinary physics-only full 15-second measurements
with the ready robot, true physics-step node velocities and the fixed
0.82--7.42 second window give 0.21293 m/s whole-object speed in both simple
and complex initial rigid scenes. These are no-policy engineering probes,
not PPO learning or grasp results. Combined motion needs its own joint map;
nominal 0.25 gave approximately 0.224--0.236 m/s rigid-component speed in two
probes, so the rigid-only conversion cannot be assumed to transfer exactly.

`--range-output-response PATH` opts into a joint inverse of physically measured
outputs. The profile uses version `measured_motion_response_v1`, scene tables
containing `axes`, `bounds`, `measured_outputs`, `driver_parameters` and
`driver_values`, and an independent physical `validation` record. Tables map
actual shape RMS amplitude in mm and residual node speed in m/s to the original
drive strength and time multiplier. Combined tables can additionally map actual
rigid speed to the nominal rigid command. Outputs are normalized before simplex
interpolation. No per-step feedback, force tape, pose hold or additional shape
contact gating is introduced. Targets remain continuous and sampled once per
episode. Internal drivers and response identity are saved in episode metadata.

The interface rejects unvalidated profiles, missing shape/combined coverage,
rank-deficient maps, requested rectangles outside the measured hull and target
extrapolation. Profile identity enters the curriculum/checkpoint version.
Independent physical validation is essential: inverse interpolation only
proposes inputs; it does not certify actual motion or resolve seed-dependent
nonmonotonic response. No accepted response profile is fabricated or shipped.
Without an accepted profile, range sampling retains honestly labelled input
strength/time axes; it cannot claim actual amplitude/speed coverage.

Measured original drives show strong seed dependence: increasing strength
1.0 to 1.5 at time multiplier 1.0 increased residual shape speed from 0.140 to
0.248 m/s for seed 20260804, but decreased it from 0.166 to 0.083 m/s for
20260805. Both are valid full episodes without robot contact. The approved
coarse expanded probe uses strength 0.75/2.0 and time multiplier 0.5/1.5.
Its measured points still do not cover every originally requested actual
amplitude/speed corner. Lack of coverage in a finite probe is not proof that
the physical target is impossible. Extra sweeps need user approval.

The calibration runner has an intentional failure probe, a real short smoke
through the same process-pool path, persistent logs, a supervisor and atomic
status. One complex-initial probe initially failed due to a missing bank path;
the eleven completed records were retained and only that missing job was
rerun using the original range input bank. Diagnostics and physical probes
are excluded from the formal 20M PPO budget. All original output directories
and old Server 2 checkpoints/processes are preserved; no new training has
been launched by this change.

Validation: joint interpolation, actual-output hull rejection, validation guard,
one-dimensional rigid map, three-dimensional combined map, original shape
force active during grasp/off-table, and existing range curriculum unit checks.
Final probe results and the remaining coverage/acceptance decision are recorded
in `outputs/range_motion_amplitude_speed_20261013/`.
