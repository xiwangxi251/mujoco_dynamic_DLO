# Static placement v2: strict 30% coverage and varied yaw

The user requested both static scenes to require only30% of nodes in sectorB
and more diverse initial orientations. New family `cable_x075_sector_speed_v2`
keeps all seven categories and changes only ordinary/complex-initial static
placement. The previous v1 names/identities and existing training/video sources
remain available. Other five scenarios have identical environment overrides.

`sector_fraction_yaw_v2` draws one uniform yaw in[-180,180) degrees per episode
before translation proposals. This yaw is held fixed while finding a placement;
failed placement must not select a new yaw or reduce the30% threshold. The
existing2048 node-anchor/area-uniform proposals are then applied to the rotated
geometry, with both minimum and fallback fractions0.3. If none meet30%, reset
fails rather than accepting a smaller overlap. No outcome-based filtering or
forced left/right quota is added.

Rotation applies to the entire cable through its free-root world position and
quaternion, preserving internal joints, relative geometry, heights and the
sampled snapshot. Rotation is about the current node mean. The entire cable
must stay on the table, with its radius margin. Runtime verifies that actual
MuJoCo body positions match the intended rigid transform, and records the
sampled yaw, coverage and used threshold. Seeded placement uses the original
independent generator(seed xor0x5EC70); legacy generators are unchanged.

SectorB retains center(0.35,0)m, radii0.28–0.56m, angles−30°–70°. Motion physics,
forces/gates, shape drivers, speed ranges, controllers, rewards and success
definitions are unchanged. This work does not switch/restart the running151
v1 PPO or rewrite previously recorded v1 videos. Changes stay uncommitted in
`experiment/static-sector-30-yaw-20261014` for user acceptance.

Validation uses50 existing paired seeds per static scene, first as pure geometry
then100 native reset/forward checks with zero simulation time. It checks strict
coverage, whole-table containment, geometry/height preservation, reproducibility,
orientation coverage and distribution. This is not a new policy success test.
Results: `outputs/static_sector30_yaw_20261014/validation.json` at workspace root.
The finite50-seed distribution is descriptive; uniform yaw does not promise
exact25/25 left/right counts or exact realized coverage in every episode.
