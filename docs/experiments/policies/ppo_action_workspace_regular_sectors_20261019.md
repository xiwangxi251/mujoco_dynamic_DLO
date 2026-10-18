# PPO regular-sector action workspace

The existing `workspace3d_20261010.json` remains the default historical
131-cell map. The regular-sector map is an explicit opt-in asset:
`workspace3d_regular_sectors_20261014.json`. Pass its path with
`--action-workspace` to use it. Training configuration records the selected
map version and SHA256; loading resolves bundled maps by that hash so relocated
checkpoints can still use their original map.

The regular sectors retain the scan's selected layer regions:

| TCP height above table | Radius | Angle |
|---|---:|---:|
| 5 / 50 mm | 0.28–0.52 m | −20° to 80° |
| 100 mm | 0.28–0.52 m | −20° to 60° |
| 150 mm | 0.20–0.52 m | −20° to 60° |
| 200 mm | 0.20–0.44 m | −20° to 40° |

Between measured heights, the map uses the intersection of adjacent sectors.
It does not extrapolate above 200 mm. The scan comprised 1,603 sampled points
and 596 passes; 207 measured points fall inside the selected map, and none of
the recorded failures are admitted. The measurements use fixed downward
orientation, closed jaws and unloaded motion. They do not certify unsampled
interiors, other orientations, open jaws or loaded grasp motion.

The map clips requested XYZ translation before inverse kinematics. It does not
change PPO observations, sampled actions/log probabilities, yaw or gripper
commands, reward, physics, or success criteria. This asset is not enabled by
default and does not silently change existing training configurations.
