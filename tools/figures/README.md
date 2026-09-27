# Task ladder illustrations

## Paper Figure 2: four benchmark motion conditions

Run from the repository root:

```powershell
$env:MUJOCO_GL = 'wgl'
conda run -n mujoco-dlo python tools/figures/render_paper_fig2.py
```

The renderer writes `fig2_four_motion_scenarios.png` and `.pdf` in
`tools/figures/renders/`, plus the four cropped MuJoCo scene images.
Panels use the registered `id_static`, `id_rigid_l1_nominal`,
`id_shape_nominal_current`, and `id_combined_l1_nominal` scenarios with the
same seed and 3.0-second rollout time. The arm and gripper are held at their
environment reset joint positions throughout each rollout; their position
is identical in every panel. Only the cable evolves. The fixed camera is the project's
`dynamicvla_opst_camera`, with its configured position, orientation, and
field of view. The cable keeps the experimental blue. For the figure only,
the tabletop is expanded and its checker texture recolored as pale wood;
the shared crop excludes the remaining sky band. Cast shadows, reflections,
and material specular highlights are disabled. In the three moving scenes, a thin pale
contour shows the actual cable state at 2.0 seconds; the solid cable shows
the state at 3.0 seconds. After each physical rollout, the displayed cable
is translated so its current centroid matches panel (b); the earlier contour
receives the same translation, preserving its displacement relative to the
current cable. The robot and camera are not moved. This is a presentation
alignment, not the raw world position used for benchmark measurements.
In panel (c), both displayed cable contours are interpolated at five samples
per simulated segment to make the tight bends visually continuous; the
simulation states and timestamps are unchanged.
No motion arrows are baked into the image. The
two factors are global motion and internal deformation. The scenes
introduce benchmark conditions; they do not depict successful grasps or
measured policy performance.

## Earlier task ladder illustration

Run from the repository root with the project's MuJoCo environment:

```powershell
$env:MUJOCO_GL = 'wgl'
conda run -n mujoco-dlo python tools/figures/render_task_ladder.py
```

The script writes four 960 × 720 PNG panels and a 1932 × 1452 composite (with white gutters) to `tools/figures/renders/`:

| Panel | Image | Depiction |
| --- | --- | --- |
| (a) | `task_a_static_cube.png` | NERO approaching a stationary cube |
| (b) | `task_b_rolling_ball.png` | NERO approaching a ball; fading copies indicate the intended rolling direction |
| (c) | `task_c_routing.png` | Cable seated in one snap clip, with its free end approaching a second clip |
| (d) | `task_d_dynamic_cable.png` | NERO approaching a cable under combined global motion and shape deformation; a separate four-step timeline shows its changing shape |

These are **illustrative posed scenes**, not grasp or routing rollouts. Panels (a)–(c) use fixed geometry, and the ball's fading copies are a motion cue rather than simulated ball states. Panel (d) uses four actual cable states at 2.5, 3.0, 3.5, and 4.0 seconds from one seeded `combined`-motion environment rollout. The upper scene shows only the final cable pose. The small inset traces the cable centroid's global motion, while the lower timeline aligns centroids so the independent shape deformation is easy to compare. A plain, non-reflective table and suppressed shadows keep the contours legible. The robot is posed by inverse kinematics at the final state. Do not use these panels as evidence of task success or method performance.

The renderer uses the project's NERO meshes and a fixed environment seed for reproducibility. The routing panel uses a closer camera to show both clips clearly. The script requires MuJoCo and Pillow; no experiment checkpoints are needed.

## Rigid object versus deforming cable

The newer MuJoCo rendered version uses a custom, matte tabletop and a
matching elevated camera for both panels. Generate it with:

```powershell
$env:MUJOCO_GL = 'wgl'
conda run -n mujoco-dlo python tools/figures/render_simulated_motion_comparison.py
```

Outputs: `sim_rigid_ball_table.png`, `sim_deforming_cable_table.png`,
`sim_rigid_vs_deforming.png`, and `sim_rigid_vs_deforming.pdf` in
`tools/figures/renders/`. Each panel shows four positions along one
tabletop direction. The ball is unmarked. Cable shapes come from a seeded
`combined` rollout with seed 20283029 at 0.60, 0.68, 0.76, and 0.84 s.
This window keeps the cable open rather than tightly coiled. Its global centroids
are repositioned along the depicted trajectory for clarity, while the
internal contours and their orientation are kept from the rollout. Thus,
the displayed positions are schematic, not measured world coordinates.
These panels illustrate motion types, not performance data.

The earlier flat illustration is described below.

Run the paired paper illustration with:

```powershell
$env:MUJOCO_GL = 'wgl'
conda run -n mujoco-dlo python tools/figures/render_rigid_vs_deforming.py
```

This writes `rigid_vs_deforming.png` and `rigid_vs_deforming.pdf`, plus `rigid_rolling_ball_table.png` and `deforming_cable_table.png`, to `tools/figures/renders/`. Both panels place four states directly on a matching tabletop, with arrows showing time order. The ball orientation marker follows angular velocity integrated from its MuJoCo rollout. The cable contours come from one seeded `combined`-motion rollout and have global translation and planar rotation removed to expose shape change. The four timestamps are 0.0, 1.3, 2.7, and 4.0 seconds. Each state is spatially arranged for legibility; the illustrated tabletop positions and arrows are not measured world trajectories.

This figure contrasts two task types. Because object type and motion process both differ, it is an illustrative task comparison rather than a controlled deformation ablation or a performance result. The controlled rigid-versus-combined DLO benchmark should use the same cable in both conditions.

## Paper schematics

For a concept figure, generate the simplified vector diagrams:

```powershell
conda run -n mujoco-dlo python tools/figures/draw_task_schematics.py
```

The output directory is `tools/figures/schematics/`. It contains four standalone panels (`a_static_object` through `d_dynamic_cable`), a horizontal `task_schematics_4panel` layout, and `task_schematics_2x2`. Each is exported as SVG, PDF, and 400 dpi PNG. SVG and PDF are suitable for paper layout; the PNGs are convenient previews. The diagrams are conceptual and do not depict measured trajectories or successful grasps.

Panel C shows a cable already seated in one snap clip and approaching the next. Panel D uses a pale prior shape and two arrows to distinguish global object motion from internal shape change; these are independent benchmark factors, not a measured decomposition of a particular trial.
