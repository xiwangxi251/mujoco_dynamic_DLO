"""Single-attempt reset diagnostics; never hides rejection in a retry loop."""
import json
import time
from train_targetnode_full6_workspace import CourseEnv, SCENES, save

env = CourseEnv()
rows = []
try:
    for index, scene in enumerate(SCENES):
        env.set_stage(min(index,6))
        mix = [0.] * len(SCENES)
        mix[index] = 1.
        env.env.set_training_scenario_distribution(SCENES, tuple(mix))
        print("RESET_BEGIN", scene, flush=True)
        start = time.monotonic()
        try:
            obs, info = env.env.reset(seed=20261004+index*100)
            row = dict(scene=scene, passed=True, goal=env.env.goal.index,
                       candidates=env.env.goal.candidates.tolist())
        except Exception as error:
            row = dict(scene=scene, passed=False, error=repr(error))
        row.update(wall_s=time.monotonic()-start, simulation_time=float(env.env.data.time))
        rows.append(row)
        save("reset_diagnostics.json", rows)
        print(json.dumps(row), flush=True)
finally:
    env.close()
