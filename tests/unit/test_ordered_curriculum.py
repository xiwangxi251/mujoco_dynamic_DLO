from panda_cable_grasp.rl.ordered_curriculum import GROUPS, MIXES, PHASES, OrderedMotionCurriculum


def curriculum(group=3):
    c = OrderedMotionCurriculum(stages=GROUPS, training_mixes=MIXES)
    c.phase_index = next(i for i, (g, _) in enumerate(PHASES) if g == group)
    return c


def row(name, success=.6, loaded=.8, grasp=.6, episodes=20):
    return dict(scenario=name, episodes=episodes, success_rate=success, loaded_lift_rate=loaded, grasp_rate=grasp)


def test_each_harder_scene_must_pass_not_pooled():
    c = curriculum()
    rows = [row(GROUPS[3][0], 1, 1, 1), row(GROUPS[3][1], .2, .6, .2)]
    assert not c.evaluation_qualifies(dict(targets=rows))
    rows[1] = row(GROUPS[3][1])
    assert c.evaluation_qualifies(dict(targets=rows))


def test_incomplete_or_duplicate_scenes_cannot_pass():
    c = curriculum()
    assert not c.evaluation_qualifies(dict(targets=[row(GROUPS[3][0])]*2))
    assert not c.evaluation_qualifies(dict(targets=[row(n, episodes=19) for n in GROUPS[3]]))


def test_retention_failure_blocks_advance():
    c = curriculum()
    assert not c.evaluation_qualifies(dict(targets=[row(n) for n in GROUPS[3]],
        retention=[row(GROUPS[2][0], success=.3, episodes=10)]))


def test_order_and_replay_mass():
    assert len(PHASES) == 14
    assert PHASES[:2] == ((0, 0.), (1, 0.))
    assert 'rigid' in GROUPS[3][0] and 'shape_nominal_current' in GROUPS[4][0]
    for g, mix in enumerate(MIXES):
        assert abs(sum(p for _, p in mix)-1) < 1e-12
        assert abs(sum(p for n, p in mix if n in GROUPS[g])-(1 if g == 0 else .8)) < 1e-12


def test_two_consecutive_complete_rounds_required():
    c = curriculum(0)
    c._apply = lambda: None
    good = dict(targets=[row(GROUPS[0][0])])
    c.observe_evaluation(good)
    assert c.phase_index == 0
    c.observe_evaluation(dict(targets=[row(GROUPS[0][0], success=.5)]))
    assert c.qualifying_evaluations == 0
    c.observe_evaluation(good)
    c.observe_evaluation(good)
    assert c.phase_index == 1
