import numpy as np
from panda_cable_grasp.rl.session_lift import SessionLiftCredit


def test_lift_drop_release_and_lower_regrasp():
    c = SessionLiftCredit([.02], .18)
    values = [c.update([.02], 0), c.update([.05], 0),
              c.update([.04], 0), c.update([.01], None),
              c.update([.01], 0), c.update([.02], 0)]
    assert np.allclose(np.asarray(values)*40, [0, 1.2, -.4, -.8, 0, .4])


def test_release_cycles_cannot_accumulate_lift():
    c = SessionLiftCredit([.01], .18)
    total = 0
    for _ in range(5):
        total += c.update([.01], 0)
        total += c.update([.17], 0)
        total += c.update([.01], None)
    assert abs(total) < 1e-12


def test_ceiling_and_descent_from_above_ceiling():
    c = SessionLiftCredit([.16], .18)
    assert np.isclose(c.update([.17], 0), .01)
    assert np.isclose(c.update([.20], 0), .01)
    assert c.update([.19], 0) == 0
    assert np.isclose(c.update([.17], 0), -.01)
    assert np.isclose(c.update([.17], None), -.01)


def test_node_switch_settles_balance_without_height_jump():
    c = SessionLiftCredit([.02, .12], .18)
    c.update([.02, .12], 0)
    assert np.isclose(c.update([.05, .12], 0), .03)
    assert np.isclose(c.update([.05, .12], 1), -.03)
    assert np.isclose(c.update([.05, .13], 1), .01)
    assert np.isclose(c.update([.05, .13], None), -.01)


def test_spawn_height_is_not_a_persistent_baseline():
    c = SessionLiftCredit([.045], .18)
    c.update([.01], None)
    assert np.isclose(c.update([.02], 0), .01)


def test_no_unbounded_debit_below_acquisition_height():
    c = SessionLiftCredit([.05], .18)
    c.update([.05], 0)
    assert c.update([.01], 0) == 0
    assert c.update([.01], None) == 0
