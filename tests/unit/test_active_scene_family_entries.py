import pytest
from panda_cable_grasp.scenarios.registry import ACTIVE_SCENARIO_FAMILIES, HISTORICAL_SCENARIO_FAMILIES, SCENARIO_FAMILIES
from panda_cable_grasp.rl.workspace import scene_family_argument

def test_daily_families_and_historical_resolution():
    assert set(ACTIVE_SCENARIO_FAMILIES)=={'cable_x075_v1','cable_x075_sector_speed_v3'}
    assert set(HISTORICAL_SCENARIO_FAMILIES)=={'cable_x075_force_v1','cable_x075_surface_v1','cable_x075_sector_speed_v1','cable_x075_sector_speed_v2'}
    for name in SCENARIO_FAMILIES:
        assert scene_family_argument(name)==name
    assert ACTIVE_SCENARIO_FAMILIES['cable_x075_sector_speed_v3'][0] in HISTORICAL_SCENARIO_FAMILIES['cable_x075_sector_speed_v2']
    with pytest.raises(Exception,match='Unknown scene family'):
        scene_family_argument('invalid')

def test_cli_help_hides_archived_families(capsys,monkeypatch):
    import sys
    from panda_cable_grasp.rl.train import parse_args
    monkeypatch.setattr(sys,'argv',['train','--help'])
    with pytest.raises(SystemExit) as result:parse_args()
    assert result.value.code==0
    helptext=capsys.readouterr().out
    assert 'cable_x075_sector_speed_v3' in helptext
    for name in HISTORICAL_SCENARIO_FAMILIES:assert name not in helptext
