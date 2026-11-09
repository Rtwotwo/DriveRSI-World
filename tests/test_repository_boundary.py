import ast
import json
from pathlib import Path

import pytest

import driversi_world
from driversi_world.jobs import _seal, verify_receipt
from driversi_world.runtime.paths import experiment_path


def test_no_policy_or_clone_source_imports():
    root=Path(driversi_world.__file__).parent
    for path in root.rglob('*.py'):
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            names=[node.module] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names] if isinstance(node,ast.Import) else []
            assert not any(n and (n=='driversi' or n.startswith(('driversi.','driversi_agents'))) for n in names),path


def test_world_receipt_detects_mutated_evidence(tmp_path):
    output=tmp_path/'native';output.mkdir()
    frame=output/'frame.bin';frame.write_bytes(b'executed-frame')
    _seal(output,'convert','completed',{}, {'scenarios':str(frame)},0)
    assert verify_receipt(output/'receipt.json')['stage']=='convert'
    frame.write_bytes(b'replaced-frame')
    with pytest.raises(ValueError,match='changed'):
        verify_receipt(output/'receipt.json')


def test_output_cannot_escape_exp(tmp_path):
    repo=Path(driversi_world.__file__).resolve().parents[1]
    with pytest.raises(ValueError):experiment_path(repo/'cache')
    link=tmp_path/'escape';link.symlink_to(repo,target_is_directory=True)
    with pytest.raises(ValueError):experiment_path(link/'outside')


def test_official_rosters_independent_of_models():
    from driversi_world.data.splits import official_filter
    _,train=official_filter('navtrain');_,test=official_filter('navtest')
    assert len(train['tokens'])==103288
    assert len(test['tokens'])==12146
    assert not set(train['tokens']) & set(test['tokens'])


@pytest.mark.parametrize('succeeded', [[True,False],[True,True]])
def test_native_completion_survives_later_failure(tmp_path,monkeypatch,succeeded):
    """Fixture simulator: test accounting only, not driving quality."""
    from driversi_world import jobs
    from driversi_world.simulation import execution
    from driversi_world.runtime.files import digest
    selected=tmp_path/'selection.json'
    selected.write_text(json.dumps({'records':[{'token':'a'},{'token':'b'}]}))
    spec=dict(selection=str(selected),selection_sha256=digest(selected),traffic_mode='NR')
    previous=tmp_path/'converted';previous.mkdir()
    scenes=previous/'fixture.pkl';scenes.write_bytes(b'fixture-only')
    _seal(previous,'convert','completed',{'fixture_only':True},{'scenarios':str(scenes)},0,spec=spec)
    spec['input_receipt']=str(previous/'receipt.json')
    monkeypatch.setattr(jobs,'qualify',lambda *a:None)
    monkeypatch.setattr(jobs,'inspect_scenarios',lambda *a:{'missing':[],'rejected':[]})
    def fixture_simulator(config):
        rows=[{'scenario_name':'log-'+token,'succeeded':ok} for token,ok in zip(['a','b'],succeeded)]
        (Path(config['output'])/'runner_report.json').write_text(json.dumps(rows))
    monkeypatch.setattr(execution,'run',fixture_simulator)
    receipt=jobs.run('rollout',spec,tmp_path/'rollout')
    assert receipt['status']=='failed'  # Partial run, or missing native scores.
    assert len(receipt['summary']['completed'])==sum(succeeded)
    assert receipt['summary']['scored']==[]
    assert receipt['trainable'] is False
