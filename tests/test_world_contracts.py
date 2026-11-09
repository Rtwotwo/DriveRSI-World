import copy
import csv
import json
import math
import pickle
from pathlib import Path
import numpy as np
import pytest
from driversi_world.data import selection as data
from driversi_world.runtime.paths import experiment_path, runtime_environment
from driversi_world.simulation.export import export_candidates

@pytest.fixture
def sources(tmp_path, monkeypatch):
    log = "training-recording_0_20"
    frames = [dict(token="a" if i == 3 else f"f{i}", log_name=log, timestamp=i*500000)
              for i in range(12)]
    source = tmp_path / "source.pkl"
    source.write_bytes(pickle.dumps(frames))
    train = dict(tokens=["a"], log_names=[log])
    test = dict(tokens=["test"], log_names=["eval-recording_0_20"])
    monkeypatch.setattr(data, "official_filter", lambda split: (None, train if split == "navtrain" else test))
    selection = dict(schema=data.SCHEMA, split="navtrain", query_split="navtrain",
                     records=[dict(token="a", log_name=log, source_path=str(source))])
    return selection, frames


def test_real_source_association_and_protected_recording(sources):
    selection, _ = sources
    assert data.qualify(selection) == ["training-recording"]
    for mutate in (lambda s: s.update(query_split="navtest"),
                   lambda s: s.update(protected_logs=["training-recording_40_60"]),
                   lambda s: s["records"][0].update(token="test"),
                   lambda s: s["records"][0].update(log_name="forged")):
        candidate = copy.deepcopy(selection)
        mutate(candidate)
        with pytest.raises(ValueError):
            data.qualify(candidate)


def test_converter_requires_full_history_and_matching_branch(tmp_path, sources):
    selection, frames = sources
    name = selection["records"][0]["log_name"] + "-a"
    scene = dict(sample_rate=10, metadata=dict(openscene_data_infos_dict=dict(enumerate(frames))))
    path = tmp_path / "scenarios.pkl"
    path.write_bytes(pickle.dumps({name: scene}))
    assert data.inspect_scenarios(path, selection)["completed"] == ["a"]
    scene["object_track"] = {"pedestrian": {"type":"PEDESTRIAN"}}
    path.write_bytes(pickle.dumps({name: scene}))
    rejected = data.inspect_scenarios(path, selection)
    assert not rejected["completed"]
    assert "unsupported_collision_participants" in rejected["rejection_reasons"][name]
    scene.pop("object_track")
    scene["metadata"]["openscene_data_infos_dict"] = dict(enumerate(frames[1:]))
    path.write_bytes(pickle.dumps({name: scene}))
    assert data.inspect_scenarios(path, selection)["missing"] == ["a"]
    path.write_bytes(pickle.dumps({"eval-a": scene}))
    with pytest.raises(ValueError):
        data.inspect_scenarios(path, selection)


def test_exp_only_output_and_cache(tmp_path):
    from driversi_world import __file__ as package_file
    def project_root(): return Path(package_file).resolve().parents[1]
    output = project_root()/"exp/worldengine-test-runtime"
    assert experiment_path(output) == output
    with pytest.raises(ValueError):
        experiment_path(project_root()/"data-output")
    env = runtime_environment(tmp_path)
    assert all(Path(env[key]).is_relative_to(tmp_path) for key in
               ("HF_HOME", "TORCH_HOME", "TRITON_CACHE_DIR", "TMPDIR", "RAY_TMPDIR"))


def test_native_high_score_is_never_an_accepted_label(tmp_path,sources):
    selection, frames = sources
    scene = selection["records"][0]["log_name"] + "-a"
    native = tmp_path/"native/openscene_format"
    (native/"meta_datas").mkdir(parents=True)
    for frame in frames:
        frame["log_name"] = scene
    (native/"meta_datas"/(scene+".pkl")).write_bytes(pickle.dumps(frames))
    score = native/"all_scenes_pdm_averages_NR.csv"
    score.write_text("token,score,no_at_fault_collisions,drivable_area_compliance\n"+scene+",1,1,1\n")
    output = tmp_path/"export"
    output.mkdir()
    card = export_candidates(native.parent, selection, output)
    assert card["score_eligible"] == 1
    assert not card["trainable"] and not card["records"][0]["trainable"]
    assert "paired_teacher_execution" in card["records"][0]["missing_admission"]
    score.write_text("token,score,no_at_fault_collisions,drivable_area_compliance\nother,1,1,1\n")
    with pytest.raises(ValueError):
        export_candidates(native.parent,selection,output)


@pytest.mark.parametrize("score", ["nan", "inf", "60", "-1"])
def test_failure_mining_rejects_bad_score_units(tmp_path,sources,score):
    selection, _ = sources
    index=tmp_path/"index.json"
    index.write_text(json.dumps(selection))
    scores=tmp_path/"scores.csv"
    scores.write_text("token,score,no_at_fault_collisions,drivable_area_compliance\na,"+score+",1,1\n")
    with pytest.raises(ValueError):
        data.mine(scores,index,tmp_path/"mined")


def test_owned_imports_without_clone():
    import driversi_world.simulation as native
    root=Path(native.__file__).parent
    assert "clones" not in root.resolve().parts
    for path in root.rglob("*.py"):
        text=path.read_text()
        assert "from worldengine" not in text and "import worldengine" not in text
    manifest=json.loads((root/"provenance.json").read_text())
    assert "13463cf" in json.dumps(manifest)


