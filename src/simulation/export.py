"""Read native executed frames/scores; export proposals, never unverified SFT labels."""
import csv
import math
import pickle
from pathlib import Path

from driversi_world.runtime.files import write_json
from driversi_world.runtime.paths import experiment_path


def export_candidates(native, selection, output):
    output = experiment_path(output)
    data = Path(native) / "openscene_format"
    scores = list(data.glob("all_scenes_pdm_averages_*.csv"))
    if len(scores) != 1:
        raise ValueError("Exactly one declared native traffic-mode score file required")
    wanted = {r["log_name"] + "-" + r["token"]:r for r in selection["records"]}
    with scores[0].open(newline="") as stream:
        raw = [r for r in csv.DictReader(stream) if r["token"] != "overall_average"]
    index = {r["token"]:r for r in raw}
    if len(index) != len(raw) or set(index) != set(wanted):
        raise ValueError("Native score cohort differs from the executed selection")
    candidates = []
    for scene, source in wanted.items():
        path = data / "meta_datas" / (scene + ".pkl")
        with path.open("rb") as handle:
            frames = pickle.load(handle)
        row = index[scene]
        score = float(row["score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Native PDMS must be finite in [0,1]")
        if len(frames) < 12 or any(f["log_name"] != scene for f in frames):
            raise ValueError("Incomplete or mixed-branch native frames")
        qualified_score = (score > .95 and float(row["no_at_fault_collisions"]) == 1
                           and float(row["drivable_area_compliance"]) == 1)
        candidates.append({"source":source, "scene":scene, "frames":str(path),
                           "sensor_root":str(data / "sensor_blobs"), "native_score":row,
                           "score_eligible":qualified_score, "trainable":False,
                           "missing_admission":["paired_teacher_execution", "source_psnr_gt25",
                                                "dynamic_branch_binding", "executed_label_alignment"]})
    card = {"schema":"driversi.worldengine.candidates.v1", "records":candidates,
            "total":len(wanted), "completed":len(candidates),
            "score_eligible":sum(c["score_eligible"] for c in candidates),
            "trainable":False, "parameter_updates":0,
            "scope":"native simulated rollout candidates, not a training release"}
    write_json(output / "candidates.json", card)
    return card
