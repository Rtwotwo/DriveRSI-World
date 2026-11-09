"""Qualify selected NAVSIM training sources before WorldEngine sees them."""
import csv
import json
import math
import pickle
from pathlib import Path

import yaml

from driversi_world.data.recordings import source_log_group
from driversi_world.data.splits import official_filter
from driversi_world.runtime.files import write_json
from driversi_world.runtime.paths import experiment_path

SCHEMA = "driversi.worldengine.selection.v1"


def qualify(selection, protected_groups=()):
    if selection.get("schema") != SCHEMA or selection.get("split") != "navtrain":
        raise ValueError("WorldEngine learning requires explicit navtrain sources")
    if selection.get("query_split") != "navtrain":
        raise ValueError("Test-derived training selection is not enabled by this protocol")
    _, train = official_filter("navtrain")
    _, test = official_filter("navtest")
    tokens, logs = set(train["tokens"]), set(train["log_names"])
    test_tokens = set(test["tokens"])
    protected = {source_log_group(x) for x in test["log_names"]}
    protected.update(source_log_group(x) for x in selection.get("protected_logs", []))
    protected.update(source_log_group(x) for x in protected_groups)
    rows = selection.get("records", [])
    if not rows or len({r["token"] for r in rows}) != len(rows):
        raise ValueError("Selected sources must be nonempty and unique")
    cache = {}
    for row in rows:
        if (row["token"] not in tokens or row["token"] in test_tokens
                or row["log_name"] not in logs or source_log_group(row["log_name"]) in protected):
            raise ValueError("Protected or non-training source: " + row["token"])
        path = Path(row["source_path"]).resolve(strict=True)
        if path not in cache:
            with path.open("rb") as handle:
                frames = pickle.load(handle)  # Operator-owned original NAVSIM logs only.
            if not isinstance(frames, list):
                raise ValueError("Expected original NAVSIM frames")
            cache[path] = {f["token"]: f["log_name"] for f in frames}
        if cache[path].get(row["token"]) != row["log_name"]:
            raise ValueError("Token/log association does not match the actual source file")
    return sorted({source_log_group(r["log_name"]) for r in rows})


def mine(scores, source_index, output, threshold=0.6, limit=2):
    """Rank failures, never confuse missing/NaN scores with hard driving cases."""
    if not 0 < threshold <= 1 or type(limit) is not int or limit < 1:
        raise ValueError("Explicit normalized score threshold and positive limit required")
    index = json.loads(Path(source_index).read_text())
    qualify(index)
    by_token = {r["token"]: r for r in index["records"]}
    failures, seen = [], set()
    with Path(scores).open(newline="") as stream:
        for row in csv.DictReader(stream):
            token = row["token"]
            if token in seen or token not in by_token:
                raise ValueError("Duplicate or out-of-split score token")
            seen.add(token)
            values = {k: float(row[k]) for k in
                      ("score", "no_at_fault_collisions", "drivable_area_compliance")}
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
                raise ValueError("Invalid normalized native score")
            kinds = []
            if values["no_at_fault_collisions"] < 1:
                kinds.append("collision")
            if values["drivable_area_compliance"] < 1:
                kinds.append("offroad")
            if "time_to_collision_within_bound" in row:
                ttc = float(row["time_to_collision_within_bound"])
                if not math.isfinite(ttc) or not 0 <= ttc <= 1:
                    raise ValueError("Invalid TTC metric")
                if ttc < 1:
                    kinds.append("ttc")
            if values["score"] < threshold or kinds:
                failures.append({**by_token[token], "score": values["score"],
                                 "failure_kinds": kinds or ["low_score"]})
    failures.sort(key=lambda r: (r["score"], r["token"]))
    selected, groups = [], set()
    for row in failures:
        group = source_log_group(row["log_name"])
        if group not in groups:
            selected.append(row)
            groups.add(group)
        if len(selected) == limit:
            break
    output = experiment_path(output)
    output.mkdir(parents=True, exist_ok=True)
    value = {**index, "records": selected, "selection_method": "native_failure_score_distinct_recording",
             "threshold": threshold, "scored_count": len(seen), "failure_count": len(failures),
             "population_count": len(index["records"]), "trainable": False}
    write_json(output / "selection.json", value)
    (output / "scene_filter.yaml").write_text(yaml.safe_dump({
        "tokens": [r["token"] for r in selected],
        "log_names": sorted({r["log_name"] for r in selected}),
    }))
    return value


def inspect_scenarios(path, selection):
    """Reject unselected windows, changed lineage and unsupported observation cadence."""
    with Path(path).open("rb") as handle:
        scenes = pickle.load(handle)
    expected = {r["token"]: r for r in selection["records"]}
    covered, rejected, reasons = set(), [], {}
    for name, scene in scenes.items():
        token = name.rsplit("-", 1)[-1]
        if token not in expected or name != expected[token]["log_name"] + "-" + token:
            raise ValueError("Converter emitted a scene outside the registered selection")
        meta = scene["metadata"]
        frames = list(meta["openscene_data_infos_dict"].values())
        unsupported = [key for key, track in scene.get("object_track", {}).items()
                       if str(track.get("type", "")).upper() in ("PEDESTRIAN", "CYCLIST")]
        stamps = [int(f["timestamp"]) for f in frames]
        valid = (scene.get("sample_rate") == 10 and len(frames) >= 12
                 and not unsupported
                 and frames[3]["token"] == token
                 and all(f["log_name"] == expected[token]["log_name"] for f in frames)
                 and all(450000 <= b-a <= 550000 for a,b in zip(stamps, stamps[1:])))
        if valid:
            covered.add(token)
        else:
            rejected.append(name)
            reasons[name] = ("unsupported_collision_participants:" + ",".join(map(str,unsupported))
                             if unsupported else "incomplete_or_misaligned_window")
    return {"planned": sorted(expected), "completed": sorted(covered), "rejected": rejected,
            "missing": sorted(set(expected)-covered), "rejection_reasons": reasons, "trainable": False}
