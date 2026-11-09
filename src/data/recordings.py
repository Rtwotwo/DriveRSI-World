import hashlib
import re
from pathlib import Path

def source_log_group(source_log):
    name = Path(source_log).name
    if name.endswith(".pkl"):
        name = name[:-4]
    return re.sub(r"_\d+_\d+$", "", name)




def recording(name):
    stem = Path(name).name
    if stem.endswith((".pkl", ".db")):
        stem = stem.rsplit(".", 1)[0]
    match = re.fullmatch(r"(\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}_veh-\d+)_\d+_\d+", stem)
    if not match:
        raise ValueError("unknown_recording_identity")
    return match[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
