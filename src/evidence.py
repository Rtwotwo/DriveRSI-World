
from pathlib import Path
from driversi_world.runtime.files import digest

def check_files(files):
    for name, sha in files.items():
        if digest(name) != sha:
            raise ValueError("Sealed skill evidence changed: " + name)


def validate_visual_admission(path):
    from driversi_world.quality.diagnostics.worldengine_admission import validate_admission
    return validate_admission(path)
