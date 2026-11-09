"""All generated artifacts belong to this checkout's exp directory."""
import os
import sys
from pathlib import Path


def experiment_path(path):
    repo = Path(__file__).resolve().parents[2]
    base = repo / "exp"
    if base.resolve() != base:
        raise ValueError("exp must be a physical directory within the repository")
    directory = Path(path).expanduser().resolve()
    if directory == base or not directory.is_relative_to(base):
        raise ValueError("Generated files must be in a child directory of repository/exp")
    return directory


def run_root():
    repo = Path(__file__).resolve().parents[2]
    return experiment_path(os.environ.get("DRIVERSI_WORLD_RUN_ROOT", repo / "exp/runs"))


def runtime_environment(output):
    output = experiment_path(output)
    cache = output / "cache"
    tmp = output / "tmp"
    for folder in (cache, tmp):
        folder.mkdir(parents=True, exist_ok=True)
    return {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(cache / "python"),
        "XDG_CACHE_HOME": str(cache),
        "HF_HOME": str(cache / "huggingface"),
        "TORCH_HOME": str(cache / "torch"),
        "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "NUMBA_CACHE_DIR": str(cache / "numba"),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "TMPDIR": str(tmp), "TMP": str(tmp), "TEMP": str(tmp),
        "RAY_TMPDIR": str(tmp / "ray"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    }


def configure_runtime(output):
    values = runtime_environment(output)
    os.environ.update(values)
    sys.dont_write_bytecode = True
    return values
