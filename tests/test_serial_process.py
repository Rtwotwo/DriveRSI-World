"""The serial renderer does not depend on optional distributed scheduling."""
import os
import sys

from omegaconf import OmegaConf
from driversi_world.simulation.utils.multithreading.process_utils import resolve_worker_placeholders


def test_worker_placeholder_without_ray(monkeypatch):
    monkeypatch.setitem(sys.modules, 'ray', None)
    cfg = OmegaConf.create({'worker_id_prefix': 'serial_', 'output': '/exp/__WORKER_ID__/frames'})
    resolved = resolve_worker_placeholders(cfg)
    assert resolved.output == f'/exp/serial_{os.getpid()}/frames'
