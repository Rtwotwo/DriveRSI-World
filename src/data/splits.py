"""Exact official NAVSIM roster snapshots, independent of any student model."""
from pathlib import Path
import yaml

def official_filter(split):
    if split not in ('navtrain', 'navtest'):
        raise ValueError('Only official navtrain/navtest splits are supported')
    path = Path(__file__).resolve().parents[1] / 'resources/navsim' / (split + '.yaml')
    value = yaml.safe_load(path.read_text())
    if not value.get('tokens') or not value.get('log_names') or len(value['tokens']) != len(set(value['tokens'])):
        raise ValueError('Exact official split identities required')
    return path, value
