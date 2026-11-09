"""Pure file integrity helpers; no policy or training dependency."""
import hashlib
import json
import os
from pathlib import Path

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def path_at(root, value):
    if not isinstance(value, str) or not value or '\x00' in value:
        raise ValueError('Explicit nonempty path required')
    return Path(os.path.abspath(root / value))

def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
