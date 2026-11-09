"""Pinned MTGS pilot assets: integrity, bounded extraction and source inspection.

This module does not download, import MTGS/torch, execute checkpoints, or claim
that a reconstruction is bound to the simulator. Pickle inspection requires an
explicit trust decision even after an archive has passed its checksum.
"""

from __future__ import annotations

import hashlib

from pathlib import Path, PurePosixPath, PureWindowsPath

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
