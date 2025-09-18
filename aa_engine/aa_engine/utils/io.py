"""I/O helpers — stub."""
from __future__ import annotations
import pathlib, json

def write_manifest(path: str | pathlib.Path, manifest: dict) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2))
