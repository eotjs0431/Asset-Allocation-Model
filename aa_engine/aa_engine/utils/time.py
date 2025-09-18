"""Time utilities — stub."""
from __future__ import annotations
from datetime import datetime
def utcnow_iso() -> str:
    return datetime.utcnow().isoformat()
